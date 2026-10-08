"""Match cleaned job titles to O*NET occupations and write the review files.

Each title is compared with every O*NET document by cosine similarity. The
top_k nearest are rescored with the heuristics in Scorer, re-ranked, and the
best final_k kept. The rank-1 score and its margin over rank 2 decide the
match status (assign_status).
"""

import argparse
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from onet.config import load_settings, require
from onet.embedding import load_embeddings
from onet.survey import EMPTY, read_clean

GENERIC_TITLES = frozenset({
    "administrator", "analyst", "assistant", "associate", "consultant",
    "coordinator", "director", "manager", "officer", "representative",
    "specialist", "supervisor",
})

EQUIPMENT_KEYWORDS = frozenset({
    # medical and diagnostic
    "ultrasound", "mri", "ct", "xray", "x-ray", "ecg", "ekg",
    "dialysis", "ventilator", "defibrillator", "catheter",
    # manufacturing and industrial
    "laser", "cnc", "lathe", "mill", "welder", "plasma",
    "forklift", "crane", "excavator", "bulldozer",
    # laboratory
    "microscope", "centrifuge", "spectroscope", "autoclave",
    # technology and electronics
    "router", "server", "mainframe", "oscilloscope", "multimeter",
    # construction
    "backhoe", "grader", "compressor", "jackhammer",
    # specialized
    "photonics", "fiber optic", "semiconductor", "turbine",
})

STATUS_ORDER = {"NEEDS_REVIEW": 0, "NO_MATCH": 1, "NA_INPUT": 2}
# Shown to reviewers beside the candidates. The answers are verbatim; the
# cleaned title is what was matched.
CONTEXT_COLUMNS = ["job_title_raw", "job_title_clean", "industry", "jobDescription"]


@dataclass(frozen=True)
class MatchConfig:
    """Scoring and status settings. Changing any of them changes the results."""

    boost_exact_alias: float = 0.15
    boost_equipment_keyword: float = 0.10
    boost_rare_token: float = 0.05  # applied only without an equipment boost
    penalty_generic: float = 0.10
    generic_penalty_max_cosine: float = 0.75  # penalty applies below this cosine

    threshold_no_match: float = 0.50
    threshold_auto_accept_score: float = 0.80
    threshold_auto_accept_margin: float = 0.15
    threshold_exact_alias_margin: float = 0.10

    rare_token_threshold: float = 0.05  # share of O*NET documents
    top_k: int = 20  # nearest documents rescored
    final_k: int = 5  # candidates kept per participant

    generic_titles: frozenset = GENERIC_TITLES
    equipment_keywords: frozenset = EQUIPMENT_KEYWORDS


def tokenize(text):
    """Lowercase alphabetic tokens of three or more letters."""
    if pd.isna(text):
        return set()
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))


def compute_rare_tokens(doc_texts, threshold):
    """Tokens found in fewer than `threshold` of the documents."""
    counts = Counter()
    for doc in doc_texts:
        counts.update(tokenize(doc))
    n = len(doc_texts)
    return {token for token, c in counts.items() if c / n < threshold}


def is_blank(text):
    return pd.isna(text) or str(text).strip() == '' or str(text).strip().lower() == EMPTY


class Scorer:
    """Heuristic adjustments to the cosine score of one candidate."""

    def __init__(self, config, rare_tokens):
        self.config = config
        self.rare_tokens = rare_tokens

    def exact_alias(self, title, doc):
        """The title equals one of the document's ';'-separated variants."""
        if pd.isna(title) or pd.isna(doc):
            return False
        variants = [v.strip().lower() for v in str(doc).split(';')]
        return str(title).lower().strip() in variants

    def equipment_overlap(self, title, doc):
        # Substring test, so short keywords also fire inside longer words
        # ("ct" in "director"). Kept as is: changing it changes candidates.
        if pd.isna(title) or pd.isna(doc):
            return []
        title, doc = str(title).lower(), str(doc).lower()
        return sorted(k for k in self.config.equipment_keywords if k in title and k in doc)

    def rare_overlap(self, title, doc):
        return sorted(tokenize(title) & tokenize(doc) & self.rare_tokens)

    def is_generic(self, onet_title):
        return any(g in onet_title.lower() for g in self.config.generic_titles)

    def score(self, title, onet_title, doc, cosine):
        """Return (final score, evidence string, exact alias flag)."""
        cfg = self.config
        score, evidence = cosine, []

        alias = self.exact_alias(title, doc)
        if alias:
            score += cfg.boost_exact_alias
            evidence.append("exact_alias")

        equipment = self.equipment_overlap(title, doc)
        if equipment:
            score += cfg.boost_equipment_keyword
            evidence.append(f"equipment_keywords:{','.join(equipment[:3])}")
        else:
            rare = self.rare_overlap(title, doc)
            if rare:
                score += cfg.boost_rare_token
                evidence.append(f"rare_tokens:{','.join(rare[:3])}")

        if self.is_generic(onet_title) and cosine < cfg.generic_penalty_max_cosine:
            score -= cfg.penalty_generic
            evidence.append("generic_penalty")

        return score, "; ".join(evidence) or "cosine_only", alias


def match(ids, titles, embeddings, onet, onet_embeddings, scorer):
    """Long table of the final_k candidates per participant, best first."""
    cfg = scorer.config
    codes, names, docs = onet.onet_code.tolist(), onet.title.tolist(), onet.doc_text.tolist()
    similarity = embeddings @ onet_embeddings.T  # embeddings are L2-normalized

    rows = []
    for i, (pid, title) in enumerate(tqdm(list(zip(ids, titles)), desc="Matching")):
        cosine = similarity[i]
        candidates = []
        for idx in np.argsort(cosine)[-cfg.top_k:][::-1]:
            c = float(cosine[idx])
            final, evidence, alias = scorer.score(title, names[idx], docs[idx], c)
            candidates.append({"onet_code": codes[idx], "onet_title": names[idx],
                               "score_cosine": c, "score_final": final,
                               "exact_alias_match": alias, "evidence": evidence})
        candidates.sort(key=lambda c: c["score_final"], reverse=True)  # stable
        candidates = candidates[:cfg.final_k]
        margin = (candidates[0]["score_final"] - candidates[1]["score_final"]
                  if len(candidates) >= 2 else 0.0)
        for rank, cand in enumerate(candidates, start=1):
            rows.append({"participant_id": pid, "job_title_clean": title, "rank": rank,
                         **cand, "margin": margin})
    return pd.DataFrame(rows)


def assign_status(best, config):
    """Match status from the rank-1 candidate.

    1. score < threshold_no_match -> NO_MATCH
    2. score >= threshold_auto_accept_score and margin >= threshold_auto_accept_margin -> AUTO_ACCEPT
    3. exact alias and margin >= threshold_exact_alias_margin -> AUTO_ACCEPT
    4. otherwise -> NEEDS_REVIEW
    """
    def status(r):
        if r.score_final < config.threshold_no_match:
            return "NO_MATCH"
        if (r.score_final >= config.threshold_auto_accept_score
                and r.margin >= config.threshold_auto_accept_margin):
            return "AUTO_ACCEPT"
        if r.exact_alias_match and r.margin >= config.threshold_exact_alias_margin:
            return "AUTO_ACCEPT"
        return "NEEDS_REVIEW"

    return best.apply(status, axis=1)


def exact_match_table(best, people):
    """One row per auto-accepted participant."""
    accepted = best[best.match_status == "AUTO_ACCEPT"]
    table = accepted[["participant_id", "onet_code", "onet_title",
                      "score_final", "margin"]].rename(columns={
                          "onet_code": "best_onet_code", "onet_title": "best_onet_title",
                          "score_final": "best_score_final"})
    table = table.merge(people[["participant_id", *CONTEXT_COLUMNS]], on="participant_id")
    table = table[["participant_id", "job_title_raw", "job_title_clean", "best_onet_code",
                   "best_onet_title", "best_score_final", "margin", "industry", "jobDescription"]]
    return table.sort_values("participant_id").reset_index(drop=True)


def review_queue_table(best, matches, people, final_k):
    """One row per participant not auto-accepted, candidates side by side.

    NEEDS_REVIEW first, then NO_MATCH, then NA_INPUT (no job title).
    """
    context = people.set_index("participant_id")
    candidates = {pid: g for pid, g in matches.groupby("participant_id", sort=False)}
    rows = []
    for r in best[best.match_status != "AUTO_ACCEPT"].itertuples():
        row = {"participant_id": r.participant_id,
               "best_score_final": r.score_final, "margin": r.margin,
               "match_status": r.match_status}
        for c in candidates[r.participant_id].itertuples():
            row.update({f"cand{c.rank}_code": c.onet_code,
                        f"cand{c.rank}_title": c.onet_title,
                        f"cand{c.rank}_score": c.score_final})
        rows.append(row)
    for p in people[people.job_title_clean.map(is_blank)].itertuples():
        rows.append({"participant_id": p.participant_id, "match_status": "NA_INPUT"})

    columns = ["participant_id", *CONTEXT_COLUMNS,
               "best_score_final", "margin", "match_status"]
    for rank in range(1, final_k + 1):
        columns += [f"cand{rank}_code", f"cand{rank}_title", f"cand{rank}_score"]

    queue = pd.DataFrame(rows, columns=columns)
    for col in CONTEXT_COLUMNS:
        queue[col] = queue.participant_id.map(context[col])
    order = queue.match_status.map(STATUS_ORDER)
    return (queue.assign(_order=order).sort_values(["_order", "participant_id"])
            .drop(columns="_order").reset_index(drop=True))


# --- Report -----------------------------------------------------------------

def _quantiles(series):
    qs = [("Min", series.min()), ("25th percentile", series.quantile(0.25)),
          ("Median", series.quantile(0.5)), ("75th percentile", series.quantile(0.75)),
          ("90th percentile", series.quantile(0.9)), ("Max", series.max())]
    return [f"| {name} | {value:.4f} |" for name, value in qs]


def write_report(path, settings, config, people, best, exact, queue, n_onet, rare_tokens):
    present = {label: int((~people[col].map(is_blank)).sum())
               for label, col in [("a job title", "job_title_clean"), ("an industry", "industry"),
                                  ("duties", "jobDescription")]}
    statuses = ["AUTO_ACCEPT", "NEEDS_REVIEW", "NO_MATCH"]
    n_na = int((queue.match_status == "NA_INPUT").sum())

    lines = [
        f"# O*NET matching report: {settings.survey_name}",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{settings.survey_file.name}`.",
        "",
        "## 1. Inputs",
        "",
        "| | Count |",
        "|---|---:|",
        f"| Participants | {len(people)} |",
        *[f"| With {label} | {n} |" for label, n in present.items()],
        f"| O*NET occupations | {n_onet} |",
        "",
        "## 2. Match status",
        "",
        "| Status | Participants |",
        "|---|---:|",
    ]
    for s in statuses:
        lines.append(f"| {s} | {int((best.match_status == s).sum())} |")
    lines += [f"| NA_INPUT (no job title) | {n_na} |", "",
              "Rules, applied in order to the rank-1 candidate:", "",
              *[line.strip() for line in assign_status.__doc__.splitlines()[2:] if line.strip()],
              ""]

    lines += ["## 3. Rank-1 scores", "", "| score_final | |", "|---|---:|",
              *_quantiles(best.score_final), "",
              "| margin over rank 2 | |", "|---|---:|", *_quantiles(best.margin), "",
              f"Exact alias on rank 1: {int(best.exact_alias_match.sum())} of {len(best)}.", ""]

    evidence = Counter(part.split(':')[0] for e in best.evidence for part in e.split('; '))
    lines += ["## 4. Evidence on the rank-1 candidate", "", "| Evidence | Count |", "|---|---:|",
              *[f"| {k} | {v} |" for k, v in evidence.most_common()], ""]

    lines += ["## 5. Auto-accepted occupations", "",
              "Auto-accepted matches are not reviewed, so check these titles, shown as "
              "the participants wrote them.", "",
              "| Code | Occupation | Count | Job titles |", "|---|---|---:|---|"]
    for (code, title), g in exact.groupby(["best_onet_code", "best_onet_title"]):
        titles = sorted({" ".join(t.split()).replace("|", "\\|") for t in g.job_title_raw})
        shown = ", ".join(titles[:10]) + (f", ... ({len(titles)} distinct)" if len(titles) > 10 else "")
        lines.append(f"| {code} | {title} | {len(g)} | {shown} |")
    lines.append("")

    lines += ["## 6. Settings", "",
              f"- Boosts: exact alias +{config.boost_exact_alias}, equipment keyword "
              f"+{config.boost_equipment_keyword}, rare token +{config.boost_rare_token} "
              "(only without an equipment boost)",
              f"- Generic-title penalty: -{config.penalty_generic} when cosine < "
              f"{config.generic_penalty_max_cosine}; terms: {', '.join(sorted(config.generic_titles))}",
              f"- Equipment keywords: {', '.join(sorted(config.equipment_keywords))}",
              f"- Thresholds: NO_MATCH below {config.threshold_no_match}; AUTO_ACCEPT at "
              f"score >= {config.threshold_auto_accept_score} with margin >= "
              f"{config.threshold_auto_accept_margin}, or exact alias with margin >= "
              f"{config.threshold_exact_alias_margin}",
              f"- Rare tokens: {len(rare_tokens)} (in < {config.rare_token_threshold:.0%} of documents)",
              f"- Candidates: top {config.top_k} by cosine rescored, {config.final_k} kept", ""]

    lines += ["## 7. Files", "",
              f"- `{settings.output_file('exact_match.csv').name}`: {len(exact)} auto-accepted participants",
              f"- `{settings.output_file('review_queue.csv').name}`: {len(queue)} participants to review", ""]
    path.write_text("\n".join(lines))


# --- Pipeline ---------------------------------------------------------------

def run(settings, config=MatchConfig()):
    require(settings.onet_docs, settings.survey_clean)
    start = time.time()

    onet = pd.read_csv(settings.onet_docs, dtype=str, keep_default_na=False)
    onet_embeddings = load_embeddings(settings.onet_embeddings, onet.onet_code, onet.doc_text)
    people = read_clean(settings.survey_clean)
    embeddings = load_embeddings(settings.survey_embeddings, people.participant_id,
                                 people.job_title_clean)
    print(f"{len(people)} participants, {len(onet)} O*NET occupations")

    titled = ~people.job_title_clean.map(is_blank).to_numpy()
    rare_tokens = compute_rare_tokens(onet.doc_text.tolist(), config.rare_token_threshold)
    print(f"{titled.sum()} with a job title, {(~titled).sum()} without; "
          f"{len(rare_tokens)} rare O*NET tokens")
    if not titled.any():
        raise SystemExit("No participant has a job title to match.")

    matched = people[titled]
    matches = match(matched.participant_id.tolist(), matched.job_title_clean.tolist(),
                    embeddings[titled], onet, onet_embeddings, Scorer(config, rare_tokens))

    best = matches[matches["rank"] == 1].reset_index(drop=True)
    best["match_status"] = assign_status(best, config)

    exact = exact_match_table(best, people)
    queue = review_queue_table(best, matches, people, config.final_k)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    exact.to_csv(settings.output_file("exact_match.csv"), index=False)
    queue.to_csv(settings.output_file("review_queue.csv"), index=False)
    write_report(settings.output_file("report.md"), settings, config, people, best,
                 exact, queue, len(onet), rare_tokens)

    print("\nMatch status:")
    for status, n in pd.concat([best.match_status, queue.match_status[
            queue.match_status == "NA_INPUT"]]).value_counts().items():
        print(f"  {status:<13}{n:>5}")
    print(f"\nWrote {len(exact)} auto-accepted, {len(queue)} to review "
          f"({time.time() - start:.1f}s):")
    for kind in ("exact_match.csv", "review_queue.csv", "report.md"):
        print(f"  {settings.output_file(kind)}")


def main(argv=None):
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    run(load_settings())
