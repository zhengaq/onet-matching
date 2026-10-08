import numpy as np
import pandas as pd
import pytest

from onet.matching import (MatchConfig, Scorer, assign_status, compute_rare_tokens,
                           exact_match_table, match, review_queue_table)

CONFIG = MatchConfig()

ONET = pd.DataFrame({
    "onet_code": ["29-1141.00", "47-2111.00", "11-9199.00"],
    "title": ["registered nurses", "electricians", "managers all other"],
    "doc_text": ["registered nurses; registered nurse; staff nurse; charge nurse",
                 "electricians; electrician; wireman; electrical contractor",
                 "managers all other; managers all other; program manager"],
})


def scorer(rare=()):
    return Scorer(CONFIG, set(rare))


def test_exact_alias_boost():
    score, evidence, alias = scorer().score("staff nurse", "registered nurses",
                                            ONET.doc_text[0], 0.70)
    assert alias and evidence == "exact_alias"
    assert score == pytest.approx(0.85)


def test_generic_penalty_only_below_cosine_cutoff():
    low, evidence, _ = scorer().score("boss", "managers all other", ONET.doc_text[2], 0.60)
    assert evidence == "generic_penalty" and low == pytest.approx(0.50)
    high, evidence, _ = scorer().score("boss", "managers all other", ONET.doc_text[2], 0.80)
    assert evidence == "cosine_only" and high == pytest.approx(0.80)


def test_rare_token_boost():
    score, evidence, _ = scorer({"wireman"}).score(
        "journeyman wireman", "electricians", ONET.doc_text[1], 0.60)
    assert evidence == "rare_tokens:wireman" and score == pytest.approx(0.65)


def test_equipment_keywords_match_substrings():
    # Pins current behaviour: "ct" fires inside "director" and "contractor".
    _, evidence, _ = scorer().score("director", "electricians", ONET.doc_text[1], 0.60)
    assert evidence == "equipment_keywords:ct"


def test_rare_tokens():
    docs = ["alpha beta"] * 30 + ["alpha gamma"]
    assert compute_rare_tokens(docs, 0.05) == {"gamma"}


@pytest.mark.parametrize("score, margin, alias, expected", [
    (0.85, 0.20, False, "AUTO_ACCEPT"),
    (0.85, 0.10, False, "NEEDS_REVIEW"),
    (0.70, 0.12, True, "AUTO_ACCEPT"),
    (0.70, 0.05, True, "NEEDS_REVIEW"),
    (0.40, 0.30, False, "NO_MATCH"),
    (0.40, 0.30, True, "NO_MATCH"),
])
def test_status_rules(score, margin, alias, expected):
    best = pd.DataFrame([{"score_final": score,
                          "margin": margin, "exact_alias_match": alias}])
    assert assign_status(best, CONFIG).iloc[0] == expected


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_review_files():
    people = pd.DataFrame({
        "participant_id": ["002", "001", "003"],
        "job_title_raw": ["Staff nurse", "cable installer, IT", ""],
        "job_title_clean": ["staff nurse", "cable installer information technology", "[empty]"],
        "industry": ["Hospital", "construction", ""],
        "jobDescription": ["ward care", "wiring, HR", ""],
    })
    onet_emb = np.eye(3, dtype=np.float32)
    people_emb = np.stack([unit(1, 0.1, 0.1), unit(0.5, 0.6, 0.4)])
    titled = people.iloc[:2]
    cfg = MatchConfig(top_k=3, final_k=3)

    matches = match(titled.participant_id.tolist(), titled.job_title_clean.tolist(),
                    people_emb, ONET, onet_emb, Scorer(cfg, set()))
    assert matches.groupby("participant_id").size().tolist() == [3, 3]
    best = matches[matches["rank"] == 1].reset_index(drop=True)
    best["match_status"] = assign_status(best, cfg)
    assert best.set_index("participant_id").match_status.to_dict() == {
        "002": "AUTO_ACCEPT", "001": "NEEDS_REVIEW"}

    exact = exact_match_table(best, people)
    assert exact.participant_id.tolist() == ["002"]
    assert exact.loc[0, ["job_title_raw", "job_title_clean", "industry"]].tolist() == [
        "Staff nurse", "staff nurse", "Hospital"]

    queue = review_queue_table(best, matches, people, cfg.final_k)
    assert queue.participant_id.tolist() == ["001", "003"]
    assert queue.match_status.tolist() == ["NEEDS_REVIEW", "NA_INPUT"]
    # Reviewers see the answers as written; the cleaned title is a separate column.
    assert queue.loc[0, ["cand1_code", "job_title_raw", "job_title_clean", "industry",
                         "jobDescription"]].tolist() == [
        "47-2111.00", "cable installer, IT", "cable installer information technology",
        "construction", "wiring, HR"]
    assert queue.loc[1, ["job_title_raw", "job_title_clean"]].tolist() == ["", "[empty]"]
    assert pd.isna(queue.loc[1, "cand1_code"])
    assert list(queue.columns[:8]) == [
        "participant_id", "job_title_raw", "job_title_clean", "industry",
        "jobDescription", "best_score_final", "margin", "match_status"]
