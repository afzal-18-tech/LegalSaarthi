"""
Unit tests for AbstentionCalibrator and AbstentionPolicy (Phase 17).
"""
import pytest
from src.evaluation.abstention_calibrator import AbstentionCalibrator, AbstentionPolicy, SweepPoint


def test_abstention_policy_threshold_bounds():
    policy = AbstentionPolicy(threshold=0.5)
    assert policy.threshold == 0.5

    with pytest.raises(ValueError):
        AbstentionPolicy(threshold=-0.1)

    with pytest.raises(ValueError):
        AbstentionPolicy(threshold=1.1)


def test_abstention_policy_decisions():
    policy = AbstentionPolicy(threshold=0.5)

    # 1. Generation failure triggers abstention
    res, reason = policy.decide({"generation_failed": True})
    assert res is True
    assert reason == "generation_failed"

    # 2. Explicit model refusal triggers abstention
    res, reason = policy.decide({
        "generation_failed": False,
        "verification": {"abstention_reason": "model_abstained", "support_rate": 1.0},
    })
    assert res is True
    assert reason == "model_abstained"

    # 3. Low support rate triggers abstention
    res, reason = policy.decide({
        "generation_failed": False,
        "verification": {"support_rate": 0.4},
    })
    assert res is True
    assert "support_rate=0.40 < threshold=0.50" in reason

    # 4. Adequate support rate allows answering
    res, reason = policy.decide({
        "generation_failed": False,
        "verification": {"support_rate": 0.6},
    })
    assert res is False
    assert reason is None


def test_calibrator_evaluate_point():
    dummy_records = [
        {
            "query_id": "Q1",
            "split": "val",
            "verification": {"support_rate": 0.8},
            "evaluation": {"correctness": 5, "factuality": 5},
        },
        {
            "query_id": "Q2",
            "split": "val",
            "verification": {"support_rate": 0.4},
            "evaluation": {"correctness": 3, "factuality": 3},
        },
        {
            "query_id": "Q3",
            "split": "val",
            "verification": {"support_rate": 0.2, "abstention_reason": "model_abstained"},
            "evaluation": {"correctness": 0, "factuality": 0},
        },
    ]

    calibrator = AbstentionCalibrator(dummy_records, thresholds=[0.0, 0.5, 0.9])

    # At tau = 0.5: Q1 answered (rate 0.8 >= 0.5), Q2 abstained (rate 0.4 < 0.5), Q3 abstained (model_abstained)
    pt = calibrator.evaluate_point(calibrator.processed_records, 0.5)
    assert pt.total == 3
    assert pt.answered == 1
    assert pt.abstained == 2
    assert pt.coverage_rate == pytest.approx(1 / 3, rel=1e-2)
    assert pt.selective_correctness == 5.0
    assert pt.selective_factuality == 5.0
    assert pt.effective_correctness == pytest.approx(5 / 3, rel=1e-2)
    assert pt.effective_factuality == pytest.approx(5 / 3, rel=1e-2)
    assert pt.mean_claim_support == 0.8


def test_calibrator_raw_eval_overrides():
    dummy_records = [
        {
            "query_id": "Q_ABSTAINED",
            "split": "val",
            "verification": {"support_rate": 0.2},
            "evaluation": {"correctness": 0, "factuality": 0},  # originally scored as 0
        }
    ]
    # Override provides the raw judged score if un-abstained
    overrides = {"Q_ABSTAINED": {"correctness": 4, "factuality": 4}}
    calibrator = AbstentionCalibrator(dummy_records, thresholds=[0.0, 0.5], raw_eval_overrides=overrides)

    # At tau = 0.0 (no threshold abstention): Q_ABSTAINED should receive overridden raw score
    pt_0 = calibrator.evaluate_point(calibrator.processed_records, 0.0)
    assert pt_0.answered == 1
    assert pt_0.selective_correctness == 4.0
    assert pt_0.selective_factuality == 4.0

    # At tau = 0.5: Q_ABSTAINED abstains
    pt_5 = calibrator.evaluate_point(calibrator.processed_records, 0.5)
    assert pt_5.answered == 0
    assert pt_5.abstained == 1


def test_calibrator_pareto_frontier():
    # Point A: high coverage (0.9), moderate factuality (4.0)
    # Point B: lower coverage (0.5), higher factuality (4.5) -> Both are on Pareto frontier
    # Point C: lower coverage (0.4), lower factuality (3.5) -> Dominated by A and B
    dummy_records = [
        {"query_id": f"Q{i}", "split": "val", "verification": {"support_rate": 0.8}, "evaluation": {"correctness": 5, "factuality": 5}}
        for i in range(10)
    ]
    calibrator = AbstentionCalibrator(dummy_records, thresholds=[0.0, 0.5, 1.0])
    pts = calibrator.run_sweep(calibrator.processed_records)
    # For identical records all at 0.8, at 0.0 & 0.5 coverage is 1.0, factuality is 5.0.
    # At 1.0, coverage is 0.0, factuality is 0.0 (dominated).
    assert pts[0].is_pareto is True
    assert pts[-1].is_pareto is False


def test_calibration_split_protection():
    # 2 val records, 2 test records
    dummy_records = [
        {"query_id": "V1", "split": "val", "verification": {"support_rate": 0.6}, "evaluation": {"correctness": 5, "factuality": 5}},
        {"query_id": "V2", "split": "val", "verification": {"support_rate": 0.3}, "evaluation": {"correctness": 2, "factuality": 2}},
        {"query_id": "T1", "split": "test", "verification": {"support_rate": 0.7}, "evaluation": {"correctness": 5, "factuality": 5}},
        {"query_id": "T2", "split": "test", "verification": {"support_rate": 0.4}, "evaluation": {"correctness": 3, "factuality": 3}},
    ]
    calibrator = AbstentionCalibrator(dummy_records, thresholds=[0.0, 0.5, 0.8])
    res = calibrator.calibrate()

    assert res["val_count"] == 2
    assert res["test_count"] == 2
    assert res["total_count"] == 4
    assert res["recommended_tau"] in [0.0, 0.5, 0.8]
    assert res["frozen_val_result"] is not None
    assert res["frozen_test_result"] is not None
    assert res["frozen_val_result"]["threshold"] == res["recommended_tau"]
    assert res["frozen_test_result"]["threshold"] == res["recommended_tau"]
    assert "balanced" in res["profiles"]
    assert res["profiles"]["balanced"]["threshold"] == 0.5
