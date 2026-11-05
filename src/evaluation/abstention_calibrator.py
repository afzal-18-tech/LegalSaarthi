"""
LegalSaarthi — Claim-Level Abstention Policy Calibrator (Phase 17)

Calibrates the abstention policy threshold (tau) to optimize the Pareto frontier
between answer coverage (1 - abstention_rate) and selective factuality/correctness.

Research Hypotheses & Methodology:
  H3: Explicit claim verification reduces unsupported claims. When support_rate < tau,
      the system abstains rather than producing potentially misleading legal advice.
  Integrity Protection (AGENTS.md Rule 4):
      Optimal threshold tau* is selected strictly using the VALIDATION set.
      The frozen policy is then evaluated on the LOCKED TEST set without post-hoc tuning.
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SweepPoint:
    threshold: float
    total: int
    answered: int
    abstained: int
    coverage_rate: float
    abstention_rate: float
    selective_correctness: float  # on answered only (scale 1-5)
    selective_factuality: float   # on answered only (scale 1-5)
    effective_correctness: float  # across all (abstain=0)
    effective_factuality: float   # across all (abstain=0)
    mean_claim_support: float     # on answered only (scale 0-1)
    utility_f05: float            # F_0.5 balancing coverage (0.2) and factuality (0.8)
    is_pareto: bool = False


@dataclass
class SplitCalibrationResult:
    split: str
    n_queries: int
    points: List[SweepPoint]
    optimal_tau: float
    pareto_points: List[SweepPoint]


class AbstentionPolicy:
    """Evaluates whether a generated response should be delivered or abstained."""

    def __init__(self, threshold: float = 0.5):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        self.threshold = threshold

    def decide(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Decides whether to abstain on a prediction record.
        Returns (should_abstain, reason).
        """
        # 1. Generation failure
        if record.get("generation_failed"):
            return True, "generation_failed"

        verification = record.get("verification", {})
        # 2. Raw model itself refused / declared lack of evidence
        if verification.get("abstention_reason") == "model_abstained":
            return True, "model_abstained"

        # 3. Check support rate against threshold
        support_rate = verification.get("support_rate", 0.0)
        if support_rate < self.threshold:
            return True, f"support_rate={support_rate:.2f} < threshold={self.threshold:.2f}"

        return False, None


class AbstentionCalibrator:
    """Calibrates abstention policy across thresholds and splits."""

    DEFAULT_THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    def __init__(
        self,
        records: List[Dict[str, Any]],
        thresholds: Optional[List[float]] = None,
        raw_eval_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Args:
            records: List of prediction records from run_generation_eval.
            thresholds: List of tau values to sweep.
            raw_eval_overrides: Optional mapping query_id -> {'correctness': int, 'factuality': int}
                providing judge evaluation of raw answers for queries that originally abstained.
        """
        self.raw_records = records
        self.thresholds = sorted(thresholds or self.DEFAULT_THRESHOLDS)
        self.raw_overrides = raw_eval_overrides or {}
        self.processed_records = self._prepare_records()

    def _prepare_records(self) -> List[Dict[str, Any]]:
        prepared = []
        for r in self.raw_records:
            qid = r.get("query_id", "")
            split = r.get("split", "val")
            verif = r.get("verification", {})
            eval_dict = r.get("evaluation", {})

            # Determine true underlying raw answer evaluation
            if qid in self.raw_overrides:
                raw_corr = float(self.raw_overrides[qid].get("correctness", 0))
                raw_fact = float(self.raw_overrides[qid].get("factuality", 0))
            else:
                raw_corr = float(eval_dict.get("correctness", 0))
                raw_fact = float(eval_dict.get("factuality", 0))

            prepared.append({
                "query_id": qid,
                "split": split,
                "difficulty": r.get("difficulty", "Medium"),
                "question_type": r.get("question_type", "factual"),
                "generation_failed": r.get("generation_failed", False),
                "model_abstained": verif.get("abstention_reason") == "model_abstained",
                "support_rate": float(verif.get("support_rate", 0.0)),
                "raw_correctness": raw_corr,
                "raw_factuality": raw_fact,
            })
        return prepared

    @staticmethod
    def _calc_f_beta(coverage: float, factuality: float, beta: float = 0.5) -> float:
        """
        F_beta utility between coverage [0, 1] and factuality normalized to [0, 1] (dividing by 5.0).
        With beta=0.5, factuality is weighted higher than coverage (appropriate for high-stakes legal QA).
        """
        fact_norm = min(max(factuality / 5.0, 0.0), 1.0)
        cov_norm = min(max(coverage, 0.0), 1.0)
        if cov_norm == 0 or fact_norm == 0:
            return 0.0
        b2 = beta ** 2
        return (1 + b2) * (cov_norm * fact_norm) / (b2 * cov_norm + fact_norm)

    def evaluate_point(self, records: List[Dict[str, Any]], threshold: float) -> SweepPoint:
        policy = AbstentionPolicy(threshold=threshold)
        total = len(records)
        if total == 0:
            return SweepPoint(threshold, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        answered = []
        abstained_count = 0
        for r in records:
            verif_dict = {
                "support_rate": r["support_rate"],
                "abstention_reason": "model_abstained" if r["model_abstained"] else None,
            }
            rec_input = {
                "generation_failed": r["generation_failed"],
                "verification": verif_dict,
            }
            should_abstain, _ = policy.decide(rec_input)
            if should_abstain:
                abstained_count += 1
            else:
                answered.append(r)

        ans_count = len(answered)
        cov = ans_count / total
        abs_rate = abstained_count / total

        if ans_count > 0:
            sel_corr = sum(r["raw_correctness"] for r in answered) / ans_count
            sel_fact = sum(r["raw_factuality"] for r in answered) / ans_count
            mean_sup = sum(r["support_rate"] for r in answered) / ans_count
        else:
            sel_corr = 0.0
            sel_fact = 0.0
            mean_sup = 0.0

        eff_corr = sum(r["raw_correctness"] for r in answered) / total
        eff_fact = sum(r["raw_factuality"] for r in answered) / total
        u_f05 = self._calc_f_beta(cov, sel_fact, beta=0.5)

        return SweepPoint(
            threshold=threshold,
            total=total,
            answered=ans_count,
            abstained=abstained_count,
            coverage_rate=cov,
            abstention_rate=abs_rate,
            selective_correctness=round(sel_corr, 2),
            selective_factuality=round(sel_fact, 2),
            effective_correctness=round(eff_corr, 2),
            effective_factuality=round(eff_fact, 2),
            mean_claim_support=round(mean_sup, 3),
            utility_f05=round(u_f05, 3),
        )

    def run_sweep(self, records: List[Dict[str, Any]]) -> List[SweepPoint]:
        points = [self.evaluate_point(records, t) for t in self.thresholds]

        # Identify Pareto frontier (maximizing coverage and selective_factuality)
        for i, pt in enumerate(points):
            dominated = False
            for other in points:
                if (
                    other.coverage_rate >= pt.coverage_rate
                    and other.selective_factuality >= pt.selective_factuality
                    and (other.coverage_rate > pt.coverage_rate or other.selective_factuality > pt.selective_factuality)
                ):
                    dominated = True
                    break
            pt.is_pareto = not dominated

        return points

    def calibrate(self) -> Dict[str, Any]:
        """
        Runs full calibration protocol:
          1. Run sweep on validation split (n=26).
          2. Compute Pareto frontiers.
          3. Evaluate standardized policy profiles:
             - Permissive (tau=0.0): Maximum raw coverage.
             - Balanced (tau=0.5): Default verified policy enforcing majority claim grounding.
             - Strict (tau=0.7): High-confidence policy for high-consequence compliance.
          4. Evaluate locked test split (n=24) across all profiles without post-hoc tuning.
        """
        val_recs = [r for r in self.processed_records if r["split"] == "val"]
        test_recs = [r for r in self.processed_records if r["split"] == "test"]
        all_recs = self.processed_records

        val_points = self.run_sweep(val_recs)
        test_points = self.run_sweep(test_recs)
        all_points = self.run_sweep(all_recs)

        def get_point(points: List[SweepPoint], tau: float) -> Optional[Dict[str, Any]]:
            pt = next((p for p in points if abs(p.threshold - tau) < 1e-4), None)
            return asdict(pt) if pt else None

        profiles = {}
        for name, tau in [("permissive", 0.0), ("balanced", 0.5), ("strict", 0.7)]:
            profiles[name] = {
                "threshold": tau,
                "validation": get_point(val_points, tau),
                "test": get_point(test_points, tau),
                "all": get_point(all_points, tau),
            }

        # Select validation-optimal tau:
        # Among grounded policies (tau >= 0.4 ensuring majority of claims supported),
        # pick the point maximizing utility_f05.
        grounded_val = [p for p in val_points if p.threshold >= 0.4]
        if grounded_val:
            sorted_grounded = sorted(grounded_val, key=lambda p: (p.utility_f05, p.coverage_rate), reverse=True)
            recommended_tau = sorted_grounded[0].threshold
        else:
            recommended_tau = 0.5

        return {
            "recommended_tau": recommended_tau,
            "profiles": profiles,
            "validation_sweep": [asdict(p) for p in val_points],
            "test_sweep": [asdict(p) for p in test_points],
            "all_sweep": [asdict(p) for p in all_points],
            "frozen_val_result": get_point(val_points, recommended_tau),
            "frozen_test_result": get_point(test_points, recommended_tau),
            "frozen_all_result": get_point(all_points, recommended_tau),
            "val_count": len(val_recs),
            "test_count": len(test_recs),
            "total_count": len(all_recs),
        }
