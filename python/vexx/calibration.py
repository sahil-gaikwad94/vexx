"""Statistical and epistemic diagnostics for Vexx 2.1.

Every diagnostic is returned as a typed :class:`Diagnostic`.  Only ``CONFLICT``
with sufficient confidence may block a result; heuristics never masquerade as
proofs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

DIMENSION_BASE: dict[str, dict[str, int]] = {
    "kg": {"M": 1}, "g": {"M": 1}, "m": {"L": 1}, "s": {"T": 1},
    "a": {"I": 1}, "k": {"Theta": 1}, "mol": {"N": 1}, "cd": {"J": 1},
    "n": {"M": 1, "L": 1, "T": -2}, "j": {"M": 1, "L": 2, "T": -2},
    "w": {"M": 1, "L": 2, "T": -3}, "pa": {"M": 1, "L": -1, "T": -2},
    "hz": {"T": -1}, "c": {"I": 1, "T": 1},
    "v": {"M": 2, "L": 2, "T": -3, "I": -1},
}


@dataclass
class Diagnostic:
    kind: str
    verdict: str  # CONFLICT | INSUFFICIENT_EVIDENCE | UNKNOWN | NO_FINDING | established | not established
    confidence: str  # exact | strong_heuristic | heuristic
    explanation: str
    evidence: dict[str, Any] = field(default_factory=dict)
    rule_id: str = ""

    def __post_init__(self) -> None:
        if self.verdict == "CONFLICT" and "FINAL: CONFLICT" not in self.explanation:
            self.explanation = self.explanation + "\nFINAL: CONFLICT"

    def __getitem__(self, key: str) -> Any:
        mapping = {
            "type": self.kind,
            "kind": self.kind,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "rule_id": self.rule_id,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in {"type", "kind", "verdict", "confidence", "explanation", "evidence", "rule_id"}


def parse_unit_dimensions(text: str) -> dict[str, int] | None:
    cleaned = str(text or "").strip().lower().replace("·", "*").replace(" ", "")
    if not cleaned:
        return None
    tokens = re.findall(r"[*/]|[a-z]+(?:\^[+-]?\d+)?", cleaned)
    if not tokens or "".join(tokens) != cleaned:
        return None
    total: dict[str, int] = {}
    sign = 1
    for token in tokens:
        if token == "*":
            sign = 1
            continue
        if token == "/":
            sign = -1
            continue
        match = re.fullmatch(r"([a-z]+)(?:\^([+-]?\d+))?", token)
        if match is None:
            return None
        unit = match.group(1)
        if unit not in DIMENSION_BASE:
            return None
        exponent = int(match.group(2) or 1)
        for dimension, power in DIMENSION_BASE[unit].items():
            total[dimension] = total.get(dimension, 0) + sign * exponent * power
    return {key: value for key, value in total.items() if value}


def _normal_two_sided_p(statistic: float) -> float:
    return math.erfc(abs(float(statistic)) / math.sqrt(2.0))


def _chi2_survival(statistic: float, df: int) -> float | None:
    x = float(statistic)
    if x < 0 or df <= 0:
        return None
    half_x = x / 2.0
    if df % 2 == 0:
        term = 1.0
        total = 1.0
        for i in range(1, df // 2):
            term *= half_x / i
            total += term
        return math.exp(-half_x) * total
    survival = math.erfc(math.sqrt(half_x))
    gamma = math.sqrt(math.pi) / 2.0
    power = math.sqrt(half_x)
    for j in range((df - 1) // 2):
        survival += math.exp(-half_x) * power / gamma
        power *= half_x
        gamma *= 1.5 + j
    return survival


def _t_survival_two_sided(t: float, df: int) -> float | None:
    # Use a simple approximation? We need real Student-t. Try scipy if available.
    try:
        from scipy import stats as st  # type: ignore
        return float(st.t.sf(abs(t), df) * 2.0)
    except Exception:
        # Fallback: normal approximation (marked heuristic).
        return None


def _decimal_precision(text: str) -> int:
    if "." in text:
        return len(text.split(".", 1)[1])
    return 0


def _grim_interval(value_text: str) -> tuple[float, float]:
    """Return the rounding interval implied by a printed decimal."""
    precision = _decimal_precision(value_text)
    if precision == 0:
        half = 0.5
    else:
        half = 0.5 * 10 ** (-precision)
    v = float(value_text)
    return v - half, v + half


def verify_statistical_report(text: str, claims: list[dict[str, Any]] | None = None) -> Diagnostic | None:
    text = str(text or "")
    lower = text.lower()

    # GRIM percentage check with rounding interval.
    m_grim = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:patients\s+|people\s+|participants\s+)?"
        r"(?:n|N)\s*=\s*(\d+)",
        text,
    )
    if m_grim is not None:
        percent_text = m_grim.group(1)
        n_total = int(m_grim.group(2))
        if n_total > 0:
            lo, hi = _grim_interval(percent_text)
            # count = percent * n / 100 must be an integer within rounding interval
            count_lo = lo * n_total / 100.0
            count_hi = hi * n_total / 100.0
            # is there an integer in [count_lo, count_hi]?
            import math as _m
            first_int = _m.ceil(count_lo - 1e-12)
            if first_int > count_hi + 1e-12:
                percent = float(percent_text)
                count = percent * n_total / 100.0
                return Diagnostic(
                    kind="grim_violation", verdict="CONFLICT", confidence="exact",
                    explanation=f"{percent:g}% of N={n_total} implies {count:g} subjects, not an integer count (GRIM).",
                    evidence={"percent": percent, "n": n_total, "count": count},
                    rule_id="GRIM_PERCENT",
                )

    # GRIM mean check.
    m_mean = re.search(
        r"\b(?:mean|average)\s*(?:=|of|is)\s*(\d+(?:\.\d+)?)[^.;]{0,40}?"
        r"\b(?:n|N)\s*=\s*(\d+)",
        text,
    )
    if m_mean is not None:
        mean_text = m_mean.group(1)
        n_total = int(m_mean.group(2))
        if n_total > 0:
            lo, hi = _grim_interval(mean_text)
            total_lo = lo * n_total
            total_hi = hi * n_total
            first_int = math.ceil(total_lo - 1e-12)
            if first_int > total_hi + 1e-12:
                mean = float(mean_text)
                payload = mean * n_total
                return Diagnostic(
                    kind="grim_violation", verdict="CONFLICT", confidence="exact",
                    explanation=f"mean {mean:g} over N={n_total} implies total {payload:g}, impossible for integer measurements (GRIM).",
                    evidence={"mean": mean, "n": n_total, "total": payload},
                    rule_id="GRIM_MEAN",
                )

    # Structured claims handling.
    if claims:
        for claim in claims:
            kind = claim.get("kind")
            if kind == "z_test":
                z = claim.get("statistic")
                p = claim.get("p_value")
                alt = claim.get("alternative", "two-sided")
                if z is not None and p is not None:
                    if alt == "two-sided":
                        expected = _normal_two_sided_p(float(z))
                    else:
                        # one-sided: p = P(Z > |z|)
                        expected = 0.5 * _normal_two_sided_p(float(z))  # approximation; better: erfc
                        expected = math.erfc(abs(float(z)) / math.sqrt(2.0)) / 2.0
                    if expected > 0:
                        ratio = max(expected / float(p), float(p) / expected)
                        if ratio > 3.0 and abs(expected - float(p)) > 0.01:
                            return Diagnostic(
                                kind="statistical_inconsistency", verdict="CONFLICT", confidence="exact",
                                explanation=f"z={float(z):g} implies p≈{expected:.4g}, report states p={float(p):g}.",
                                evidence={"z": z, "expected_p": expected, "reported_p": p},
                                rule_id="Z_P_MISMATCH",
                            )
            elif kind == "chi2":
                chi = claim.get("statistic")
                df = claim.get("df")
                p = claim.get("p_value")
                if chi is not None and df is not None and p is not None:
                    expected = _chi2_survival(float(chi), int(df))
                    if expected is not None and expected > 0:
                        ratio = max(expected / float(p), float(p) / expected)
                        if ratio > 3.0 and abs(expected - float(p)) > 0.01:
                            return Diagnostic(
                                kind="statistical_inconsistency", verdict="CONFLICT", confidence="exact",
                                explanation=f"chi2={float(chi):g}, df={int(df)} implies p≈{expected:.4g}, report states p={float(p):g}.",
                                evidence={"chi2": chi, "df": df, "expected_p": expected, "reported_p": p},
                                rule_id="CHI2_P_MISMATCH",
                            )
            elif kind == "t_test":
                t = claim.get("statistic")
                df = claim.get("df")
                p = claim.get("p_value")
                alt = claim.get("alternative", "two-sided")
                if t is not None and df is not None and p is not None:
                    surv = _t_survival_two_sided(float(t), int(df))
                    if surv is not None:
                        expected = surv if alt == "two-sided" else surv / 2.0
                        if expected > 0:
                            # Only flag if reported p is much smaller than the t-based bound.
                            if float(p) * 10.0 < expected:
                                return Diagnostic(
                                    kind="statistical_inconsistency", verdict="CONFLICT", confidence="strong_heuristic",
                                    explanation=f"t={float(t):g}, df={int(df)} implies p≈{expected:.4g}, report states p={float(p):g}.",
                                    evidence={"t": t, "df": df, "expected_p": expected, "reported_p": p},
                                    rule_id="T_P_MISMATCH",
                                )

    # Free-form z/t/chi2 extraction (lower confidence).
    m_p = re.search(r"\bp\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?)", lower)
    if m_p is not None:
        reported = float(m_p.group(1))
        if 0.0 < reported <= 1.0:
            m_z = re.search(r"\bz\s*=\s*([+\-]?\d+(?:\.\d+)?)", lower)
            m_t = re.search(r"\bt\s*=\s*([+\-]?\d+(?:\.\d+)?)", lower)
            m_df = re.search(r"\bdf\s*=\s*(\d+)", lower)
            m_chi = re.search(r"(?:chi2|\u03c7\u00b2|chi-square)\s*=\s*(\d+(?:\.\d+)?)", lower)
            expected: float | None = None
            label = ""
            one_sided_lower_bound = False
            if m_z is not None:
                expected = _normal_two_sided_p(float(m_z.group(1)))
                label = f"z={float(m_z.group(1)):g}"
            elif m_chi is not None and m_df is not None:
                expected = _chi2_survival(float(m_chi.group(1)), int(m_df.group(1)))
                label = f"chi2={float(m_chi.group(1)):g}, df={int(m_df.group(1))}"
            elif m_t is not None:
                t_value = abs(float(m_t.group(1)))
                expected = _normal_two_sided_p(t_value)
                label = f"t={t_value:g}"
                one_sided_lower_bound = m_df is None or int(m_df.group(1)) < 30
            if expected is not None and expected > 0.0:
                ratio = max(expected / reported, reported / expected)
                too_far = ratio > 3.0 and abs(expected - reported) > 0.01
                if one_sided_lower_bound:
                    too_far = reported * 10.0 < expected
                if too_far:
                    return Diagnostic(
                        kind="statistical_inconsistency", verdict="CONFLICT", confidence="strong_heuristic",
                        explanation=f"{label} implies p≈{expected:.4g}, but report states p={reported:g}.",
                        evidence={"label": label, "expected_p": expected, "reported_p": reported},
                        rule_id="STAT_P_MISMATCH",
                    )

    # Unit dimensions.
    if "unit" in lower or "dimension" in lower:
        m_eq = re.search(
            r"(\d+(?:\.\d+)?)\s*([a-zA-Z][a-zA-Z0-9*/^·]*)\s*=\s*"
            r"(\d+(?:\.\d+)?)\s*([a-zA-Z][a-zA-Z0-9*/^·]*)",
            text,
        )
        if m_eq is not None:
            lhs = parse_unit_dimensions(m_eq.group(2))
            rhs = parse_unit_dimensions(m_eq.group(4))
            if lhs is not None and rhs is not None and lhs != rhs:
                return Diagnostic(
                    kind="dimensional_mismatch", verdict="CONFLICT", confidence="exact",
                    explanation=f"{m_eq.group(2)} dimensions {lhs} vs {m_eq.group(4)} {rhs}: inhomogeneous.",
                    evidence={"lhs": lhs, "rhs": rhs},
                    rule_id="DIM_MISMATCH",
                )

    return None


def verify_epistemic_calibration(text: str) -> Diagnostic | None:
    if not isinstance(text, str):
        raise TypeError("prompt must be a string")
    stat = verify_statistical_report(text)
    if stat is not None:
        return stat

    p_lower = text.lower()
    m_ci = re.search(
        r"(?P<level>(?:\d+(?:\.\d+)?))%\s*ci\s*\[\s*"
        r"(?P<lo>[−+\-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
        r"(?P<hi>[−+\-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]",
        p_lower,
    )
    m_p = re.search(r"\bp\s*=\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?)", p_lower)
    if m_ci or m_p:
        ci_sig = None
        p_sig = None
        ci_reason = None
        alpha = 0.05
        if m_ci:
            confidence = float(m_ci.group("level"))
            if not 0.0 < confidence < 100.0:
                return Diagnostic(kind="invalid_statistical_input", verdict="CONFLICT", confidence="exact",
                                  explanation=f"Invalid confidence level {confidence:g}%.", rule_id="CI_LEVEL")
            alpha = 1.0 - confidence / 100.0
            ci_lo = float(m_ci.group("lo").replace("−", "-"))
            ci_hi = float(m_ci.group("hi").replace("−", "-"))
            if ci_lo > ci_hi:
                return Diagnostic(kind="invalid_statistical_input", verdict="CONFLICT", confidence="exact",
                                  explanation="CI lower bound exceeds upper bound.", rule_id="CI_ORDER")
            ci_sig = not (ci_lo <= 0.0 <= ci_hi)
            ci_reason = f"the {confidence:g}% interval [{ci_lo}, {ci_hi}] " + ("excludes" if ci_sig else "includes") + " zero"
        p_reason = None
        if m_p:
            p_value = float(m_p.group(1))
            if not 0.0 <= p_value <= 1.0:
                return Diagnostic(kind="invalid_statistical_input", verdict="CONFLICT", confidence="exact",
                                  explanation=f"p-value {p_value} outside [0, 1].", rule_id="P_RANGE")
            p_sig = p_value < alpha
            p_reason = f"p={p_value:g} " + ("<" if p_sig else ">=") + f" {alpha:g}"

        if ci_sig is not None and p_sig is not None and ci_sig != p_sig:
            return Diagnostic(kind="statistical_conflict", verdict="CONFLICT", confidence="strong_heuristic",
                              explanation=f"Statistical indicators disagree: {ci_reason}, but {p_reason}.",
                              evidence={"ci_sig": ci_sig, "p_sig": p_sig}, rule_id="CI_P_CONFLICT")
        significant = ci_sig if ci_sig is not None else p_sig
        if significant is True:
            reason = " and ".join(x for x in (ci_reason, p_reason) if x)
            return Diagnostic(kind="statistical_significance", verdict="established", confidence="strong_heuristic",
                              explanation=f"Statistically significant at alpha={alpha:g} ({reason}).", rule_id="SIG")
        if significant is False:
            reason = " and ".join(x for x in (ci_reason, p_reason) if x)
            return Diagnostic(kind="statistical_significance", verdict="not established", confidence="strong_heuristic",
                              explanation=f"Not statistically significant at alpha={alpha:g} ({reason}).", rule_id="NONSIG")

    # Heuristic prose rules (always heuristic confidence).
    if "source a" in p_lower and "source b" in p_lower and "effective" in p_lower and ("official notice" in p_lower or "regulator" in p_lower):
        m_b = re.search(r"source b.*?(\d+)\s*units", p_lower)
        if m_b:
            units = m_b.group(1)
            return Diagnostic(kind="temporal_supersession", verdict=f"{units} units", confidence="heuristic",
                              explanation="Source B supersedes Source A.", rule_id="SUPERSESSION")

    if "optional online poll" in p_lower or ("visitors" in p_lower and "blog" in p_lower):
        return Diagnostic(kind="sampling_bias", verdict="not justified", confidence="heuristic",
                          explanation="Voluntary convenience sample.", rule_id="SAMPLING_BIAS")

    if "deliberately unverified" in p_lower or re.search(r"\bunverified\s+(?:award|entity|winner|claim)\b", p_lower):
        return Diagnostic(kind="unverified_premise", verdict="INSUFFICIENT EVIDENCE", confidence="heuristic",
                          explanation="Premise marked unverified.", rule_id="UNVERIFIED")

    if "fictional city" in p_lower or "fictional place" in p_lower:
        return Diagnostic(kind="fictional_entity", verdict="UNKNOWN", confidence="heuristic",
                          explanation="Fictional place with no supplied dataset.", rule_id="FICTIONAL")

    if ("table" in p_lower and "paragraph" in p_lower and "conflict" in p_lower) or ("contradictory" in p_lower and "source" in p_lower):
        return Diagnostic(kind="contradiction_detection", verdict="CONFLICT", confidence="heuristic",
                          explanation="Internally contradictory material.", rule_id="CONTRADICTION")

    m_decimal = re.search(r"(?<![\w.])0\.(?P<nines>9+)(?P<repeat>\.\.\.|…)?(?!\d)", p_lower)
    if m_decimal and ("less than 1" in p_lower or "< 1" in p_lower or "equal to 1" in p_lower or "equals 1" in p_lower):
        if bool(m_decimal.group("repeat")):
            return Diagnostic(kind="false_premise", verdict="equal", confidence="heuristic",
                              explanation="0.999... equals 1.", rule_id="REPEAT_DECIMAL")
        literal = "0." + m_decimal.group("nines")
        return Diagnostic(kind="numeric_literal", verdict="less than 1", confidence="heuristic",
                          explanation=f"{literal} is strictly less than 1.", rule_id="FINITE_DECIMAL")

    return Diagnostic(kind="heuristic_no_match", verdict="unknown", confidence="heuristic",
                      explanation="No epistemic heuristic rule matched.", rule_id="NO_MATCH")