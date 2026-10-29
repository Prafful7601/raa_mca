"""The JSON check language (Step 2) and its pandas executor.

Schema (one check):
{
  "filters": [{"column": str, "operator": "="|"!="|">"|"<"|">="|"<="|"contains", "value": any}, ...],
  "target_column": str | null,
  "operation": "lookup"|"count"|"sum"|"avg"|"max"|"min"|"argmax"|"argmin"|"compare"|"difference",
  "expected_value": any,
  "comparator": "="|"!="|">"|"<"|">="|"<=" | null   # how expected_value relates to the computed result
}
Execution returns a dict: {ok, computed_value, verdict, error}. Any exception or
schema violation yields ok=False, verdict="INVALID" -- these count as abstain votes.
"""
from __future__ import annotations
from typing import Any

from src.number_parsing import parse_number

TOLERANCE = 1e-6
VALID_OPS = {"=", "!=", ">", "<", ">=", "<=", "contains"}
VALID_OPERATIONS = {"lookup", "count", "sum", "avg", "max", "min", "argmax", "argmin", "compare", "difference"}


class CheckError(Exception):
    pass


def _coerce(series, value):
    """Try numeric comparison first, fall back to case-insensitive string compare."""
    num_value = parse_number(value)
    return num_value


_OP_ALIASES = {"==": "=", "<>": "!=", "eq": "=", "ne": "!=", "gt": ">", "lt": "<", "ge": ">=", "le": "<="}


def _normalize_operator(op) -> str:
    """Tolerate cosmetic variation (stray whitespace, '==' for '=', word forms)
    an LLM may emit despite the schema; the *comparator* field faces the same
    risk and is normalized the same way in _apply_comparator below."""
    op = str(op).strip()
    return _OP_ALIASES.get(op, op)


def apply_filters(df, filters: list[dict]):
    out = df
    for f in filters or []:
        col = f.get("column")
        op = _normalize_operator(f.get("operator"))
        val = f.get("value")
        if col not in out.columns:
            raise CheckError(f"unknown column '{col}'")
        if op not in VALID_OPS:
            raise CheckError(f"unknown operator '{op}'")
        col_series = out[col].astype(str)
        if op == "contains":
            mask = col_series.str.contains(str(val), case=False, na=False, regex=False)
        else:
            num_val = parse_number(val)
            if num_val is not None:
                parsed_col = col_series.apply(parse_number)
                if op == "=":
                    mask = (parsed_col - num_val).abs() < TOLERANCE
                elif op == "!=":
                    mask = ~((parsed_col - num_val).abs() < TOLERANCE)
                elif op == ">":
                    mask = parsed_col > num_val
                elif op == "<":
                    mask = parsed_col < num_val
                elif op == ">=":
                    mask = parsed_col >= num_val
                else:
                    mask = parsed_col <= num_val
                mask = mask.fillna(False)
            else:
                sval = str(val).strip().lower()
                col_lower = col_series.str.strip().str.lower()
                if op == "=":
                    mask = col_lower == sval
                elif op == "!=":
                    mask = col_lower != sval
                else:
                    raise CheckError(f"operator '{op}' requires a numeric value")
        out = out[mask]
    return out


def execute_check(df, check: dict) -> dict:
    try:
        operation = check.get("operation")
        if operation not in VALID_OPERATIONS:
            return {"ok": False, "computed_value": None, "verdict": "INVALID",
                    "error": f"unknown operation '{operation}'"}
        filtered = apply_filters(df, check.get("filters"))
        target_col = check.get("target_column")

        if operation == "count":
            computed = float(len(filtered))
        elif operation in ("sum", "avg", "max", "min"):
            if target_col not in df.columns:
                raise CheckError(f"unknown target_column '{target_col}'")
            vals = [parse_number(v) for v in filtered[target_col]]
            vals = [v for v in vals if v is not None]
            if not vals:
                raise CheckError("no numeric values in target column after filtering")
            if operation == "sum":
                computed = float(sum(vals))
            elif operation == "avg":
                computed = float(sum(vals) / len(vals))
            elif operation == "max":
                computed = float(max(vals))
            else:
                computed = float(min(vals))
        elif operation in ("lookup", "argmax", "argmin"):
            if target_col not in df.columns:
                raise CheckError(f"unknown target_column '{target_col}'")
            if len(filtered) == 0:
                raise CheckError("filter matched no rows")
            if operation == "lookup":
                raw = filtered.iloc[0][target_col]
                num = parse_number(raw)
                computed = num if num is not None else str(raw).strip()
            else:
                vals = filtered[target_col].apply(parse_number)
                if vals.isna().all():
                    raise CheckError("no numeric values to argmax/argmin over")
                idx = vals.idxmax() if operation == "argmax" else vals.idxmin()
                row = filtered.loc[idx]
                computed = str(row[target_col]).strip()
        elif operation in ("compare", "difference"):
            if target_col not in df.columns:
                raise CheckError(f"unknown target_column '{target_col}'")
            vals = [parse_number(v) for v in filtered[target_col]]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                raise CheckError("compare/difference needs >=2 matching numeric rows")
            # `compare` and `difference` both reduce to the signed gap between the
            # first two matching rows; the comparator (e.g. >, =) is what expresses
            # "greater than" vs. "equal to" for `compare`, applied to this value below.
            computed = float(vals[0] - vals[1])
        else:
            raise CheckError("unreachable")

        expected = check.get("expected_value")
        comparator = check.get("comparator", "=")
        verdict = _apply_comparator(computed, expected, comparator)
        return {"ok": True, "computed_value": computed, "verdict": verdict, "error": None}
    except CheckError as e:
        return {"ok": False, "computed_value": None, "verdict": "INVALID", "error": str(e)}
    except Exception as e:  # defensive: any executor bug -> abstain, never crash the API
        return {"ok": False, "computed_value": None, "verdict": "INVALID", "error": f"executor error: {e}"}


def _apply_comparator(computed: Any, expected: Any, comparator: str) -> str:
    comparator = _normalize_operator(comparator) if comparator else "="
    exp_num = parse_number(expected)
    comp_num = parse_number(computed) if not isinstance(computed, (int, float)) else computed
    if exp_num is not None and isinstance(comp_num, (int, float)):
        diff = comp_num - exp_num
        if comparator == "=":
            result = abs(diff) < TOLERANCE
        elif comparator == "!=":
            result = abs(diff) >= TOLERANCE
        elif comparator == ">":
            result = diff > TOLERANCE
        elif comparator == "<":
            result = diff < -TOLERANCE
        elif comparator == ">=":
            result = diff > -TOLERANCE
        elif comparator == "<=":
            result = diff < TOLERANCE
        else:
            raise CheckError(f"unknown comparator '{comparator}'")
    else:
        c = str(computed).strip().lower()
        e = str(expected).strip().lower()
        if comparator == "=":
            result = c == e
        elif comparator == "!=":
            result = c != e
        else:
            raise CheckError(f"comparator '{comparator}' requires numeric values")
    return "TRUE" if result else "FALSE"
