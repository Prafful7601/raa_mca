"""Core verification logic shared by the offline evaluation pipeline and the
FastAPI service. Implements baselines B1/B2/B3 and Ours (K sampled checks +
execution agreement), per Step 2 of the spec."""
from __future__ import annotations
import json
import re
from typing import Optional

from src.checks import execute_check
from src.llm_client import LLMClient
from src.table_utils import df_to_markdown, load_table_cached

SYSTEM_PROMPT = (
    "You are a careful fact-checking assistant that verifies claims about "
    "tabular data. Answer only in the exact format requested."
)

B1_PROMPT = """Table:
{table_md}

Claim: {claim}

Is this claim TRUE or FALSE based only on the table above? Answer with exactly one word: TRUE or FALSE."""

B2_PROMPT = """Table:
{table_md}

Claim: {claim}

Is this claim TRUE or FALSE based only on the table above?
Respond in exactly this format on two lines:
VERDICT: TRUE or FALSE
CONFIDENCE: <an integer 0-100, your confidence that the verdict is correct>"""

CHECK_GEN_PROMPT = """Table:
{table_md}

Claim: {claim}

Translate this claim into a single JSON "check" object that can be mechanically executed against the table with pandas, so the verdict is computed by code rather than by you. Use exactly this schema:
{{
  "filters": [{{"column": <column name from the table>, "operator": <operator>, "value": <value>}}, ...],
  "target_column": <column name or null>,
  "operation": "lookup"|"count"|"sum"|"avg"|"max"|"min"|"argmax"|"argmin"|"compare"|"difference",
  "expected_value": <the value the claim asserts>,
  "comparator": <operator>
}}

STRICT RULES (checks that break these are discarded, so follow them exactly):
- <operator> must be EXACTLY one of these five characters or fewer, nothing else: = != > < >= <=
  Never write "==", never write "<>", never add a space before or after it (write "=", not " =" or "==").
- Every "column" and "target_column" value must be copied character-for-character from the table's header row above -- do not paraphrase, translate, or guess a column name.
- Output ONLY the JSON object. No explanation, no markdown code fences, no text before or after it.

Worked examples (the tables in these examples are illustrations of the format, not the table above):

Example 1 -- lookup:
Table:
| name | department | salary |
| --- | --- | --- |
| ravi | sales | 40000 |
| meera | it | 55000 |
Claim: "meera works in the it department"
{{"filters": [{{"column": "name", "operator": "=", "value": "meera"}}], "target_column": "department", "operation": "lookup", "expected_value": "it", "comparator": "="}}

Example 2 -- aggregation/count:
Table:
| city | region |
| --- | --- |
| pune | west |
| nashik | west |
| indore | central |
Claim: "there are 2 cities in the west region"
{{"filters": [{{"column": "region", "operator": "=", "value": "west"}}], "target_column": null, "operation": "count", "expected_value": 2, "comparator": "="}}

Now produce the JSON check for the real table and claim above."""


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _parse_b1(text: str) -> str:
    t = text.strip().upper()
    if "TRUE" in t and "FALSE" not in t:
        return "TRUE"
    if "FALSE" in t and "TRUE" not in t:
        return "FALSE"
    return "TRUE" if t.startswith("T") else "FALSE" if t.startswith("F") else "INVALID"


def _parse_b2(text: str):
    verdict_m = re.search(r"VERDICT:\s*(TRUE|FALSE)", text, re.IGNORECASE)
    conf_m = re.search(r"CONFIDENCE:\s*(\d+)", text)
    verdict = verdict_m.group(1).upper() if verdict_m else "INVALID"
    confidence = int(conf_m.group(1)) if conf_m else None
    return verdict, confidence


async def _gen_check(client: LLMClient, model: str, claim: str, table_md: str, temperature: float,
                      nonce: int = 0) -> dict:
    prompt = CHECK_GEN_PROMPT.format(table_md=table_md, claim=claim)
    resp = await client.complete(model=model, prompt=prompt, system=SYSTEM_PROMPT,
                                  temperature=temperature, max_tokens=400, nonce=nonce)
    check = _extract_json(resp["text"])
    return {"raw_text": resp["text"], "check": check, "usage": resp["usage"]}


async def run_b1(client: LLMClient, model: str, claim: str, table_md: str) -> dict:
    prompt = B1_PROMPT.format(table_md=table_md, claim=claim)
    resp = await client.complete(model=model, prompt=prompt, system=SYSTEM_PROMPT, temperature=0.0, max_tokens=10)
    verdict = _parse_b1(resp["text"])
    return {"method": "B1", "verdict": verdict, "raw_text": resp["text"], "usage": resp["usage"]}


async def run_b2(client: LLMClient, model: str, claim: str, table_md: str) -> dict:
    prompt = B2_PROMPT.format(table_md=table_md, claim=claim)
    resp = await client.complete(model=model, prompt=prompt, system=SYSTEM_PROMPT, temperature=0.0, max_tokens=30)
    verdict, confidence = _parse_b2(resp["text"])
    return {"method": "B2", "verdict": verdict, "confidence": confidence, "raw_text": resp["text"], "usage": resp["usage"]}


async def run_b3(client: LLMClient, model: str, claim: str, table_md: str, df) -> dict:
    gen = await _gen_check(client, model, claim, table_md, temperature=0.0)
    if gen["check"] is None:
        return {"method": "B3", "verdict": "UNVERIFIABLE", "executed_checks": [gen], "usage": gen["usage"]}
    result = execute_check(df, gen["check"])
    verdict = result["verdict"] if result["ok"] else "UNVERIFIABLE"
    return {"method": "B3", "verdict": verdict, "executed_checks": [{**gen, "result": result}], "usage": gen["usage"]}


async def run_ours(client: LLMClient, model: str, claim: str, table_md: str, df, k: int, temperature: float) -> dict:
    import asyncio
    gens = await asyncio.gather(*[_gen_check(client, model, claim, table_md, temperature, nonce=i) for i in range(k)])
    executed = []
    votes = []
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    for gen in gens:
        total_usage["input_tokens"] += gen["usage"]["input_tokens"]
        total_usage["output_tokens"] += gen["usage"]["output_tokens"]
        if gen["check"] is None:
            executed.append({**gen, "result": {"ok": False, "verdict": "INVALID", "error": "no JSON parsed"}})
            continue
        result = execute_check(df, gen["check"])
        executed.append({**gen, "result": result})
        if result["ok"]:
            votes.append(result["verdict"])
    if not votes:
        return {"method": "Ours", "verdict": "UNVERIFIABLE", "agreement_score": 0.0,
                "executed_checks": executed, "usage": total_usage}
    n_true = votes.count("TRUE")
    n_false = votes.count("FALSE")
    majority = "TRUE" if n_true >= n_false else "FALSE"
    agreement = max(n_true, n_false) / k  # invalid/failed checks count as abstain votes (denominator = k)
    return {"method": "Ours", "verdict": majority, "agreement_score": agreement,
            "n_valid": len(votes), "k": k, "executed_checks": executed, "usage": total_usage}


async def verify_claim_df(client: LLMClient, model: str, claim: str, df,
                           k: int = 5, temperature: float = 0.7, method: str = "ours") -> dict:
    table_md = df_to_markdown(df)
    if method == "b1":
        return await run_b1(client, model, claim, table_md)
    if method == "b2":
        return await run_b2(client, model, claim, table_md)
    if method == "b3":
        return await run_b3(client, model, claim, table_md, df)
    return await run_ours(client, model, claim, table_md, df, k, temperature)


async def verify_claim(client: LLMClient, model: str, claim: str, table_csv_path: str,
                        k: int = 5, temperature: float = 0.7, method: str = "ours") -> dict:
    df = load_table_cached(table_csv_path)
    return await verify_claim_df(client, model, claim, df, k, temperature, method)
