"""Helpers for LLM-based multi-label coding of writers' free-text preference reasons.

Coding uses an OpenRouter-hosted model through the OpenAI SDK. Every response gets all
codebook codes that apply. Reasons that no code captures are listed and trigger the
`other` code. Results are cached per codebook version.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.contrib.concurrent import thread_map

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
DEFAULT_MAX_WORKERS = 20
DEFAULT_MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 60
OTHER_CODE = "other"

PREFERENCE_DESCRIPTIONS = {
    "edited": "the AI-edited version of their paragraph",
    "original": "their own original paragraph",
    "equal": "neither paragraph (they liked both equally)",
}

DEFAULT_QUESTION_CONTEXT = "They were then asked to explain their preference in free text."

PROMPT_TEMPLATE = """A study participant wrote a short opinion paragraph on a political proposition. An AI model then edited the paragraph. The participant was shown both versions and said they preferred {preference_description}. {question_context}

Your task is to code the participant's explanation using the CODEBOOK below. Each code is a directional claim about the preferred paragraph, written from the participant's perspective ("me" and "my" refer to the participant). A code applies if the explanation makes that claim, either about the preferred paragraph directly or by saying the other paragraph has the opposite property (e.g. "the AI version is too formal" for a participant who preferred their own paragraph means their preferred paragraph is less formal).

CODEBOOK:
{codebook_text}

Instructions:
- Assign EVERY code whose reason is expressed in the explanation. An explanation can have several codes.
- Only code reasons that are actually expressed. Do not infer reasons that are not stated.
- Ignore concessions, i.e. praise for the paragraph the participant did NOT prefer (e.g. "the AI version is more articulate, but I prefer mine"). These are not reasons for the preference: do not code them and do not list them as uncovered reasons.
- If the explanation expresses a reason that none of the codes capture, describe each such reason in a short phrase (max 8 words) under "uncovered_reasons". Otherwise leave "uncovered_reasons" empty.
- Do not use the code "{other_code}" yourself; it is assigned automatically from "uncovered_reasons".

EXPLANATION:
\"\"\"{text}\"\"\"

Respond only with a JSON object of this form:
{{"codes": ["<code>", ...], "uncovered_reasons": ["<short phrase>", ...]}}"""


# =============================================================================
# DATA AND CODEBOOK
# =============================================================================

def load_reason_texts(csv_path: Path, text_col: str, condition_col: str | None = None) -> pd.DataFrame:
    """Load non-empty free-text reasons with one row per writer-proposition pair."""
    df = pd.read_csv(csv_path)
    text = df[text_col].fillna("").astype(str).str.strip()
    df = df.loc[text != ""].copy()
    out = pd.DataFrame({
        "response_id": df["writer_id"].astype(str) + "_" + df["proposition_id"].astype(str),
        "writer_id": df["writer_id"],
        "proposition_id": df["proposition_id"],
        "writer_preference": df["writer_preference"],
        "condition": df[condition_col] if condition_col else pd.NA,
        "text": text.loc[df.index],
    })
    if out["response_id"].duplicated().any():
        raise ValueError("response_id is not unique")
    return out.reset_index(drop=True)


def load_codebook(path: Path) -> dict:
    with open(path) as f:
        codebook = json.load(f)
    codes = [entry["code"] for entry in codebook["codes"]]
    if OTHER_CODE not in codes:
        raise ValueError(f"Codebook must contain the code '{OTHER_CODE}'")
    if len(set(codes)) != len(codes):
        raise ValueError("Codebook contains duplicate codes")
    return codebook


def codebook_codes(codebook: dict) -> list[str]:
    return [entry["code"] for entry in codebook["codes"]]


def format_codebook(codebook: dict) -> str:
    lines = []
    for entry in codebook["codes"]:
        if entry["code"] == OTHER_CODE:
            continue
        line = f"- {entry['code']}: {entry['definition']}"
        if entry.get("examples"):
            line += " Examples: " + "; ".join(f'"{ex}"' for ex in entry["examples"])
        lines.append(line)
    return "\n".join(lines)


# =============================================================================
# LLM CODING
# =============================================================================

def _parse_response(content: str, valid_codes: set[str]) -> tuple[list[str], list[str]]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content[content.find("{"):]
    start, end = content.find("{"), content.rfind("}")
    parsed = json.loads(content[start:end + 1])
    codes = [str(c).strip() for c in parsed.get("codes", [])]
    invalid = [c for c in codes if c not in valid_codes]
    if invalid:
        raise ValueError(f"Invalid codes: {invalid}")
    uncovered = [str(r).strip() for r in parsed.get("uncovered_reasons", []) if str(r).strip()]
    codes = [c for c in codes if c != OTHER_CODE]
    if uncovered:
        codes.append(OTHER_CODE)
    return sorted(set(codes)), uncovered


def _load_cache(cache_path: Path, version: str, model: str) -> dict:
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                record = json.loads(line)
                if record["codebook_version"] == version and record["model"] == model:
                    cache[record["response_id"]] = record
    return cache


def code_responses(
    df: pd.DataFrame,
    codebook: dict,
    cache_path: Path,
    model: str = DEFAULT_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    question_context: str = DEFAULT_QUESTION_CONTEXT,
) -> pd.DataFrame:
    """Code each response in `df`. Returns `df` with `codes`, `uncovered_reasons` and one 0/1 column per code."""
    version = str(codebook["version"])
    codes = codebook_codes(codebook)
    valid_codes = set(codes)
    codebook_text = format_codebook(codebook)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path, version, model)
    todo = df.loc[~df["response_id"].isin(cache.keys())]

    if len(todo) > 0:
        api_key = os.environ.get(API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"Set {API_KEY_ENV} in .env")
        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        lock = threading.Lock()

        def code_one(row) -> dict | None:
            prompt = PROMPT_TEMPLATE.format(
                preference_description=PREFERENCE_DESCRIPTIONS[row.writer_preference],
                question_context=question_context,
                codebook_text=codebook_text,
                other_code=OTHER_CODE,
                text=row.text,
            )
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        response_format={"type": "json_object"},
                    )
                    assigned, uncovered = _parse_response(response.choices[0].message.content, valid_codes)
                    record = {
                        "response_id": row.response_id,
                        "codebook_version": version,
                        "model": model,
                        "codes": assigned,
                        "uncovered_reasons": uncovered,
                    }
                    with lock, open(cache_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    return record
                except Exception as exc:
                    if attempt == max_retries - 1:
                        print(f"Failed {row.response_id}: {exc}")
                        return None
                    time.sleep(2 ** attempt)

        results = thread_map(code_one, list(todo.itertuples(index=False)), max_workers=max_workers, desc="Coding")
        for record in results:
            if record is not None:
                cache[record["response_id"]] = record

    n_failed = (~df["response_id"].isin(cache.keys())).sum()
    if n_failed:
        print(f"WARNING: {n_failed} responses could not be coded (re-run to retry)")

    out = df.loc[df["response_id"].isin(cache.keys())].copy()
    out["codes"] = out["response_id"].map(lambda rid: cache[rid]["codes"])
    out["uncovered_reasons"] = out["response_id"].map(lambda rid: cache[rid]["uncovered_reasons"])
    for code in codes:
        out[code] = out["codes"].map(lambda assigned, code=code: int(code in assigned))
    out["n_codes"] = out["codes"].map(len)
    out["codebook_version"] = version
    return out.reset_index(drop=True)


def flatten_for_csv(coded: pd.DataFrame) -> pd.DataFrame:
    out = coded.copy()
    out["codes"] = out["codes"].map(lambda xs: "; ".join(xs))
    out["uncovered_reasons"] = out["uncovered_reasons"].map(lambda xs: "; ".join(xs))
    return out

