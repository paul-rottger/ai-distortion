#!/usr/bin/env python3

# =============================================================================
# RERANKING PIPELINE - STEP 3: GENERATE REWARD-MODEL SCORES
#
# Scores reward-model test examples using a base or fine-tuned OpenAI model.
#
# - Loads test prompts from `reranking/finetuning_data/` for the selected dataset.
# - Sends concurrent API requests and parses structured stance predictions.
# - Writes per-model score outputs to `reranking/rm_scores/`.
#
# =============================================================================

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from tqdm.contrib.concurrent import thread_map

from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
FINETUNING_DATA_DIR = SCRIPT_DIR / "finetuning_data"
RESULTS_DIR = SCRIPT_DIR / "rm_scores"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"
DEFAULT_MAX_WORKERS = 30
DEFAULT_MAX_RETRIES = 5
DATASET_CHOICES = ("bullet_rm", "paragraph_rm")
TEST_REQUEST_PROMPT = "Tell me a joke."

PARAGRAPH_PROMPT_TEMPLATE = """You will be shown an OPINION PARAGRAPH concerning a PROPOSITION.
We showed this paragraph to 10 independent raters and asked a RATING QUESTION.
For this rating question, your task is to predict the MEAN rating value across all raters.

RATING QUESTION:
writer_stance: What is the author's level of agreement with the proposition? [0-100 scale, where 0 = "Strongly disagree", 50 = "Ambivalent", and 100 = "Strongly agree"]

PROPOSITION: {proposition}

OPINION PARAGRAPH: {paragraph}
"""

BULLET_PROMPT_TEMPLATE = """You will be shown BULLET POINTS written by a person to explain their position on a PROPOSITION.
The person who wrote these bullet points expanded them into a full paragraph.
We showed this paragraph to 10 independent raters and asked a RATING QUESTION.
For this rating question, your task is to predict the MEAN rating value across all raters.

RATING QUESTION:
writer_stance: What is the author's level of agreement with the proposition? [0-100 scale, where 0 = "Strongly disagree", 50 = "Ambivalent", and 100 = "Strongly agree"]

PROPOSITION: {proposition}

BULLET POINTS: {writer_bullets}
"""


class EvalResponse(BaseModel):
  writer_stance: float


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Score reward-model test examples with a fine-tuned OpenAI model."
  )
  parser.add_argument(
    "--dataset",
    choices=DATASET_CHOICES,
    default="paragraph_rm",
    help="Dataset under reranking/finetuning_data to score.",
  )
  parser.add_argument(
    "--model",
    default=DEFAULT_MODEL,
    help="Model id used for scoring. Can be a standard base model or a fine-tuned model.",
  )
  parser.add_argument(
    "--max-workers",
    type=int,
    default=DEFAULT_MAX_WORKERS,
    help="Maximum concurrent API requests.",
  )
  parser.add_argument(
    "--api-key-env",
    default=DEFAULT_API_KEY_ENV,
    help="Environment variable containing the OpenAI API key.",
  )
  parser.add_argument(
    "--base-url",
    default=DEFAULT_BASE_URL,
    help="OpenAI API base URL.",
  )
  parser.add_argument(
    "--max-tokens",
    type=int,
    default=256,
    help="Max tokens per completion.",
  )
  parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="Sampling temperature.",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite an existing final score file.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate inputs and output paths without calling the API.",
  )
  parser.add_argument(
    "--test-request",
    action="store_true",
    help="Send a single plain-text test request to the specified model and exit.",
  )
  return parser.parse_args()


def build_client(api_key_env: str, base_url: str) -> OpenAI:
  api_key = os.getenv(api_key_env)
  if not api_key:
    raise ValueError(f"Environment variable {api_key_env} is not set.")
  return OpenAI(base_url=base_url, api_key=api_key)


def model_slug(model_name: str) -> str:
  return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")


def dataset_path(dataset_name: str) -> Path:
  return FINETUNING_DATA_DIR / dataset_name / "test.csv"


def load_test_data(dataset_name: str) -> pd.DataFrame:
  path = dataset_path(dataset_name)
  if not path.exists():
    raise FileNotFoundError(f"Missing test file: {path}")
  return pd.read_csv(path)


def build_prompt(row: pd.Series, dataset_name: str) -> str:
  if dataset_name == "paragraph_rm":
    return PARAGRAPH_PROMPT_TEMPLATE.format(
      proposition=row["proposition"],
      paragraph=str(row["paragraph"]).strip(),
    )
  if dataset_name == "bullet_rm":
    return BULLET_PROMPT_TEMPLATE.format(
      proposition=row["proposition"],
      writer_bullets=str(row["writer_bullets"]).strip(),
    )
  raise ValueError(f"Unsupported dataset: {dataset_name}")


def with_eval_prompts(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
  prompted_df = df.copy()
  prompted_df["eval_prompt"] = prompted_df.apply(build_prompt, axis=1, dataset_name=dataset_name)
  return prompted_df


def split_batches(df: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
  return [df.iloc[index : index + batch_size].copy() for index in range(0, len(df), batch_size)]


class OpenAIModelScorer:
  def __init__(
    self,
    *,
    client: OpenAI,
    model_name: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
  ) -> None:
    self.client = client
    self.model_name = model_name
    self.max_tokens = max_tokens
    self.temperature = temperature
    self.max_retries = max_retries

  def _score_prompt_once(self, prompt: str) -> dict[str, str | float]:
    completion = self.client.chat.completions.parse(
      model=self.model_name,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=self.max_tokens,
      temperature=self.temperature,
      response_format=EvalResponse,
    )

    message = completion.choices[0].message
    parsed = getattr(message, "parsed", None)
    if parsed is None:
      raise ValueError("Structured response parsing failed.")

    response_text = message.content or ""
    if isinstance(response_text, list):
      response_text = "\n".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in response_text
      )

    return {
      "eval_text": str(response_text).strip(),
      "pred_writer_stance": float(parsed.writer_stance),
    }

  def test_request(self, prompt: str) -> str:
    completion = self.client.chat.completions.create(
      model=self.model_name,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=self.max_tokens,
      temperature=self.temperature,
    )

    message = completion.choices[0].message
    response_text = message.content or ""
    if isinstance(response_text, list):
      response_text = "\n".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in response_text
      )
    return str(response_text).strip()

  def score_prompt(self, prompt: str) -> dict[str, str | float]:
    last_error: Exception | None = None

    for attempt in range(1, self.max_retries + 1):
      try:
        return self._score_prompt_once(prompt)
      except Exception as error:
        last_error = error
        if attempt == self.max_retries:
          break
        backoff_seconds = min(10, 2 ** (attempt - 1))
        time.sleep(backoff_seconds)

    raise RuntimeError(
      f"Failed to score prompt after {self.max_retries} attempts."
    ) from last_error

  def score_prompts(self, prompts: list[str], max_workers: int) -> list[dict[str, str | float]]:
    return list(thread_map(self.score_prompt, prompts, max_workers=max_workers))


def collect_responses(batch_df: pd.DataFrame, scorer: OpenAIModelScorer, max_workers: int) -> pd.DataFrame:
  scored_df = batch_df.copy()
  results = scorer.score_prompts(scored_df["eval_prompt"].tolist(), max_workers=max_workers)
  results_df = pd.DataFrame(results)
  scored_df = pd.concat([scored_df.reset_index(drop=True), results_df], axis=1)
  scored_df["eval_model"] = scorer.model_name
  return scored_df


def final_output_path(dataset_name: str, model_name: str) -> Path:
  return RESULTS_DIR / dataset_name / f"{model_slug(model_name)}.csv"


def main() -> None:
  args = parse_args()

  test_df = with_eval_prompts(load_test_data(args.dataset), args.dataset)
  if test_df.empty:
    raise ValueError("No test rows found to score.")

  final_path = final_output_path(args.dataset, args.model)

  print(f"Dataset: {args.dataset}")
  print(f"Model: {args.model}")
  print(f"Test rows: {len(test_df)}")
  print(f"Final output path: {final_path}")
  print(f"Max workers: {args.max_workers}")
  if args.test_request:
    print(f"Test request prompt: {TEST_REQUEST_PROMPT}")

  if args.dry_run:
    return

  if final_path.exists() and not args.overwrite and not args.test_request:
    raise FileExistsError(
      f"Score file already exists: {final_path}. Use --overwrite to replace it."
    )

  client = build_client(api_key_env=args.api_key_env, base_url=args.base_url)
  scorer = OpenAIModelScorer(
    client=client,
    model_name=args.model,
    max_tokens=args.max_tokens,
    temperature=args.temperature,
    max_retries=DEFAULT_MAX_RETRIES,
  )

  if args.test_request:
    response_text = scorer.test_request(TEST_REQUEST_PROMPT)
    print("Test response:")
    print(response_text)
    return

  print("Collecting responses...")
  final_df = collect_responses(test_df, scorer, max_workers=args.max_workers)

  final_path.parent.mkdir(parents=True, exist_ok=True)
  final_df.to_csv(final_path, index=False)
  print(f"Saved scored responses: {final_path}")


if __name__ == "__main__":
  main()