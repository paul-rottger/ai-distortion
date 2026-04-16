#!/usr/bin/env python3

import argparse
import datetime as dt
import os
from pathlib import Path

import dotenv
from openai import OpenAI

dotenv.load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
FINETUNING_DATA_DIR = SCRIPT_DIR / "finetuning_data"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"
DATASET_CHOICES = ("bullet_rm", "paragraph_rm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an OpenAI fine-tuning job for a prepared reward-model dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=None,
        help="Dataset under reranking/finetuning_data to use.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=1000,
        help="Training subset size. The script expects train_<size>.jsonl to exist.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Optional explicit fine-tuning suffix. Defaults to <dataset>-<train-size>.",
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
        "--list-jobs",
        action="store_true",
        help="List existing fine-tuning jobs before creating a new one.",
    )
    return parser.parse_args()


def build_client(api_key_env: str, base_url: str) -> OpenAI:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env} is not set.")
    return OpenAI(base_url=base_url, api_key=api_key)


def list_jobs(client: OpenAI) -> None:
    jobs = client.fine_tuning.jobs.list()
    for job in jobs.data:
        created_at = dt.datetime.fromtimestamp(job.created_at, tz=dt.timezone.utc).astimezone()
        print(created_at.isoformat(), job.id, job.status)


def dataset_dir(dataset_name: str) -> Path:
    return FINETUNING_DATA_DIR / dataset_name


def available_training_sizes(dataset_name: str) -> list[int]:
    sizes: list[int] = []
    for path in dataset_dir(dataset_name).glob("train_*.jsonl"):
        suffix = path.stem.removeprefix("train_")
        if suffix.isdigit():
            sizes.append(int(suffix))
    return sorted(sizes)


def resolve_data_paths(dataset_name: str, train_size: int) -> tuple[Path, Path]:
    base_dir = dataset_dir(dataset_name)
    train_path = base_dir / f"train_{train_size}.jsonl"
    valid_path = base_dir / "valid.jsonl"

    if not train_path.exists():
        available_sizes = available_training_sizes(dataset_name)
        raise FileNotFoundError(
            f"Missing training file: {train_path}. Available train sizes: {available_sizes}"
        )
    if not valid_path.exists():
        raise FileNotFoundError(f"Missing validation file: {valid_path}")

    return train_path, valid_path


def default_suffix(dataset_name: str, train_size: int) -> str:
    return f"{dataset_name.replace('_', '-')}-{train_size}"


def upload_training_files(client: OpenAI, train_path: Path, valid_path: Path) -> tuple[str, str]:
    with train_path.open("rb") as train_handle:
        train_file = client.files.create(file=train_handle, purpose="fine-tune")
    with valid_path.open("rb") as valid_handle:
        valid_file = client.files.create(file=valid_handle, purpose="fine-tune")
    return train_file.id, valid_file.id


def create_finetuning_job(
    client: OpenAI,
    model: str,
    training_file_id: str,
    validation_file_id: str,
    suffix: str,
):
    return client.fine_tuning.jobs.create(
        model=model,
        training_file=training_file_id,
        validation_file=validation_file_id,
        suffix=suffix,
    )


def main() -> None:
    args = parse_args()
    client = build_client(api_key_env=args.api_key_env, base_url=args.base_url)

    if args.list_jobs:
        list_jobs(client)

    train_path, valid_path = resolve_data_paths(args.dataset, args.train_size)
    training_file_id, validation_file_id = upload_training_files(client, train_path, valid_path)

    suffix = args.suffix or default_suffix(args.dataset, args.train_size)
    job = create_finetuning_job(
        client=client,
        model=args.model,
        training_file_id=training_file_id,
        validation_file_id=validation_file_id,
        suffix=suffix,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Train file: {train_path}")
    print(f"Validation file: {valid_path}")
    print(f"Training file id: {training_file_id}")
    print(f"Validation file id: {validation_file_id}")
    print(f"Job: {job.id}")


if __name__ == "__main__":
    main()
