# Reranking to Mitigate Distortions from AI Writing Assistance

This directory contains the data-preparation, fine-tuning, scoring, and evaluation scripts used for the Reranking experiments.

The workflow trains Reward Models (RMs) to predict human `writer_stance` ratings from either full paragraphs (Paragraph RM) or writer bullet points (Bullet RM), then evaluates those RMs on a held-out test split.

For fine-tuning, we use the OpenAI fine-tuning API, as documented [here](https://developers.openai.com/api/docs/guides/supervised-fine-tuning).

## Setup

To make API calls, you need a `.env` file in the repository root with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Reward Models

The reranking pipeline currently supports training and evaluating two types of Reward Models (RMs):
- `paragraph_rm` predicts the mean `writer_stance` score from a full opinion paragraph.
- `bullet_rm` predicts the mean `writer_stance` score from the writer's original bullet points.

Since the models operate on different input data, they are trained and evaluated separately, but the same workflow applies to both.

## Workflow

### 1. Prepare fine-tuning data

This builds train, validation, and test splits for each RM and exports both CSV and JSONL files.

```bash
env/bin/python reranking/1_prepare_finetuning.py
```

Outputs for each RM dataset include:
- `train.jsonl`, `valid.jsonl`, `test.jsonl`
- `train_<n>.jsonl` subsamples for selected training sizes
- matching CSV files for inspection

The split is done by `writer_id`, so the same writer does not appear across train, validation, and test.

### 2. Launch a fine-tuning job

This uploads the prepared training and validation JSONL files and creates an OpenAI fine-tuning job.

Example:

```bash
env/bin/python reranking/2_launch_finetuning.py \
	--dataset paragraph_rm \
	--train-size 1000 \
	--model gpt-4.1-nano-2025-04-14
```

Useful options:
- `--dataset {bullet_rm,paragraph_rm}` 
- `--train-size <n>` selects `train_<n>.jsonl`
- `--model <base-model>` chooses the base model to fine-tune
- `--list-jobs` prints existing fine-tuning jobs before creating a new one

By default, the fine-tuned model name suffix is `<dataset>-<train-size>`, e.g. `paragraph_rm-1000`. You can customize this with the `--model-suffix` option.

Fine-tuned models will be listed in the OpenAI dashboard and accessible with the returned model name, e.g. `your-finetuned-model`.

### 3. Score the held-out test set

Once you have a base model (e.g. `gpt-4.1-nano-2025-04-14`) or fine-tuned model (`your-finetuned-model`), score the test split:

```bash
env/bin/python reranking/3_get_rm_scores.py \
	--dataset paragraph_rm \
	--model your-finetuned-model
```

The score file is written to:

```text
reranking/rm_scores/<dataset>/<model-slug>.csv
```

Useful options:

- `--dry-run` validates inputs and output paths without calling the API
- `--test-request` sends a single plain-text request to confirm the model is reachable
- `--overwrite` replaces an existing score file
- `--max-workers <n>` controls concurrent API requests

### 4. Evaluate score files

This compares predicted `writer_stance` values against the held-out test labels.

```bash
env/bin/python reranking/4_evaluate_rms.py --dataset paragraph_rm
```

By default, the script evaluates all CSV score files in `reranking/rm_scores/<dataset>/`.

Outputs are written to:

```text
reranking/results/rm_evaluations/<dataset>/
```

Key outputs:

- `summary.csv` with aggregate metrics for each scored model
- one merged CSV per model with predictions joined back to the test set

Reported metrics include:

- mean absolute error
- root mean squared error
- Pearson correlation
- three-way stance bucket accuracy
- three-way stance bucket macro F1

## Typical End-to-End Run

```bash
source env/bin/activate
env/bin/python reranking/1_prepare_finetuning.py
env/bin/python reranking/2_launch_finetuning.py --dataset paragraph_rm --train-size 1000 --model gpt-4.1-nano-2025-04-14
env/bin/python reranking/3_get_rm_scores.py --dataset paragraph_rm --model your-finetuned-model
env/bin/python reranking/4_evaluate_rms.py --dataset paragraph_rm
```

## Notes

- The preparation step currently targets the `writer_stance` annotation only.
- Fine-tuning and scoring both require valid OpenAI credentials in the root `.env` file.
- If you want to inspect paths before making API calls, use the `--dry-run` options on the scoring and evaluation scripts.