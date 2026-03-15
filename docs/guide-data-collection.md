# Data Collection Guide

This guide covers how to collect the labeled training data that the router learns from. Good data is the most important factor in routing quality — more important than hyperparameter tuning or model architecture.

---

## Table of Contents

- [What the Router Needs to Learn](#what-the-router-needs-to-learn)
- [Data Format](#data-format)
- [The Collect Command](#the-collect-command)
- [Judging Methods](#judging-methods)
- [Designing Your Question Set](#designing-your-question-set)
- [Splitting Train and Test](#splitting-train-and-test)
- [Manual Data Preparation](#manual-data-preparation)
- [Data Quality Checklist](#data-quality-checklist)
- [Troubleshooting](#troubleshooting)

---

## What the Router Needs to Learn

The router predicts, for each question, which models will answer correctly and which won't. To learn this, it needs labeled examples: questions where we know the ground truth correctness of every model in the pool.

Each training example says: "For question Q, model M produced a correct/incorrect answer."

The more examples, and the more diverse the questions, the better the router learns to distinguish straightforward questions (where lightweight models maintain accuracy) from complex ones (where more capable models are needed).

---

## Data Format

The training CSV has four columns:

| Column | Type | Description |
|--------|------|-------------|
| `question` | string | The question text |
| `model` | string | Model name (must match the `name` field in your pool config YAML) |
| `isCorrect` | int | `1` if the model's answer was correct, `0` if not |
| `output_tokens` | int | Number of tokens in the model's response |

Each question should have **one row per model** in the pool. For a 9-model pool with 500 questions, you'd have 4,500 rows.

Example:

```csv
question,model,isCorrect,output_tokens
What is the capital of France?,nemotron-3-nano-reasoning,1,45
What is the capital of France?,gpt-oss-20b-high,1,38
What is the capital of France?,claude-opus-4-6-high,1,52
Prove the Riemann hypothesis is true,nemotron-3-nano-reasoning,0,312
Prove the Riemann hypothesis is true,gpt-oss-20b-high,0,287
Prove the Riemann hypothesis is true,claude-opus-4-6-high,0,445
```

---

## The Collect Command

The `model-router collect` command automates data collection: it runs every model in your pool on each question and judges correctness.

### Prerequisites

```bash
pip install -e '.[prefill,training]'
export OPENROUTER_API_KEY=your-key  # or NVIDIA_API_KEY
```

### Basic usage

Create a `questions.txt` file with one question per line:

```
What is the capital of France?
Explain the difference between TCP and UDP.
Write a Python function to find the nth Fibonacci number.
What are the implications of Gödel's incompleteness theorems?
```

Run collection:

```bash
model-router collect \
  --config configs/v1-9models-qwen08b.yaml \
  --questions questions.txt \
  --output data/collected.csv \
  --judge vote
```

This will:
1. Load the pool config (9 models)
2. For each question, call each model via LiteLLM
3. Judge correctness using majority vote
4. Write results to `data/collected.csv`
5. Print per-model accuracy summary

### Cost considerations

Collection calls every model on every question. For a 9-model pool:

| Questions | API calls | Approximate cost (with default pool) |
|-----------|-----------|--------------------------------------|
| 100 | 900 | ~$1–5 |
| 500 | 4,500 | ~$5–25 |
| 1,000 | 9,000 | ~$10–50 |
| 5,000 | 45,000 | ~$50–250 |

Costs vary significantly by model and response length. The expensive models (GPT-5.4, Claude Opus) dominate the cost.

---

## Judging Methods

### Majority Vote (`--judge vote`)

No ground truth needed. All models answer the same question, and the majority answer is treated as correct. Models whose (normalized) answers match the majority are marked correct.

```bash
model-router collect --judge vote ...
```

**How it works**:
1. All model outputs are normalized (lowercased, whitespace-collapsed)
2. The most common normalized answer across all models is the "vote winner"
3. Models matching the vote winner get `isCorrect=1`, others get `isCorrect=0`

**Strengths**:
- No preparation needed beyond the questions themselves
- Works well when most models agree on correct answers
- Naturally handles open-ended questions

**Weaknesses**:
- Breaks down when the majority is wrong (rare but possible on very hard questions)
- Not suitable for creative tasks where "correct" is subjective
- Normalization can miss semantically equivalent but textually different answers

### Reference-Based (`--judge reference`)

Uses ground truth answers you provide.

```bash
model-router collect --judge reference --references answers.csv ...
```

The references CSV needs two columns: `question` and `answer`. Questions must match exactly between `questions.txt` and the references CSV.

```csv
question,answer
What is the capital of France?,Paris
What is 2+2?,4
```

**How it works**:
1. Model output is normalized (lowercased, whitespace-collapsed)
2. Reference answer is normalized the same way
3. Exact string match after normalization determines correctness

**Strengths**:
- Ground truth is unambiguous
- Works even when most models get the answer wrong
- Good for factual, closed-form questions

**Weaknesses**:
- Requires curating reference answers
- Exact match can be too strict (e.g., "Paris, France" vs "Paris")
- Not suitable for open-ended or creative questions

### LLM-as-Judge (`--judge llm`)

**Not yet implemented.** Planned to use a strong LLM (e.g., GPT-5.4) to judge correctness of other models' outputs.

---

## Designing Your Question Set

The quality and diversity of your questions directly determines routing quality. Here's how to build a good set.

### Question volume

| Pool size | Minimum | Recommended | Ideal |
|-----------|---------|-------------|-------|
| 2–3 models | 200 | 500 | 1,000+ |
| 5–9 models | 500 | 1,000 | 3,000+ |
| 10+ models | 1,000 | 2,000 | 5,000+ |

More models need more data because the router needs to learn fine-grained distinctions between similar models.

### Difficulty distribution

Aim for a mix of difficulties:

| Difficulty | Target % | Why |
|------------|----------|-----|
| Easy (all models correct) | 30–40% | Router learns to route cheaply |
| Medium (some correct, some not) | 30–40% | This is where routing adds the most value |
| Hard (few or no models correct) | 20–30% | Router learns the limits of the pool |

If your questions are too easy (>80% all-correct), the router has little to learn — any model works. If too hard (>50% all-wrong), there's no routing opportunity.

### Topic diversity

Cover the domains your application will encounter:

- Factual recall ("What year was X invented?")
- Reasoning ("If A implies B, and B implies C, what can we conclude?")
- Math ("Solve for x: 3x + 7 = 22")
- Coding ("Write a function that reverses a linked list")
- Creative ("Write a haiku about machine learning")
- Multi-step ("Plan a 3-day itinerary for Tokyo")

The router can only route well on question types it has seen during training.

### Avoid pitfalls

- **No duplicate questions**: Duplicates inflate metrics and waste API budget
- **No trivially identical questions**: "What is 2+2?" and "What's 2+2?" are effectively the same
- **Representative of production**: If your app serves coding questions, don't train on trivia
- **Avoid very long questions**: The encoder has a max token limit (2048 by default). Very long questions get truncated.

---

## Splitting Train and Test

After collecting, split the data into training and test sets. The critical rule: **no question should appear in both sets.**

### Why question-level splits matter

If you split at the row level (randomly assigning rows), the same question can end up in both train and test sets with different models. This leaks information and inflates evaluation metrics.

Always split at the **question level**: all rows for a given question go to either train or test, never both.

### Split script

```python
import csv
import random
from collections import defaultdict

rows_by_q = defaultdict(list)
with open("data/collected.csv") as f:
    for row in csv.DictReader(f):
        rows_by_q[row["question"]].append(row)

questions = list(rows_by_q.keys())
random.seed(42)
random.shuffle(questions)

split = int(len(questions) * 0.8)
train_qs = set(questions[:split])
test_qs = set(questions[split:])

for name, qs in [("data/train.csv", train_qs), ("data/test.csv", test_qs)]:
    with open(name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "model", "isCorrect", "output_tokens"])
        w.writeheader()
        for q in sorted(qs):
            w.writerows(rows_by_q[q])

print(f"Train: {len(train_qs)} questions, Test: {len(test_qs)} questions")
```

### Recommended split

80% train / 20% test is standard. For small datasets (<300 questions), consider 70/30.

---

## Manual Data Preparation

If you already have labeled data (e.g., from internal evaluation pipelines), you can skip the `collect` command and create the CSV manually. Just ensure:

1. The CSV has columns: `question`, `model`, `isCorrect`, `output_tokens`
2. The `model` column values exactly match the `name` fields in your pool config YAML
3. Every question has one row per model
4. `isCorrect` is `1` or `0`
5. `output_tokens` is a positive integer (used for cost estimation; if unknown, use a reasonable estimate like 200)

---

## Data Quality Checklist

Before training, verify your data:

- [ ] **Volume**: At least 500 questions (for a 9-model pool)
- [ ] **Coverage**: Every model has rows in the CSV
- [ ] **Balance**: Not all questions are easy or all hard
- [ ] **No duplicates**: No question appears more than once per model
- [ ] **Name match**: Model names in CSV match config YAML exactly
- [ ] **Split properly**: Train and test sets have no overlapping questions
- [ ] **Diverse topics**: Mix of difficulty levels and subject areas

Quick check:

```python
import csv
from collections import Counter

with open("data/train.csv") as f:
    rows = list(csv.DictReader(f))

# Check model coverage
models = Counter(r["model"] for r in rows)
print("Models:", dict(models))

# Check question count
questions = set(r["question"] for r in rows)
print(f"Unique questions: {len(questions)}")

# Check correctness distribution per model
for model in sorted(models):
    model_rows = [r for r in rows if r["model"] == model]
    correct = sum(int(r["isCorrect"]) for r in model_rows)
    print(f"  {model}: {correct}/{len(model_rows)} correct ({100*correct/len(model_rows):.0f}%)")
```

---

## Troubleshooting

### "Model X has 0% accuracy"

This usually means the judging method isn't working for that model. Common causes:
- The model produces very verbose output that doesn't match the majority vote after normalization
- The model uses a different answer format (e.g., "The answer is 4" vs "4")
- Try reference-based judging with explicit ground truth

### "All models have the same accuracy"

Your questions might be too easy or too hard. Check the agreement zone distribution:
- High "all correct" → questions are too easy, add harder ones
- High "all wrong" → questions are too hard, add easier ones
- You want at least 20% in the "disagree" zone for meaningful routing

### "Collection is too expensive"

Options:
- Start with fewer questions (200–300) and iterate
- Remove the most expensive models from the pool during collection
- Use a smaller pool (3–5 models) for initial experiments
- Collect in batches to control spend

### "API rate limits"

The collect command calls models sequentially. If you hit rate limits:
- The LiteLLM layer handles retries automatically
- For aggressive rate limits, you may need to add delays between questions
- Consider collecting during off-peak hours
