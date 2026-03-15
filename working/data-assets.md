# Data Assets

Shipped data files for reproducing and extending the v1 routing checkpoint.

---

## Dataset Governance

### What is the name of the dataset?

**Model Router Toolkit v1 Dataset** — training and evaluation data for the prefill-based LLM routing checkpoint.

The dataset has two components:

| Component | Files |
|-----------|-------|
| Training labels | `data/train_v1.csv` |
| Test labels | `data/test_v1.csv` |

### What does the dataset contain?

**Label CSVs** (`train_v1.csv`, `test_v1.csv`): Each row records whether a specific LLM answered a benchmark question correctly. Columns are `question` (full text), `model` (which of 9 LLMs was tested), `isCorrect` (binary), `output_tokens` (token count), and `embedding_id` (integer index). The 12,299 training questions and 2,170 test questions are drawn from three public academic benchmarks: MMLU Pro, LiveCodeBench, and Humanity's Last Exam (HLE). Each question is evaluated against all 9 models, producing 110,691 training rows and 19,530 test rows.

**Personal data:** The dataset contains **no personal data**. All questions come from public academic benchmarks (math, science, coding, trivia). A regex scan for emails, phone numbers, and SSNs found zero real matches (19 phone-pattern hits were all large integers in math/coding problems, e.g. `1000000000`, `1234567890` in fictional legal scenarios). No names, addresses, or other PII are present beyond what appears in publicly published benchmark questions.

**Confidential data:** The dataset contains **no NVIDIA-confidential data**. The questions are from public benchmarks. The model outputs (answers) are not included — only a binary correctness flag. The feature files are dimensionality-reduced numerical vectors with no proprietary model internals.

### How will we obtain the dataset?

The label CSVs were produced internally by running 9 LLMs (via OpenRouter API) on questions from three public benchmarks and judging correctness.

**Source benchmark licenses:**

| Benchmark | License | Source |
|-----------|---------|--------|
| MMLU Pro | Apache 2.0 | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| LiveCodeBench | MIT | [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) |
| Humanity's Last Exam (HLE) | MIT | [cais/hle](https://huggingface.co/datasets/cais/hle) |

**License restrictions of note:** HLE's README requests that users "not publicly share, re-upload, or distribute the dataset" to preserve benchmark integrity. Our CSVs include the full question text from HLE, so this redistribution concern applies. MMLU Pro (Apache 2.0) and LiveCodeBench (MIT) have no such restrictions.

### How large is the dataset?

| File | Size |
|------|-----:|
| `data/train_v1.csv` | 77 MB |
| `data/test_v1.csv` | 14 MB |
| **Total** | **91 MB** |

All files are tracked via Git LFS.

### Will we share the dataset?

Yes. These files ship in the public repository as Git LFS objects to enable reproducibility. Users who clone the repo and run `git lfs pull` will receive the label CSVs.

### How will we use the dataset?

The dataset serves two purposes:

1. **Training** — Users run `model-router train` with the label CSVs to train a routing checkpoint from scratch.
2. **Evaluation** — Users run `model-router evaluate` against `test_v1.csv` to verify router accuracy, cost savings, and per-model AUC before deploying.

The label CSVs are also used as reference data when training custom routers on different model pools or encoder configurations.

### Did you modify the dataset?

Yes. We made the following transformations from the raw source data:

1. **Question selection and deduplication** — Questions were sampled from MMLU Pro, LiveCodeBench, and HLE, then deduplicated by content.
2. **Correctness labeling** — Each question was sent to all 9 models via the OpenRouter API. Model outputs were judged for correctness (vote-based judging for open-ended questions). Only the binary `isCorrect` flag and `output_tokens` count were retained; the actual model-generated answers were discarded.
3. **Train/test split** — The 14,469 total questions were split into 12,299 train and 2,170 test.

---

## Detailed Contents

### Label CSVs

**Columns:** `question`, `model`, `isCorrect`, `output_tokens`, `embedding_id`

**9-model pool** (cost range $0.05 -- $2.77 per M input tokens):

| Model | Train Accuracy | Test Accuracy |
|-------|---------------:|--------------:|
| nemotron-3-nano-reasoning | 69.0% | 67.7% |
| gpt-oss-20b-high | 65.0% | 64.5% |
| nemotron-3-super | 70.7% | 70.4% |
| gpt-oss-120b-high | 67.3% | 67.4% |
| qwen-3-5-35b | 73.6% | 74.0% |
| qwen-3-5-122b | 76.2% | 76.5% |
| gpt-5-2-high | 77.5% | 77.2% |
| gpt-5-4-high | 79.9% | 80.1% |
| claude-opus-4-6-high | 80.9% | 82.2% |

