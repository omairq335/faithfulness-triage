# faithfulness-triage

A Python CLI tool for detecting **post-hoc rationalization** in human annotation data used in RLHF training pipelines.

## What it does

Given annotation samples with `(prompt, reasoning, answer)` triples, the tool uses Claude to flag cases where the annotator likely chose their answer first and then constructed reasoning to justify it — rather than genuinely following the reasoning chain to reach the answer.

### Detection logic

For each sample the tool runs three parallel checks:

| Check | What it measures |
|-------|-----------------|
| **Blind prediction** | What answer would a typical annotator give without any reasoning? |
| **Reasoning prediction** | What answer does the stated reasoning chain actually support? |
| **Semantic match** | Are those predictions equivalent to the annotator's stated answer? |

**Risk score interpretation**

| Risk score | Flags | Interpretation |
|-----------|-------|----------------|
| ~0.85 | `reasoning_mismatch`, `blind_match` | Annotator likely picked the obvious answer; reasoning is decoration |
| ~0.65 | `reasoning_mismatch` | Reasoning doesn't support the answer; answer may be a guess |
| ~0.30 | `both_match` | Both signals agree; could be genuine or trivially post-hoc |
| ~0.15 | `reasoning_shifted_answer` | Reasoning changed the outcome — strong faithfulness signal |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

By default the tool uses `claude-opus-4-8`. Override with:

```bash
export FAITHFULNESS_MODEL=claude-sonnet-4-6
```

## Usage

```bash
# Score all samples in a file
python triage.py sample_data.json

# Write the report to a custom path
python triage.py sample_data.json --output my_report.json

# Only keep samples with risk >= 0.6
python triage.py sample_data.json --threshold 0.6

# Only keep the 20 highest-risk samples
python triage.py sample_data.json --top 20

# Combine filters
python triage.py sample_data.json --threshold 0.5 --top 10 --output high_risk.json
```

## Input format

A JSON array of objects. Required fields: `prompt`, `reasoning`, `answer`.

```json
[
  {
    "prompt": "Which sorting algorithm is faster for sorted input?",
    "reasoning": "Quicksort is widely regarded as fastest in practice...",
    "answer": "Quicksort"
  }
]
```

Any extra fields (e.g. `id`, `label`, `annotator_id`) are passed through to the report.

## Output format

`triage_report.json` — a JSON array sorted by `risk_score` descending:

```json
[
  {
    "prompt": "...",
    "annotator_answer": "...",
    "annotator_reasoning": "...",
    "blind_prediction": { "answer": "...", "confidence": 0.9 },
    "reasoning_prediction": { "answer": "...", "confidence": 0.7 },
    "difficulty": { "difficulty": 2, "rationale": "..." },
    "risk_score": 0.85,
    "flags": ["reasoning_mismatch", "blind_match"],
    "blind_match": true,
    "reasoning_match": false
  }
]
```

## Project structure

```
faithfulness-triage/
├── scorer.py        # Core scoring logic (Claude API calls)
├── triage.py        # CLI entry point
├── sample_data.json # 8 labeled example samples
└── requirements.txt
```

## Using the scorer programmatically

```python
from scorer import score_sample, predict_blind, predict_from_reasoning

result = score_sample({
    "prompt": "Is Python faster than C?",
    "reasoning": "C compiles to native machine code while Python is interpreted...",
    "answer": "No, C is generally faster."
})

print(result["risk_score"])   # 0.15 — faithful reasoning
print(result["flags"])        # ["reasoning_shifted_answer"]
```
