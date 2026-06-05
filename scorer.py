import os
import json
import anthropic

MODEL = os.environ.get("FAITHFULNESS_MODEL", "claude-opus-4-8")

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _call(system: str, user: str, max_tokens: int = 512) -> str:
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def _parse_json(text: str, fallback: dict) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return fallback


def predict_blind(prompt: str) -> dict:
    """Predict what answer a typical annotator would give without seeing any reasoning."""
    system = (
        "You are simulating a human annotator evaluating AI outputs. "
        "Given only a prompt or question, predict what answer a typical careful annotator "
        "would choose. Respond ONLY with a JSON object in this exact format:\n"
        '{"answer": "<predicted answer>", "confidence": <0.0 to 1.0>}'
    )
    raw = _call(system, f"Prompt:\n{prompt}")
    return _parse_json(raw, {"answer": raw.strip()[:200], "confidence": 0.5})


def predict_from_reasoning(prompt: str, reasoning: str) -> dict:
    """Determine what answer the given reasoning chain actually supports."""
    system = (
        "You are a logical consistency evaluator. "
        "Given a prompt and a chain of reasoning, determine what answer or conclusion "
        "the reasoning logically supports — regardless of what anyone actually claimed. "
        "Respond ONLY with a JSON object in this exact format:\n"
        '{"answer": "<answer the reasoning supports>", "confidence": <0.0 to 1.0>}'
    )
    raw = _call(system, f"Prompt:\n{prompt}\n\nReasoning chain:\n{reasoning}")
    return _parse_json(raw, {"answer": raw.strip()[:200], "confidence": 0.5})


def answers_match(a1: str, a2: str, context: str = "") -> bool:
    """Ask whether two answers are semantically equivalent given the context."""
    system = (
        "You are a semantic equivalence judge. "
        "Determine whether two answers express the same conclusion in context. "
        "Minor wording differences are fine; focus on whether they agree on the core claim. "
        "Respond ONLY with a JSON object: "
        '{"match": true} or {"match": false}'
    )
    user = f"Context/question: {context}\n\nAnswer A: {a1}\nAnswer B: {a2}"
    raw = _call(system, user, max_tokens=128)
    result = _parse_json(raw, {"match": False})
    return bool(result.get("match", False))


def assess_reasoning_faithfulness(prompt: str, reasoning: str, answer: str) -> dict:
    """Directly assess whether a reasoning chain is genuine or post-hoc.

    Evaluates four criteria:
    - Specificity: does it engage with the prompt's specific facts/numbers/constraints?
    - Necessity: could the same reasoning support the opposite conclusion?
    - Coverage: does it address the key discriminating factor(s)?
    - Validity: is it logically sound, or does it contain fallacies or unsupported leaps?
    """
    system = (
        "You are an expert in detecting post-hoc rationalization in human annotation data.\n\n"
        "Post-hoc rationalization: the annotator chose an answer based on intuition or bias, "
        "then wrote reasoning that sounds plausible but does not genuinely justify the conclusion.\n\n"
        "Evaluate the reasoning on four criteria:\n"
        "1. SPECIFICITY — Does it engage with the specific facts, numbers, or constraints in the "
        "prompt, or only use generic principles that would apply to any similar question?\n"
        "2. NECESSITY — Could the same reasoning, with minimal changes, support the opposite conclusion?\n"
        "3. COVERAGE — Does it identify and address the key discriminating factor(s) in the prompt?\n"
        "4. VALIDITY — Is it logically sound, or does it contain fallacies, unsupported leaps, "
        "or selective use of evidence?\n\n"
        "Return ONLY a JSON object:\n"
        '{"faithfulness_score": <0.0 (post-hoc) to 1.0 (genuine)>, '
        '"issues": ["<specific issue>", ...], '
        '"key_factors_addressed": <true|false>}'
    )
    user = f"Prompt:\n{prompt}\n\nReasoning:\n{reasoning}\n\nStated answer:\n{answer}"
    raw = _call(system, user, max_tokens=512)
    return _parse_json(
        raw,
        {"faithfulness_score": 0.5, "issues": [], "key_factors_addressed": True},
    )


def estimate_difficulty(prompt: str) -> dict:
    """Estimate annotation difficulty on a 1–5 scale."""
    system = (
        "Rate how difficult this question is for a human annotator on a 1–5 scale:\n"
        "1 = trivially easy  2 = easy  3 = moderate  4 = hard  5 = expert-level\n"
        "Respond ONLY with JSON: "
        '{"difficulty": <1-5>, "rationale": "<one sentence>"}'
    )
    raw = _call(system, prompt, max_tokens=256)
    return _parse_json(raw, {"difficulty": 3, "rationale": ""})


def compute_risk_score(
    annotator_answer: str,
    blind_pred: dict,
    reasoning_pred: dict,
    faithfulness: dict,
    prompt: str,
) -> dict:
    """Compute a post-hoc rationalization risk score (0.0 = low, 1.0 = high).

    Primary signal: faithfulness_score from assess_reasoning_faithfulness (inverted).
    Secondary signal: ±0.10 based on whether reasoning shifted the blind prediction.
    """
    faithfulness_score = float(faithfulness.get("faithfulness_score", 0.5))
    base_risk = 1.0 - faithfulness_score

    # Does the reasoning chain actually change what you'd predict compared to no reasoning?
    # If not, the reasoning added no new information — a weak signal toward post-hoc.
    predictions_agree = answers_match(blind_pred["answer"], reasoning_pred["answer"], prompt)
    shift_modifier = 0.10 if predictions_agree else -0.10

    risk_score = round(max(0.0, min(1.0, base_risk + shift_modifier)), 2)

    flags = []
    if predictions_agree:
        flags.append("reasoning_no_shift")
    else:
        flags.append("reasoning_shifted_prediction")
    flags.extend(faithfulness.get("issues", []))

    return {
        "risk_score": risk_score,
        "flags": flags,
        "faithfulness_score": faithfulness_score,
        "key_factors_addressed": faithfulness.get("key_factors_addressed", True),
    }


def score_sample(sample: dict) -> dict:
    """Run the full faithfulness-triage pipeline on one annotation sample."""
    prompt = sample["prompt"]
    reasoning = sample.get("reasoning", "")
    annotator_answer = sample["answer"]

    blind = predict_blind(prompt)
    from_reasoning = predict_from_reasoning(prompt, reasoning)
    difficulty = estimate_difficulty(prompt)
    faithfulness = assess_reasoning_faithfulness(prompt, reasoning, annotator_answer)
    risk = compute_risk_score(annotator_answer, blind, from_reasoning, faithfulness, prompt)

    return {
        "prompt": prompt,
        "annotator_answer": annotator_answer,
        "annotator_reasoning": reasoning,
        "blind_prediction": blind,
        "reasoning_prediction": from_reasoning,
        "difficulty": difficulty,
        **risk,
    }
