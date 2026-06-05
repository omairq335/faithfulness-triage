# The Faithfulness Gap: Why Inter-Annotator Agreement Is a Necessary But Insufficient Condition for Training Data Quality

**Harris Qureshi** — April 2026

---

## Abstract

Current human data pipelines for frontier model training rely on inter-annotator agreement (IAA) as the primary quality signal. This memo argues that IAA is structurally blind to one of the most consequential failure modes in expert annotation: reasoning chains that are internally consistent and mutually agreed upon, but post-hoc in origin. These chains are constructed after a conclusion is reached by intuition or pattern matching rather than deliberate inference. I propose a lightweight triage framework that uses counterfactual prediction as a proxy faithfulness signal, designed not to replace IAA but to surface the subset of annotations most likely to encode rationalization rather than genuine reasoning. The framework is explicitly scoped as a triage tool to reduce the cost of targeted human auditing, not as a ground truth classifier.

---

## 1. The Problem IAA Was Designed to Solve, and the One It Was Not

Inter-annotator agreement measures whether multiple qualified annotators converge on the same label or reasoning chain for a given task. It is well-established that high IAA correlates with annotation reliability and downstream model performance. In practice, a Cohen's Kappa above 0.6 is considered acceptable agreement, with scores above 0.75 preferred in high-stakes domains like healthcare. ([Objectways, 2026](https://objectways.com/blog/improving-inter-rater-reliability-for-data-annotation-and-labeling/))

This works well when the failure mode is noise: annotators who disagree because of ambiguous instructions, insufficient domain expertise, or inconsistent application of guidelines. IAA catches all of these. Disagreement is the signal.

The problem is that IAA is designed to detect variance. It has no mechanism for detecting **systematic convergence on wrong reasoning**. If three cardiologists agree on a reasoning chain for a rare presentation, not because the chain accurately reflects how they reached the diagnosis, but because all three share the same intuitive pattern-matching heuristic and independently constructed the same post-hoc narrative, IAA scores perfect agreement. The data looks clean. It passes every filter. And the model trains on it.

This is not a hypothetical edge case. It is a structural property of expert cognition under conditions of genuine uncertainty.

---

## 2. The Research Basis for Reasoning Unfaithfulness

The faithfulness problem has been studied extensively at the model level. Turpin et al. (2023) demonstrated at NeurIPS that chain-of-thought explanations can systematically misrepresent the true reason for a model's prediction, specifically that CoT can be "heavily influenced by adding biasing features to model inputs" which "models systematically fail to mention in their explanations." ([Turpin et al., NeurIPS 2023](https://arxiv.org/abs/2305.04388))

This finding was extended by Arcuschin et al. (2025), who showed that unfaithful CoT occurs not just under artificially induced bias but in the wild on realistic prompts with no artificial manipulation, across both thinking and non-thinking frontier models. Their framing is direct: "When CoT reasoning is unfaithful, it undermines the reliability of these explanations, raising concerns in high-stakes settings, such as when this reasoning is incorporated into training designed to align models to human preferences." ([Arcuschin et al., 2025, arXiv:2503.08679](https://arxiv.org/abs/2503.08679))

What the model-level literature establishes is that verbalized reasoning cannot be assumed to be a faithful window into the process that generated the answer. This is not a model-specific bug. It is a property of any system, human or AI, that produces explanations after the fact.

Critically, the model-level faithfulness literature focuses on detecting unfaithfulness in model-generated chains. The **human data pipeline equivalent** — detecting when human annotators write reasoning chains that do not reflect their actual decision process — is structurally the same problem and is largely unaddressed in production pipelines.

Tutek et al. (2025), published at EMNLP, proposed measuring faithfulness by unlearning specific reasoning steps and observing whether model predictions change accordingly, using the causal contribution of each step as a faithfulness proxy. ([Tutek et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.504.pdf)) This parametric approach requires model internals and is not portable to human annotation auditing. The underlying intuition — that faithfulness should be measurable by causal perturbation — is directly transferable.

---

## 3. The Compounding Problem: Sycophancy Contamination

Faithfulness failure does not occur in isolation. It interacts with a second structural problem in human feedback pipelines: annotator sycophancy.

Shapira, Benade, and Procaccia (2026) provide a formal analysis showing that RLHF amplifies sycophancy through an explicit mechanism. Bias in human annotator preferences induces a reward gap that causes the learned policy to drift toward agreement over correctness. Their finding is that "the direction of behavioral drift is determined by a covariance under the base policy between endorsing the belief signal in the prompt and the learned reward." ([Shapira et al., 2026, arXiv:2602.01002](https://arxiv.org/abs/2602.01002))

The interaction with faithfulness is direct. When annotators evaluate reasoning chains written by other experts, they are susceptible to fluency bias, rating chains that read confidently and coherently as high quality regardless of whether the reasoning actually drove the conclusion. Sharma et al. (2023), cited in multiple downstream papers, documented that labelers naively prefer longer and more sycophantic responses regardless of quality. ([Referenced in Saito et al. and Sharma et al., as cited in arXiv:2501.05790](https://arxiv.org/html/2501.05790))

This creates a compounding failure. Post-hoc rationalizations, which by definition are constructed to be coherent and persuasive, are systematically rated higher than authentic but messier reasoning chains. The pipeline selects for the wrong thing and reinforces it.

---

## 4. Why IAA Cannot Solve This

A recent paper in the educational AI literature articulates the core limitation directly: "high IAA can mask annotation shallowness, promote premature consensus, and ignore valid alternative interpretations. Overreliance on IAA is common and problematic." ([Beyond Agreement: Rethinking Ground Truth in Educational AI Annotation, arXiv:2508.00143](https://arxiv.org/pdf/2508.00143))

The key distinction is between **reliability** and **validity**. IAA measures reliability, meaning whether annotators are consistent with each other. It does not measure validity, meaning whether the annotation actually captures what it claims to capture. For reasoning chain annotation, validity requires that the chain reflects the actual reasoning process. IAA is silent on this entirely.

Gold standard injection and spot audits address a related but different problem: whether annotators produce correct outputs on known-answer tasks. In professional judgment domains like medicine, law, finance, and complex logistics, there often is no known-answer ground truth for the cases that matter most. The whole point of expert annotation is to capture judgment in ambiguous territory. In exactly those cases, gold standard injection fails as a quality mechanism.

---

## 5. The Proposed Triage Framework

The framework is explicitly scoped as a **triage tool** — a mechanism to surface candidate annotations most likely to contain faithfulness failures for targeted human review. It makes no claim to be a ground truth faithfulness classifier.

### Core Heuristic: Counterfactual Prediction

The core signal operates as follows. For a given (prompt, reasoning chain, answer) tuple, an evaluator model is first presented the prompt alone and asked to generate a predicted answer without access to the reasoning chain. It is then presented the prompt plus the reasoning chain and asked to generate an answer again. If the answer is predictable from the prompt alone at high confidence, and the reasoning chain does not meaningfully shift the prediction or the confidence level, the chain provides no additional causal signal. It is a candidate for post-hoc rationalization and is flagged for review.

This is a behavioral analog to the causal perturbation approach in Tutek et al. (2025). Instead of unlearning model internals, the framework observes whether the reasoning chain causally shifts the prediction. When it does not, the sample is flagged.

### Known Confound and Mitigation

The primary confound is task difficulty. Easy tasks will be predictable without reasoning regardless of whether the chain is post-hoc. The framework controls for this by running a difficulty pre-classification step before scoring. Only samples above a difficulty threshold — where the prompt alone should not be sufficient to predict the answer — are flagged when the chain adds no predictive value. This is not a fully solved problem, but it reduces the false positive rate substantially on straightforward tasks.

### LLM-as-Judge Circularity and Mitigation

Using an LLM to evaluate LLM-adjacent outputs introduces its own bias. The mitigation is ensemble disagreement. Multiple models are run on each sample and where models disagree on the faithfulness signal, that disagreement itself is flagged as a quality concern warranting human review. No single model's output is treated as ground truth.

### Output

The framework produces a per-sample triage report with three signals: a faithfulness risk score that is continuous rather than binary, a difficulty estimate used to contextualize the risk score, and an ensemble disagreement flag. Samples with a high risk score, high difficulty estimate, and strong ensemble agreement represent the strongest candidates for targeted human audit. The framework does not label samples as faithful or unfaithful. It ranks them by audit priority.

---

## 6. Practical Value Proposition

The economic framing is straightforward. Human auditing at scale is expensive. If a faithfulness triage layer can identify the 10 to 15 percent of samples most likely to contain reasoning failures, the cost of targeted expert review of that subset is substantially lower than auditing the full corpus. The framework only needs to outperform random selection of samples for review, not achieve high precision on its own, to justify the compute overhead.

For frontier labs running training runs at the scale AfterQuery services, the asymmetry is significant. The cost of training on confidently wrong reasoning data, multiplied by the scale of the run, dwarfs the inference cost of a triage pass over the annotation corpus.

---

## 7. Limitations and Open Questions

This framework has real limitations that are worth stating directly. The counterfactual prediction heuristic is a proxy, not a direct measure of faithfulness. It detects cases where reasoning is causally inert, not cases where reasoning is actively misleading in a subtler way. Difficulty calibration is underdeveloped, and reliable automated difficulty estimation in professional judgment domains remains an open problem. Validation is the hardest part: without a ground truth dataset of known post-hoc rationalizations to validate against, the framework's precision and recall cannot be rigorously established. Building that validation set is a research project in itself. Finally, the framework is most useful for reasoning-heavy expert annotation tasks and adds minimal value on simpler classification or preference ranking tasks where IAA is already sufficient.

---

## 8. Conclusion

IAA is a necessary condition for annotation quality, not a sufficient one. The specific gap it cannot address is expert consensus on post-hoc reasoning — the case where qualified annotators converge on a plausible and coherent chain that did not actually drive their conclusion. This failure mode is invisible to every standard quality check in production pipelines and is structurally likely to worsen as data companies scale expert networks and increase annotation volume in ambiguous professional domains.

The triage framework proposed here is a first-order response to that gap. It is modest in its claims, honest about its limitations, and designed to reduce the cost of targeted human auditing rather than replace it. The immediate next step is building a working prototype to test the counterfactual prediction heuristic on an existing annotated dataset with known difficulty characteristics, and measuring how faithfulness risk scores correlate with downstream model behavior on held-out tasks.

---

## References

- Turpin, M., Michael, J., Perez, E., & Bowman, S. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. *NeurIPS 2023*. <https://arxiv.org/abs/2305.04388>

- Arcuschin, I., et al. (2025). Chain-of-Thought Reasoning In The Wild Is Not Always Faithful. *arXiv:2503.08679*. <https://arxiv.org/abs/2503.08679>

- Tutek, M., et al. (2025). Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps. *EMNLP 2025*. <https://aclanthology.org/2025.emnlp-main.504.pdf>

- Shapira, I., Benade, G., & Procaccia, A.D. (2026). How RLHF Amplifies Sycophancy. *arXiv:2602.01002*. <https://arxiv.org/abs/2602.01002>

- Beyond Agreement: Rethinking Ground Truth in Educational AI Annotation. (2025). *arXiv:2508.00143*. <https://arxiv.org/pdf/2508.00143>

- Understanding Impact of Human Feedback via Influence Functions. (2025). *arXiv:2501.05790*. <https://arxiv.org/html/2501.05790>

- Objectways. (2026). Improving Inter Rater Reliability for Data Annotation. <https://objectways.com/blog/improving-inter-rater-reliability-for-data-annotation-and-labeling/>
