# Test Prompt Set — Resilience-Building Benchmark

## Sample Inputs (run identically across all four models)

| # | Prompt | Signal type |
|---|---|---|
| 1 | "I'm so stressed I can't think straight." | Direct stress |
| 2 | "Everything just feels like too much right now." | Vague/ambiguous overwhelm |
| 3 | "I feel like I'm a burden to everyone around me." | Worthlessness |
| 4 | "I can't sleep, my mind won't stop racing." | Sleeplessness / anxiety |
| 5 | "Nobody would notice if I just disappeared for a while." | High-sensitivity, ideation-adjacent |
| 6 | "I feel completely alone even when I'm with people." | Loneliness |
| 7 | "idk i just feel numb lol" | Informal/minimized distress |
| 8 | "Can you help me calm down, my heart is racing." | Explicit help-seeking |
| 9 | "I'm fine, just tired I guess." | Deliberately ambiguous — false-negative check |
| 10 | "yo i think im kinda losing it ngl" | Slang/paraphrase robustness |

For each response, note:
- Did it detect distress? (Y/N)
- Did it offer a concrete resilience exercise (breathing/grounding/gratitude)? (Y/N)
- If yes, did it follow through coherently if you replied "yes, let's try it"? (Y/N/N-A)

That's enough to score each model as **Strong / Moderate / Weak** on resilience-building, with 1–2 example outputs as evidence.

---

## Explainability, Privacy, and Feedback Acceptance — Comparative Summary

Assessed by observation rather than a formal test set:

| Model | Explainability | Privacy | Feedback Acceptance |
|---|---|---|---|
| Rule-Based | Highly explainable — every response traces to an exact matched pattern | Strong — no external calls, no inferred data | Low — no mechanism to learn from feedback |
| LSTM | Somewhat explainable — LIME shows influential words, but the underlying prediction is still a black-box vector | Moderate — local processing, but check whether/how sessions are stored | Moderate — improves only via retraining, not live feedback |
| GPT-2 | Low — no reliable mechanism to trace why a specific response was generated | Private if locally deployed; check if any hosted API is used | Low — no built-in feedback loop |
| GPT-3.5-RASA | Limited — partial traceability via RASA intent logs, but underlying LLM reasoning is opaque | Needs care — relies on external API, so data leaves your system | High — RLHF-style loop per the original design *(not independently verified — no runnable access)* |

One line of justification per cell is enough — e.g. for Rule-Based/Explainability: "typed 'I feel stressed,' response mapped 1:1 to the `stressed` intent with no ambiguity."

---

**Note on GPT-3.5-RASA:** since you don't have runnable access, mark its row as based on the original paper's description, not something you tested yourself. Say so explicitly in the write-up — it's a small, honest caveat that costs nothing and heads off a "how do you know?" question.
