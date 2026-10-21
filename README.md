# ResiliBot — Dynamic Resilience Building in Mental Health Chatbots

A comparative study and implementation of four different chatbot architectures for
mental health support: **Rule-Based**, **LSTM**, **fine-tuned GPT-2**, and
**GPT-3.5 Turbo + RASA**. Built to explore a core tension in applied ML: as models
get more capable and adaptive, they tend to get harder to explain — and in a
sensitive domain like mental health, that tradeoff actually matters.

## Research Question

How can we design mental health chatbots that (1) build user resilience while
remaining explainable, and (2) protect user privacy while still being able to
learn from feedback?

## The Four Models

| Model | Approach | Explainability | Adaptability |
|---|---|---|---|
| **Rule-Based** | Exact pattern matching against a fixed intent set | Fully transparent — every response traces to one matched pattern | None — cannot handle paraphrased input |
| **LSTM** | Trained intent classifier + VADER sentiment, LIME for word-level explanation | Partial — LIME shows influential words, prediction itself is a black box | Moderate — generalizes somewhat beyond exact patterns |
| **GPT-2 (fine-tuned)** | Fine-tuned transformer for open-ended empathetic responses | Low — no reliable mechanism to trace why a response was generated | High, but inconsistent — sometimes mirrors user input rather than responding meaningfully |
| **GPT-3.5 + RASA** | RASA handles structured intents/dialogue flow; GPT-3.5 is a fallback for low-confidence input, with an RLHF-style feedback loop | Limited but improved — RASA intent logs give partial traceability | Highest — dynamically fine-tunes on positive feedback over time |

## Dataset

`MHResilience.json` — 120 intents, ~6,400 user input patterns, ~6,000 curated
resilience-focused responses, covering everything from neutral small talk
(greetings, time of day) to emotionally sensitive topics (stress, anxiety,
worthlessness, suicidal ideation). Same dataset used across all four models.


 

## Evaluation & Results

The Mental Health Care Chatbots were assessed using both
technical performance metrics and user-centric evaluations
to understand their effectiveness in delivering emotionally
supportive conversations.
Four chatbot models were developed—rule-based, LSTM
based, fine-tuned GPT-2, and GPT-3.5 Turbo-RASA Framework
and were evaluated on intent accuracy, perplexity, F1-score,
response relevance, and user satisfaction. The user interface for
the rule-based, LSTM-based, and GPT-2 models was created
using Streamlit. The rule-based model achieved 89.49 per cent
accuracy but lacked adaptability. The LSTM model reached 87
per cent intent classification accuracy and demonstrated strong
sensitivity to short emotional cues. GPT-2 achieved a perplexity of
4.62 but often mirrored user inputs, resulting in low satisfaction
(=2.2/5). GPT-3.5 Turbo–RASA outperformed all models with an
F1-score of 90 per cent, accurate emotion capture, and the highest
user satisfaction (4.88/5).

