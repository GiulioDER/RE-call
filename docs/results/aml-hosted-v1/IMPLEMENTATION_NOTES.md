# RE-call Hosted 1.0 implementation notes

## 2026-09-11 provider clarification

The generative provider is OpenRouter. Every Add compiler request and Search facet request uses
OpenAI `gpt-4o-mini` through the fixed OpenRouter model identifier `openai/gpt-4o-mini` and the
OpenAI-compatible endpoint `https://openrouter.ai/api/v1`. The runtime credential is
`OPENROUTER_API_KEY`; `OPENAI_API_KEY` is not read by `recall_aml`.

`RECALL_AML_DATABASE_URL` and `RECALL_AML_API_KEY` are admission-time inputs that will be available
when the next run opens. Their absence before that event is expected. They remain mandatory at
service startup because the hosted service must not run without persistent storage or evaluation
authentication.

This clarification does not change the committed preregistration. The underlying generative model
remains `gpt-4o-mini`; only its transport and provider-qualified identifier are made explicit here.
