# ATM-Bench DeepSeek reasoning smoke

This is a provider connectivity smoke test, not an ATM quality measurement.
The reasoning arm is configured for the OpenRouter model
`deepseek/deepseek-v4-pro` with a requested `reasoning.effort=medium`.

The API request must use the OpenAI-compatible OpenRouter endpoint and must
not print the request, response content, reasoning content, or credentials.
The smoke prediction is HTTP 200, one completed choice, and non-empty
reasoning content.

Execution on 2026-08-19:

* HTTP status: 200
* returned model: `deepseek-v4-pro`
* choices: 1
* finish reason: `stop`
* answer characters: 11
* reasoning characters: 123

DeepSeek's current thinking-mode documentation says that `medium` is accepted
as a compatibility value and maps to `high`. The current RE-call ATM runner
does not wire an answer provider, so this smoke proves provider reachability,
not end-to-end reasoning quality or an official ATM answer score.
