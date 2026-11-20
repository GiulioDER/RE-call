/** No test may reach a paid model or a credentialed service, because an in-process OpenCode host
 * configures a provider from any key it finds in the environment. Measured 2026-11-11 by the bridge
 * session: a CLI run with OPENROUTER_API_KEY inherited reached a paid OpenRouter model ($0.000949
 * reported) for one "hello". The owner's rule is that a test never calls a real API (2026-10-31).
 * `tests/preload.ts` applies this before every `bun test` file; this module only defines it, so a test
 * can import it without stripping the environment as a side effect. */

/** A name that carries a credential or points a client at a hosted provider. */
export const CREDENTIAL = /(API_KEY|_TOKEN|_SECRET|_PASSWORD)$|^(OPENROUTER|ANTHROPIC|OPENAI|VOYAGE|DEEPINFRA|GEMINI|GOOGLE_API|GROQ|MISTRAL|COHERE|TOGETHER|FIREWORKS|XAI|AZURE_OPENAI|AWS_)/i;

/** The names `stripCredentials` removes from `env`, which it edits in place. */
export function stripCredentials(env: Record<string, string | undefined>): string[] {
  const removed = Object.keys(env).filter((k) => CREDENTIAL.test(k));
  for (const k of removed) delete env[k];
  return removed;
}
