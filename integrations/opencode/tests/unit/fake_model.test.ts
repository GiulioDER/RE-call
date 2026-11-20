import { afterEach, expect, test } from "bun:test";
import { startFakeModel, type FakeModel } from "../../harness/fake_model";

let model: FakeModel | undefined;
afterEach(async () => {
  await model?.stop();
  model = undefined;
});

async function post(body: object) {
  return fetch(`${model!.baseURL}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

// Parse an SSE body the way an OpenAI client does: `data:` frames, `[DONE]` last.
async function sse(res: Response) {
  const frames = (await res.text())
    .split("\n\n")
    .filter((f) => f.startsWith("data: "))
    .map((f) => f.slice("data: ".length));
  expect(frames.at(-1)).toBe("[DONE]");
  const events = frames.slice(0, -1).map((f) => JSON.parse(f));
  let text = "";
  const calls: { id: string; name: string; arguments: string }[] = [];
  let finish: string | null = null;
  for (const e of events) {
    const choice = e.choices[0];
    if (choice === undefined) continue;
    if (choice.delta.content) text += choice.delta.content;
    for (const d of choice.delta.tool_calls ?? []) {
      const call = (calls[d.index] ??= { id: "", name: "", arguments: "" });
      if (d.id) call.id = d.id;
      if (d.function?.name) call.name = d.function.name;
      if (d.function?.arguments) call.arguments += d.function.arguments;
    }
    if (choice.finish_reason) finish = choice.finish_reason;
  }
  return { text, calls, finish, events };
}

/**
 * Invariant: a streaming request gets the scripted tool call (name, and arguments that parse back to
 * the scripted object after the client joins the deltas), finish reason `tool_calls`, and the body
 * is kept exactly as sent.
 * Red proof, row FAKE-STREAM-ARGS: `t.arguments.slice(cut)` replaced by `""` in harness/fake_model.ts
 * (the second argument delta dropped); fails at `expect(calls[0].arguments).toBe(...)`.
 * Row FAKE-RECORD: `requests.push(body)` replaced by `requests.push({ ...body, messages: [] })`;
 * fails at `expect(model.requests).toEqual([body])`.
 */
test("the fake model returns the scripted tool call and records the request", async () => {
  model = await startFakeModel([{ toolCalls: [{ name: "echo", arguments: { text: "hi", n: 2 } }] }]);
  const body = { model: "scripted", stream: true, messages: [{ role: "user", content: "say hi" }], tools: [{ type: "function", function: { name: "echo" } }] };
  const res = await post(body);
  expect(res.headers.get("content-type")).toBe("text/event-stream");
  const { calls, finish } = await sse(res);
  expect(calls.length).toBe(1);
  expect(calls[0]!.name).toBe("echo");
  // Compared as text: a broken join would make JSON.parse throw outside `expect`.
  expect(calls[0]!.arguments).toBe(JSON.stringify({ text: "hi", n: 2 }));
  expect(finish).toBe("tool_calls");
  expect(model.requests).toEqual([body]);
});

/**
 * Invariant: a non-streaming request gets one `chat.completion` object carrying the scripted text and
 * finish reason `stop`.
 * Red proof, row FAKE-NONSTREAM: `body.stream === true ? streaming(completion) : nonStreaming(completion)`
 * replaced by `streaming(completion)`; fails at `expect(json.object).toBe("chat.completion")`
 * (the body is then SSE text, so `res.json()` is replaced by a text parse in the test to keep the
 * failure inside `expect`).
 */
test("a non-streaming request gets the scripted text as one completion", async () => {
  model = await startFakeModel([{ text: "hello there" }]);
  const res = await post({ model: "scripted", messages: [{ role: "user", content: "hi" }] });
  const raw = await res.text();
  const json = raw.startsWith("{") ? JSON.parse(raw) : { object: raw.slice(0, 40) };
  expect(json.object).toBe("chat.completion");
  expect(json.choices[0].message).toEqual({ role: "assistant", content: "hello there" });
  expect(json.choices[0].finish_reason).toBe("stop");
});

/**
 * Invariant: scripted turns are consumed one per request, in order.
 * Red proof, row FAKE-ORDER: `next += 1;` replaced by `next += 0;`; fails at
 * `expect(second.text).toBe("two")`.
 */
test("each request consumes the next scripted turn", async () => {
  model = await startFakeModel([{ text: "one" }, { text: "two" }]);
  const first = await sse(await post({ stream: true, messages: [] }));
  const second = await sse(await post({ stream: true, messages: [] }));
  expect(first.text).toBe("one");
  expect(second.text).toBe("two");
});

/**
 * Invariant: a request past the end of the script is an error the client sees (HTTP 500 with an
 * OpenAI error body, marked `x-should-retry: false` so OpenCode fails the turn instead of retrying
 * with backoff), never an invented answer, and the request is still recorded.
 * Red proof, row FAKE-NO-RETRY: `{ status: 500, headers: { "x-should-retry": "false" } }` replaced
 * by `{ status: 500 }`; fails at `expect(res.headers.get("x-should-retry")).toBe("false")`.
 * Red proof, row FAKE-EXHAUSTED: `const turn = script[next];` replaced by
 * `const turn = script[next] ?? { text: "" };` (an invented empty answer); fails at
 * `expect(res.status).toBe(500)`.
 */
test("a request past the end of the script gets an error and is still recorded", async () => {
  model = await startFakeModel([]);
  const res = await post({ stream: true, messages: [{ role: "user", content: "x" }] });
  expect(res.status).toBe(500);
  expect(res.headers.get("x-should-retry")).toBe("false");
  const raw = await res.text();
  const json = raw.startsWith("{") ? JSON.parse(raw) : { error: { code: raw.slice(0, 40) } };
  expect(json.error.code).toBe("script_exhausted");
  expect(model.requests.length).toBe(1);
});
