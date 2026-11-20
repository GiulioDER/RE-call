// A scripted model that speaks enough of OpenAI's chat completions API for OpenCode 2.2.7's
// openai-compatible provider: `POST /v1/chat/completions`, streaming (SSE) and non-streaming.
// Each request consumes the next scripted turn. Every request body is kept, parsed, exactly as
// received, so a test or the driver can read what the model was given.

export type ScriptedTurn = {
  toolCalls?: { name: string; arguments: object }[];
  text?: string;
};

export type FakeModel = {
  baseURL: string;
  requests: object[];
  stop(): Promise<void>;
};

type Completion = {
  id: string;
  text: string;
  toolCalls: { id: string; name: string; arguments: string }[];
  finish: "stop" | "tool_calls";
};

const MODEL = "scripted";

function completionFor(turn: ScriptedTurn, n: number): Completion {
  const toolCalls = (turn.toolCalls ?? []).map((call, i) => ({
    id: `call_${n}_${i}`,
    name: call.name,
    arguments: JSON.stringify(call.arguments),
  }));
  return {
    id: `chatcmpl-${n}`,
    text: turn.text ?? "",
    toolCalls,
    finish: toolCalls.length > 0 ? "tool_calls" : "stop",
  };
}

const USAGE = { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 };

function nonStreaming(c: Completion): Response {
  const message: Record<string, unknown> = { role: "assistant", content: c.text === "" ? null : c.text };
  if (c.toolCalls.length > 0) {
    message.tool_calls = c.toolCalls.map((t) => ({
      id: t.id,
      type: "function",
      function: { name: t.name, arguments: t.arguments },
    }));
  }
  return Response.json({
    id: c.id,
    object: "chat.completion",
    created: 0,
    model: MODEL,
    choices: [{ index: 0, message, finish_reason: c.finish }],
    usage: USAGE,
  });
}

function streaming(c: Completion): Response {
  const chunk = (delta: object, finish: string | null = null, extra: object = {}) =>
    `data: ${JSON.stringify({
      id: c.id,
      object: "chat.completion.chunk",
      created: 0,
      model: MODEL,
      choices: [{ index: 0, delta, finish_reason: finish }],
      ...extra,
    })}\n\n`;
  const frames: string[] = [chunk({ role: "assistant", content: "" })];
  if (c.text !== "") frames.push(chunk({ content: c.text }));
  c.toolCalls.forEach((t, index) => {
    // The arguments arrive in two deltas, as real servers split them, so the client must accumulate.
    const cut = Math.floor(t.arguments.length / 2);
    frames.push(
      chunk({
        tool_calls: [{ index, id: t.id, type: "function", function: { name: t.name, arguments: t.arguments.slice(0, cut) } }],
      }),
    );
    frames.push(chunk({ tool_calls: [{ index, function: { arguments: t.arguments.slice(cut) } }] }));
  });
  frames.push(chunk({}, c.finish));
  frames.push(`data: ${JSON.stringify({ id: c.id, object: "chat.completion.chunk", created: 0, model: MODEL, choices: [], usage: USAGE })}\n\n`);
  frames.push("data: [DONE]\n\n");
  return new Response(frames.join(""), {
    headers: { "content-type": "text/event-stream", "cache-control": "no-cache" },
  });
}

export async function startFakeModel(script: ScriptedTurn[]): Promise<FakeModel> {
  const requests: object[] = [];
  let next = 0;
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(req) {
      const path = new URL(req.url).pathname;
      if (req.method === "GET" && path === "/v1/models") {
        return Response.json({ object: "list", data: [{ id: MODEL, object: "model", owned_by: "fake" }] });
      }
      if (req.method !== "POST" || path !== "/v1/chat/completions") {
        return new Response(`fake model: no route ${req.method} ${path}`, { status: 404 });
      }
      const body = (await req.json()) as Record<string, unknown>;
      requests.push(body);
      const turn = script[next];
      if (turn === undefined) {
        // An exhausted script is a test bug; say so loudly rather than invent an answer. OpenCode
        // retries a 500 with backoff unless the response says not to.
        return Response.json(
          { error: { message: `fake model: script exhausted at request ${requests.length}`, code: "script_exhausted" } },
          { status: 500, headers: { "x-should-retry": "false" } },
        );
      }
      const completion = completionFor(turn, next);
      next += 1;
      return body.stream === true ? streaming(completion) : nonStreaming(completion);
    },
  });
  return {
    baseURL: `http://127.0.0.1:${server.port}/v1`,
    requests,
    stop: () => server.stop(true),
  };
}
