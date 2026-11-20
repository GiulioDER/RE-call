"""A scripted stand-in for `recall-mcp`, run as a real child process over a real pipe.

FAKE_SCENARIO names a JSON file with these keys, all optional but `tools`:

- "tools": {tool: [answer, ...]}. Each call takes the next answer of its tool (the last one
  repeats). An answer {"isError": true, "text": ...} is sent as a tool error; an answer
  {"raw_result": ...} is sent as the JSON-RPC result exactly as given (to script malformed answers).
- "delay": {tool: seconds} before every answer; "delay_first": {tool: seconds} before the first only;
  "delay_text": {text: seconds} before every answer to a call with that `text` argument;
  "no_answer_text": [text, ...], calls with that `text` argument are never answered.
- "exit_on": tool, exit without answering when that tool is called.
- "init_delay": seconds before answering `initialize`.
- "stall_stdin": true, never read input; "stall_after_init": true, stop reading after the
  handshake (a wedged server).
- "tools_list": [tool, ...], the answer to `tools/list` (`{"tools": [...]}`); without it `tools/list`
  is never answered, as in the Hermes copy. Added in recall-opencode 2026-11-10 (the Hermes copy at
  86fdc528 answers only `initialize` and `tools/call`); vendor/PINS.md's hash is for that copy.
- "init_error": {"code": ..., "message": ...}, answer `initialize` with that JSON-RPC error.
  Added in recall-opencode 2026-11-10, as is the `cancelled` event below.
- "split_utf8": true, write every answer as UTF-8 (no `ensure_ascii`) in two flushes 0.2 s apart,
  cut inside its first multibyte character (or at its middle when it has none), so the reader's
  framing and decoding meet a split line and a split character. Added in recall-opencode 2026-11-11.

FAKE_LOG receives one JSON line per event: the start (with pid, cwd and environment), every
tools/call with its arguments, every `notifications/cancelled` with its `requestId`, and the exit
when input closes.
"""

import json
import os
import sys
import time

scenario = json.load(open(os.environ["FAKE_SCENARIO"], encoding="utf-8"))
log = open(os.environ["FAKE_LOG"], "a", encoding="utf-8")


def record(event):
    log.write(json.dumps(event) + "\n")
    log.flush()


def send(message):
    if scenario.get("split_utf8") and "id" in message:
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        cut = next((i + 1 for i, b in enumerate(data) if b >= 0xC0), len(data) // 2)
        for part in (data[:cut], data[cut:]):
            sys.stdout.buffer.write(part)
            sys.stdout.buffer.flush()
            time.sleep(0.2)
        return
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


record({"event": "start", "pid": os.getpid(), "cwd": os.getcwd(), "env": dict(os.environ)})
if scenario.get("stall_stdin"):
    time.sleep(120)
    sys.exit(0)
calls = {}
replies = {}
arrived = set()
forgotten = set()
for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize" and "init_error" in scenario:
        send({"jsonrpc": "2.0", "id": message["id"], "error": scenario["init_error"]})
    elif method == "notifications/cancelled":
        record({"event": "cancelled", "requestId": message["params"].get("requestId")})
    elif method == "initialize":
        time.sleep(scenario.get("init_delay", 0))
        send({"jsonrpc": "2.0", "id": message["id"], "result": {"protocolVersion": "2025-06-18",
              "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "0"}}})
        if scenario.get("stall_after_init"):
            time.sleep(120)
            sys.exit(0)
    elif method == "tools/list" and "tools_list" in scenario:
        record({"event": "list"})
        send({"jsonrpc": "2.0", "id": message["id"], "result": {"tools": scenario["tools_list"]}})
    elif method == "tools/call":
        name = message["params"]["name"]
        record({"event": "call", "tool": name, "arguments": message["params"]["arguments"]})
        if scenario.get("exit_on") == name:
            sys.exit(3)
        if str(message["params"]["arguments"].get("text")) in scenario.get("no_answer_text", []):
            continue  # the answer is lost: nothing comes back for this request, the server goes on
        # A repeated request_id behaves as RE-call's memo request journal (recall/lite/memo_requests.py):
        # the same arguments get the first answer, other arguments `request_mismatch`, and a request
        # whose memo was forgotten since `memo_missing`.
        arguments = message["params"]["arguments"]
        request_id = arguments.get("request_id")
        answers = scenario["tools"].get(name, [{"isError": True, "text": f"no such tool {name}"}])
        if name == "recall_forget_memo":
            forgotten.add(arguments.get("memo_id"))
        if request_id and (name, request_id) in replies:
            index, first_arguments = replies[(name, request_id)]
            answer = answers[min(index, len(answers) - 1)]
            if first_arguments != arguments:
                answer = {"refused": "request_id was already used", "conflict": "request_mismatch"}
            elif isinstance(answer, dict) and answer.get("memo_id") in forgotten:
                answer = {"refused": "the request's memo was explicitly forgotten", "conflict": "memo_missing"}
        else:
            index = calls.get(name, 0)
            calls[name] = index + 1
            if request_id:
                replies[(name, request_id)] = (index, arguments)
            answer = answers[min(index, len(answers) - 1)]
        time.sleep(scenario.get("delay", {}).get(name, 0))
        time.sleep(scenario.get("delay_text", {}).get(str(arguments.get("text")), 0))
        if name not in arrived:  # the first call to arrive, whatever its answer
            time.sleep(scenario.get("delay_first", {}).get(name, 0))
        arrived.add(name)
        if isinstance(answer, dict) and "raw_result" in answer:
            result = answer["raw_result"]
        elif isinstance(answer, dict) and answer.get("isError"):
            result = {"content": [{"type": "text", "text": answer["text"]}], "isError": True}
        else:
            result = {"content": [{"type": "text", "text": json.dumps(answer)}], "isError": False}
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info"}})
        send({"jsonrpc": "2.0", "id": message["id"], "result": result})
record({"event": "exit"})
