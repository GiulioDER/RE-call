#!/usr/bin/env bash
# Tests for `scripts/session-mcp-close.sh`, which kills processes.
#
# The whole file is about ONE question: which pids does it select? Everything else it does is a
# print. So the process table is a fixture and the killer is a log, which lets the parent-chain
# walk be exercised on any machine, with no Claude running, no ssh, and no VPS2.
#
# The fixture is drawn from the real machine on 2026-08-26, because the hazard is real there: the
# same `recall_mcp.server` command line belonged to transports under `claude.exe` AND under
# `codex.exe`, so a command-line sweep would have killed another agent's live servers. Test 2 is
# what pins that, and it is the reason ownership is decided by ancestry.
#
# Mutation-tested 2026-08-26, five ways. Measured reds, not predicted ones:
#
#   `_descends_from` always true                   1, 2, 4, 6, 9, 10, 11 red
#   `_descends_from` always false                  1, 2, 4, 6, 9, 10, 11 red
#   the SELF exclusion removed                     1, 3 red
#   the hop cap raised to `while true`             8 hangs; the suite times out after 7 results
#   the missing-CLAUDE_PID refusal removed         5 red
#
# ⚠️ The third of those SURVIVED the first version of this file, and the repair is worth reading
# before editing the fixture. Pid 700 (the script's own ssh) was parented to `__SELF__`, but
# `__SELF__` itself was absent from the table, so the ownership walk failed at the first hop and
# 700 was excluded for the WRONG reason. Every test stayed green with the guard deleted. The
# fixture now places the script under the session client, which is where it really sits.
set -uo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session-mcp-close.sh"
BASE="${TMPDIR:-/tmp}/recall-mcpclose"
pass=0; fail=0
ok() { pass=$((pass+1)); printf 'PASS  %s\n' "$1"; }
no() { fail=$((fail+1)); printf 'FAIL  %s\n     %s\n' "$1" "${2:-}"; }

rm -rf "$BASE"; mkdir -p "$BASE/bin"

# --- the kill log -------------------------------------------------------------------------------
cat > "$BASE/bin/killstub" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$1" >> "$KILL_LOG"
exit "${FAKE_KILL_RC:-0}"
STUB

# --- the process table, printed the way both platform branches print it -------------------------
# The script replaces nothing; this stub does, because the fixture has to name the script's own
# pid to prove the self-exclusion and cannot know it in advance.
cat > "$BASE/bin/pstable" <<'STUB'
#!/usr/bin/env bash
sed "s/__SELF__/${SELF_PID:-0}/" "$TABLE_FILE"
STUB

# --- the ssh stub, standing in for the fleet count on VPS2 --------------------------------------
cat > "$BASE/bin/ssh" <<'STUB'
#!/usr/bin/env bash
printf 'ssh %s\n' "$*" >> "$SSH_LOG"
# `n gb hours`, the shape the real awk emits. The count drops on the second call so the
# before/after report has something to show and a test can prove both calls happened.
if [ -f "$SSH_LOG.count" ]; then
    printf '87 20.9 69.1'
else
    : > "$SSH_LOG.count"
    printf '88 21.2 69.1'
fi
STUB
chmod +x "$BASE/bin/killstub" "$BASE/bin/pstable" "$BASE/bin/ssh"

MCP='ssh -o BatchMode=yes vps2 cd ~/recall-repos/serving && export RECALL_TENANT=%s RECALL_EMBEDDER=voyage:voyage-4 && exec python -m recall_mcp.server'
{
    printf '900 1 claude.exe --session\n'
    printf '901 900 %s\n' "$(printf "$MCP" memory)"
    printf '902 900 %s\n' "$(printf "$MCP" re-call-code-gen)"
    printf '910 900 node.exe some-wrapper\n'
    printf '911 910 %s\n' "$(printf "$MCP" re-call-docs)"
    printf '920 900 ssh -o BatchMode=yes vps2 tail -f /var/log/something\n'
    printf '800 1 codex.exe app-server\n'
    printf '801 800 %s\n' "$(printf "$MCP" memory)"
    # This script, and the ssh IT runs to count servers. The parent line is load-bearing: the
    # script really does descend from the session client, so without it pid 700 fails the
    # ownership test for the wrong reason and the self-exclusion guard can be deleted with every
    # test still green. Mutation testing caught exactly that, and this line is the repair.
    printf '__SELF__ 900 bash scripts/session-mcp-close.sh close\n'
    printf '700 __SELF__ ssh -o BatchMode=yes vps2 ps -eo rss,etimes,args grep python -m recall_mcp.server\n'
} > "$BASE/table.txt"

# A table whose parent chain is a cycle: 601 -> 602 -> 601. A recycled pid produces this, and a
# walk without a hop cap never returns from it.
{
    printf '900 1 claude.exe --session\n'
    printf '601 602 %s\n' "$(printf "$MCP" memory)"
    printf '602 601 node.exe cycle\n'
} > "$BASE/cycle.txt"

RC=0
OUT=""
run() {
    : > "$BASE/kill.log"; : > "$BASE/ssh.log"; rm -f "$BASE/ssh.log.count"
    OUT="$(env PATH="$BASE/bin:$PATH" \
        CLAUDE_PID="${WANT_SESSION_PID-900}" \
        KILL_LOG="$BASE/kill.log" \
        SSH_LOG="$BASE/ssh.log" \
        TABLE_FILE="${TABLE_FILE:-$BASE/table.txt}" \
        RECALL_MCP_PS="$BASE/bin/pstable" \
        RECALL_MCP_KILL="$BASE/bin/killstub" \
        RECALL_MCP_CLOSE_SETTLE=0 \
        bash "$SCRIPT" "$@" 2>&1)"
    RC=$?
}
killed() { printf ' %s' "$(tr '\n' ' ' < "$BASE/kill.log")"; }
sshlog() { cat "$BASE/ssh.log" 2>/dev/null; }

# --- 1. report names ours and kills nothing ------------------------------------------------------
run report
if [ "$RC" -eq 0 ] && printf '%s' "$OUT" | grep -q 'TRANSPORTS  3 this session' \
   && [ -z "$(killed | tr -d ' ')" ]; then
    ok "1  report finds this session's three transports and kills nothing"
else
    no "1  report finds this session's three transports and kills nothing" "rc=$RC $OUT killed=[$(killed)]"
fi

# --- 2. close kills ours and NOT the other agent's ----------------------------------------------
# The one that matters. 801 is a live transport under codex.exe with an identical command line,
# and killing it would take down another agent's server mid-query.
run close
if [ "$RC" -eq 0 ] && killed | grep -q ' 901 ' && killed | grep -q ' 902 ' \
   && ! killed | grep -q ' 801 '; then
    ok "2  close kills this session's transports and leaves another agent's alone"
else
    no "2  close kills this session's transports and leaves another agent's alone" "rc=$RC killed=[$(killed)]"
fi

# --- 3. it never kills the ssh it is itself running ---------------------------------------------
# Pid 700 matches the pattern AND descends from this script, which descends from the session.
# Ownership alone would select it, and the script would kill its own fleet query.
if ! killed | grep -q ' 700 '; then
    ok "3  the script's own ssh is excluded even though it matches and is ours"
else
    no "3  the script's own ssh is excluded even though it matches and is ours" "killed=[$(killed)]"
fi

# --- 4. ancestry is walked, not just the immediate parent ---------------------------------------
# 911's parent is a wrapper whose parent is the client. One hop would miss it.
if killed | grep -q ' 911 '; then
    ok "4  a grandchild transport is still this session's"
else
    no "4  a grandchild transport is still this session's" "killed=[$(killed)]"
fi

# --- 5. no session pid, no kill -----------------------------------------------------------------
WANT_SESSION_PID="" run close
if [ "$RC" -eq 3 ] && [ -z "$(killed | tr -d ' ')" ] && printf '%s' "$OUT" | grep -q 'CLAUDE_PID'; then
    ok "5  without CLAUDE_PID it refuses rather than guessing"
else
    no "5  without CLAUDE_PID it refuses rather than guessing" "rc=$RC $OUT killed=[$(killed)]"
fi

# --- 6. --dry-run names them and kills none ------------------------------------------------------
run close --dry-run
if [ "$RC" -eq 0 ] && [ -z "$(killed | tr -d ' ')" ] && printf '%s' "$OUT" | grep -q 'DRY-RUN'; then
    ok "6  --dry-run names the transports and kills none"
else
    no "6  --dry-run names the transports and kills none" "rc=$RC $OUT killed=[$(killed)]"
fi

# --- 7. an unrelated ssh under the same session is not a transport -------------------------------
run close
if ! killed | grep -q ' 920 '; then
    ok "7  an ssh that is not an MCP transport is left alone"
else
    no "7  an ssh that is not an MCP transport is left alone" "killed=[$(killed)]"
fi

# --- 8. a cycle in the process table terminates --------------------------------------------------
TABLE_FILE="$BASE/cycle.txt" run close
if [ "$RC" -eq 0 ] && [ -z "$(killed | tr -d ' ')" ]; then
    ok "8  a parent-chain cycle terminates and selects nothing"
else
    no "8  a parent-chain cycle terminates and selects nothing" "rc=$RC killed=[$(killed)]"
fi

# --- 9. the fleet is counted before AND after a close --------------------------------------------
# A kill returning 0 says a signal was sent, not that 850 MB was freed on another machine.
run close
if printf '%s' "$OUT" | grep -q 'FLEET before' && printf '%s' "$OUT" | grep -q 'FLEET after' \
   && [ "$(sshlog | grep -c '^ssh ')" -eq 2 ]; then
    ok "9  the host is asked for a server count before and after"
else
    no "9  the host is asked for a server count before and after" "$OUT ssh=[$(sshlog)]"
fi

# --- 10. --no-fleet makes no ssh call at all -----------------------------------------------------
run close --no-fleet
if [ "$RC" -eq 0 ] && [ -z "$(sshlog)" ] && killed | grep -q ' 901 '; then
    ok "10 --no-fleet closes without touching the network"
else
    no "10 --no-fleet closes without touching the network" "rc=$RC ssh=[$(sshlog)]"
fi

# --- 11. a kill that fails is reported, not swallowed --------------------------------------------
export FAKE_KILL_RC=1
run close
unset FAKE_KILL_RC
if [ "$RC" -eq 1 ] && printf '%s' "$OUT" | grep -q 'FAILED'; then
    ok "11 a failed kill is reported and exits non-zero"
else
    no "11 a failed kill is reported and exits non-zero" "rc=$RC $OUT"
fi

printf '\n%s passed, %s failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
