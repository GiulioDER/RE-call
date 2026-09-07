#!/usr/bin/env python3
"""
Runs a command with a 3-second wall-clock timeout.

On timeout, prints "TIMEOUT" and exits 124. Kills the entire process tree,
not just the direct child. Otherwise passes through the command's exit code.

Used to guard pre-commit hooks, so returning promptly is essential.
"""

import os
import signal
import subprocess
import sys
import threading


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} COMMAND [ARGS...]", file=sys.stderr)
        sys.exit(2)

    command = sys.argv[1:]
    timeout_seconds = 3
    timed_out = threading.Event()
    proc = None

    def kill_tree():
        """Kill the entire process tree when timeout fires."""
        timed_out.set()
        if proc is not None and proc.poll() is None:
            try:
                # On Unix, kill the process group; on Windows, kill the tree
                if hasattr(signal, "SIGTERM"):
                    # Unix: kill the process group
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    # Windows: use taskkill
                    subprocess.run(
                        ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                        capture_output=True,
                    )
            except (ProcessLookupError, OSError):
                pass  # Process already died

    # Start the timer thread
    timer = threading.Timer(timeout_seconds, kill_tree)
    timer.daemon = True
    timer.start()

    try:
        # On Unix, create a new process group so we can kill the entire tree
        # On Windows, this is ignored
        preexec_fn = os.setsid if hasattr(os, "setsid") else None

        proc = subprocess.Popen(
            command,
            preexec_fn=preexec_fn,
            stdin=subprocess.DEVNULL,  # Prevent hanging on prompts
        )
        returncode = proc.wait()

        # Cancel the timer if we finished before timeout
        timer.cancel()

        if timed_out.is_set():
            print("TIMEOUT")
            sys.exit(124)
        else:
            sys.exit(returncode)

    except Exception as e:
        timer.cancel()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
