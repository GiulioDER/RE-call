"""Time the two Docker daemon probes, and check they agree on every state that matters.

Record: `docs/preregistrations/2026-08-25-docker-probe-latency.md`.

`recall.quickstart.docker_unavailable_reason` runs before anything is written, on the first command
a new user types. Whatever it costs is a fixed tax on `recall quickstart` and `recall doctor`, paid
by someone who has not yet seen the product work.

Arms are ALTERNATED rather than run in blocks, because Docker Desktop warms up: five-then-five
would attribute the warm-up entirely to whichever arm went first. The median is reported rather
than the mean for the same reason.

⚠️ **Two apparatus defects were removed before this was committed, and both would have passed.**

The first draft tested "docker is not installed" by handing `subprocess.run` an environment whose
`PATH` had the docker directory stripped out. **On Windows that does not do what it reads as:**
`CreateProcess` resolves a bare executable name against the CALLING process's `PATH`, not against
the `env` mapping passed to the child, so both arms would have found docker anyway, both would have
returned the same thing, and the case would have been reported as agreement having never been
exercised. It is now checked where the real function checks it, which is `shutil.which`, and
`shutil.which` genuinely honours an explicit search path.

The second: the exception path returned `1`, the same code a refusing daemon returns, so a 120
second timeout would have been indistinguishable from "the daemon answered no". It returns `-1`.
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import time

RUNS = 5

#: The probe in `recall/quickstart.py` today.
INFO = ["docker", "info", "--format", "{{.ServerVersion}}"]
#: The candidate. One round trip to the daemon's /version endpoint rather than a full inventory.
VERSION = ["docker", "version", "--format", "{{.Server.Version}}"]

#: Returned when the probe never produced an exit status at all (timeout, or no such executable).
#: Deliberately not `1`: that is what a reachable daemon returns when it refuses, and a benchmark
#: that cannot tell "no answer" from "answered no" is measuring two things under one name.
NO_ANSWER = -1


def _run(argv: list[str], env: dict[str, str] | None = None) -> tuple[int, float, str]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return NO_ANSWER, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"
    output = (completed.stdout or completed.stderr or "").strip().splitlines()
    return completed.returncode, time.perf_counter() - started, output[0] if output else ""


def timings() -> None:
    samples: dict[str, list[float]] = {"info": [], "version": []}
    for index in range(RUNS):
        for name, argv in (("info", INFO), ("version", VERSION)):
            code, seconds, first = _run(argv)
            samples[name].append(seconds)
            print(f"  run {index + 1} {name:8} {seconds:7.2f}s  rc={code}  {first}")
    print()
    medians = {name: statistics.median(values) for name, values in samples.items()}
    for name, value in medians.items():
        print(
            f"{name:8} median {value:7.2f}s  (n={RUNS}, all: "
            f"{', '.join(f'{v:.2f}' for v in samples[name])})"
        )
    if medians["version"]:
        print(f"\nspeedup: {medians['info'] / medians['version']:.1f}x")


def agreement() -> None:
    """The half that decides whether the faster probe is USABLE, not merely faster.

    A probe that reports a dead daemon as healthy is worse than a slow correct one: the quickstart
    writes a compose file and starts a container on the strength of this answer.

    Only two of the three states can discriminate between the arms. "docker is not installed" is
    decided by `shutil.which` in `docker_unavailable_reason`, ABOVE the branch either probe would
    reach, so the two arms are identical there by construction. That is stated and checked rather
    than dressed up as a third comparison, because a comparison whose answer is forced tells you
    nothing and reads as if it did.
    """
    print("\nstate agreement (rc=0 means 'docker is usable'):")
    dead = dict(os.environ)
    dead["DOCKER_HOST"] = "tcp://127.0.0.1:1"

    for state, env in (("healthy", None), ("daemon dead", dead)):
        row = []
        for name, argv in (("info", INFO), ("version", VERSION)):
            code, seconds, first = _run(argv, env=env)
            row.append(f"{name}: rc={code} ({seconds:.2f}s) {first[:60]}")
        print(f"  {state:14} " + " | ".join(row))

    docker = shutil.which("docker")
    stripped = os.pathsep.join(
        part
        for part in os.environ.get("PATH", "").split(os.pathsep)
        if docker and part and part.lower() != os.path.dirname(docker).lower()
    )
    absent = shutil.which("docker", path=stripped)
    print(
        f"  {'docker absent':14} shutil.which -> {absent!r} "
        f"(shared by both probes; not an arm comparison)"
    )


if __name__ == "__main__":
    timings()
    agreement()
