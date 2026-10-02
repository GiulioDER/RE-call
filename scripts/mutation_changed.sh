#!/usr/bin/env bash
# Mutation-test the shipped Python a change touched, against the tests that import it.
#
#   bash scripts/mutation_changed.sh [BASE]        # BASE defaults to origin/master
#   MUTATE_FILES="recall/x.py recall_mcp/y.py" bash scripts/mutation_changed.sh
#
# Why: this repository's rule is that a test is evidence only once it has been seen to fail for the
# defect it guards. Mutation testing asks that question mechanically for every line a change
# touches: mutmut rewrites the code one small defect at a time and reports each mutant no test
# noticed ("survived"). A survivor is either a missing assertion or dead behaviour.
#
# Scope, chosen so a run stays in minutes rather than hours:
#   * only files changed against BASE under the shipped packages (mutmut `only_mutate`), capped at
#     MAX_FILES, the rest listed as not run;
#   * only test files that import a changed module (`pytest_add_cli_args_test_selection`), at most
#     MAX_TESTS_PER_FILE (6) per changed module, the most coupled ones, the rest listed. A changed
#     module that no test imports is reported as such, which is itself the finding;
#   * mutants that fail mypy are discarded before any test runs (`type_check_command`);
#   * the mutmut run gets a time budget of its own, MUTATION_BUDGET_SECONDS (default 720), inside
#     the CI job's timeout. When it runs out the run is stopped, the summary reports the mutants it
#     finished and marks the rest "not checked", and the script still exits 0.
#
# Database tests skip when RECALL_TEST_DSN is unset, as in CI, so a mutant reachable only through
# them is reported "no tests" rather than "survived". mutmut forks per mutant, which bypasses the
# per-xdist-worker databases in tests/conftest.py, so pointing this at a shared database is unsafe.
#
# POSIX only (mutmut needs os.fork): CI, WSL or a Linux host, never native Windows. Needs mutmut
# and mypy on PATH. Writes setup.cfg and mutants/ in the working tree and removes setup.cfg after.
# Report only: it exits 0 whatever survives, and non-zero only if mutmut itself could not run.
set -euo pipefail

BASE="${1:-origin/master}"
MAX_FILES="${MAX_FILES:-20}"
MAX_TESTS_PER_FILE="${MAX_TESTS_PER_FILE:-6}"
BUDGET="${MUTATION_BUDGET_SECONDS:-720}"
PYTHON="${PYTHON:-python3}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/stdout}"
PACKAGES=(recall recall_mcp recall_agent recall_hooks recall_aml)

cd "$(git rev-parse --show-toplevel)"
if [ -e setup.cfg ]; then
  echo "REFUSED: setup.cfg exists; this script generates its own mutmut configuration there" >&2
  exit 2
fi

nocode=()
if [ -n "${MUTATE_FILES:-}" ]; then
  read -r -a changed <<<"$MUTATE_FILES"
else
  # Quoted on purpose: git's pathspec `*` already crosses directories, and an unquoted glob would
  # be expanded by the shell against the working tree instead.
  pathspecs=()
  for package in "${PACKAGES[@]}"; do pathspecs+=("$package/*.py"); done
  mapfile -t touched < <(git diff --name-only --diff-filter=AM "$BASE"...HEAD -- "${pathspecs[@]}" | sort -u)
  # Mutate only files whose change ADDS executable code, most added code first, so the MAX_FILES
  # cap keeps the files a mutant can actually say something about. A file changed only by deletion
  # or by comment and docstring edits is listed, not mutated: mutmut mutates whole files, and on
  # PR 849 such files filled the cap and ran the job into its 20-minute timeout.
  changed=()
  if [ ${#touched[@]} -gt 0 ]; then
    while IFS=$'\t' read -r kind _count path; do
      if [ "$kind" = code ]; then changed+=("$path"); else nocode+=("$path"); fi
    done < <("$PYTHON" scripts/mutation_code_changes.py "$BASE" "${touched[@]}")
  fi
fi

files=()
untested=()
declare -A tests=()
declare -A dropped_tests=()
for file in "${changed[@]}"; do
  [ -f "$file" ] || continue
  module="${file%.py}"; module="${module//\//.}"; module="${module%.__init__}"
  escaped="${module//./\\.}"
  # `import recall.store` and `from recall.store import x`; NOT `import recall.store.sub`, which
  # imports a submodule rather than this file.
  pattern="^[[:space:]]*(import ${escaped}([[:space:],]|$)|from ${escaped}[[:space:]]+import"
  if [[ "$module" == *.* ]]; then
    # `from recall import store`, only for a dotted module: for a top-level package it would match
    # every `from recall import anything`.
    parent="${module%.*}"; leaf="${module##*.}"
    pattern+="|from ${parent//./\\.}[[:space:]]+import[[:space:]].*\\b${leaf}\\b"
  fi
  pattern+=")"
  mapfile -t importers < <(grep -lE "$pattern" tests/*.py 2>/dev/null || true)
  if [ ${#importers[@]} -eq 0 ]; then
    untested+=("$file")
    continue
  fi
  files+=("$file")
  # Keep the importers most coupled to this module, at most MAX_TESTS_PER_FILE. A module imported
  # almost everywhere (settings, a store) otherwise drags dozens of incidental test files into
  # every mutant: #847's run selected 32 files for two changed modules and took 25 minutes.
  # Ranked by: the test file is named after the module, then how often it names the module.
  leaf_name="${module##*.}"
  mapfile -t ranked < <(
    for t in "${importers[@]}"; do
      named=0; case "$(basename "$t")" in *"$leaf_name"*) named=1 ;; esac
      mentions=$(grep -c "$leaf_name" "$t" 2>/dev/null || true)
      printf '%d %06d %s\n' "$named" "$mentions" "$t"
    done | sort -k1,1nr -k2,2nr -k3,3 | awk '{print $3}'
  )
  for t in "${ranked[@]:0:$MAX_TESTS_PER_FILE}"; do tests["$t"]=1; done
  for t in "${ranked[@]:$MAX_TESTS_PER_FILE}"; do dropped_tests["$t"]=1; done
done
for t in "${!tests[@]}"; do unset "dropped_tests[$t]"; done

{
  echo "## Mutation testing (report only)"
  echo
  echo "Base: \`$BASE\`. Changed shipped files: ${#changed[@]}."
} >>"$SUMMARY"

if [ ${#nocode[@]} -gt 0 ]; then
  {
    echo
    echo "**Changed only by deletion, comments or docstrings** (not mutated; no added code to test):"
    printf -- "- \`%s\`\n" "${nocode[@]}"
  } >>"$SUMMARY"
fi

if [ ${#untested[@]} -gt 0 ]; then
  {
    echo
    echo "**Changed modules that no test imports** (not mutated; nothing could kill a mutant):"
    printf -- "- \`%s\`\n" "${untested[@]}"
  } >>"$SUMMARY"
fi

if [ ${#files[@]} -eq 0 ]; then
  echo >>"$SUMMARY"
  echo "Nothing to mutate." >>"$SUMMARY"
  exit 0
fi

skipped=()
if [ ${#files[@]} -gt "$MAX_FILES" ]; then
  skipped=("${files[@]:$MAX_FILES}")
  files=("${files[@]:0:$MAX_FILES}")
fi

MUTMUT_TYPE_FILTER_REPORT="$(mktemp)"
export MUTMUT_TYPE_FILTER_REPORT
trap 'rm -f setup.cfg "$MUTMUT_TYPE_FILTER_REPORT"' EXIT
{
  echo "[mutmut]"
  echo "source_paths="
  printf "    %s/\n" "${PACKAGES[@]}"
  echo "only_mutate="
  printf "    %s\n" "${files[@]}"
  echo "pytest_add_cli_args_test_selection="
  printf "    %s\n" "${!tests[@]}" | sort
  echo "pytest_add_cli_args="
  printf "    %s\n" -p no:randomly -p no:cacheprovider -q
  # Every other tracked top-level entry, because tests read the tree around them (docs, scripts,
  # results, the README) and run inside mutants/, where only what is listed here exists.
  # `.git` too (a directory in CI, a one-line gitdir pointer in a worktree), so tests that ask git
  # which files are tracked see a checkout instead of failing under mutmut's `-x`. Not GIT_DIR in
  # the environment: a test that runs `git init` in a temporary directory would then write into
  # the real repository.
  echo "also_copy="
  echo "    .git"
  git ls-files | cut -d/ -f1 | sort -u | while read -r entry; do
    case " ${PACKAGES[*]} " in *" $entry "*) continue ;; esac
    if [ -d "$entry" ]; then echo "    $entry/"; else echo "    $entry"; fi
  done
  echo "process_isolation=forkserver"
  # MUTATION_DEBUG=1 shows pytest's own output, which mutmut otherwise swallows; it is the only way
  # to see why "failed to collect stats" happened.
  [ -n "${MUTATION_DEBUG:-}" ] && echo "debug=true"
  # The mutated files named explicitly. Without them mypy follows `[tool.mypy] files`, which names
  # packages mutmut did not copy into mutants/, and mutmut then crashes opening the missing path
  # (seen on mutmut 3.8.0 with `recall_consistency`).
  # Through scripts/mutmut_type_filter.py, which drops the errors mutmut's own rewriting causes and
  # those no mutant owns: mutmut 3.8.0 raises on the second kind and loses the run (#854, where a
  # mutated Protocol stub broke every `project_current_state(self, ...)` in recall/store.py). By
  # absolute path, because mutmut runs it from inside mutants/.
  echo "type_check_command="
  printf "    %s\n" "$PYTHON" "$PWD/scripts/mutmut_type_filter.py" \
    mypy --output json --disable-error-code unused-ignore "${files[@]}"
} >setup.cfg

# Mutate only the functions a change added code to, named the way mutmut names their mutants. A
# whole-file run of one large module is thousands of mutants for a three-line change, and on PR 849
# eleven such files ran the job into its 20-minute timeout. MUTATE_FILES (a manual run) keeps the
# whole-file behaviour.
patterns=()
if [ -z "${MUTATE_FILES:-}" ]; then
  mapfile -t patterns < <("$PYTHON" scripts/mutation_code_changes.py --patterns "$BASE" "${files[@]}")
  if [ ${#patterns[@]} -eq 0 ]; then
    {
      echo
      echo "The added code is all outside functions (module-level), which mutmut does not mutate. Nothing to mutate."
    } >>"$SUMMARY"
    exit 0
  fi
fi

echo "mutating ${#patterns[@]} function(s) in ${#files[@]} file(s) against ${#tests[@]} test file(s), budget ${BUDGET}s"
run_log="$(mktemp)"
# A budget inside the job's timeout. When the runner kills a job at its timeout it shows "cancelled"
# and reports nothing; a run stopped here reports every mutant it finished. SIGINT because that is
# the signal mutmut handles: it stops its workers and keeps the results already saved (mutmut
# 3.8.0, the `except KeyboardInterrupt` in `_run`). timeout(1) answers 124 when the budget ran out,
# and 137 if the run then ignored SIGINT for --kill-after and was killed.
# Into a file, then printed, rather than through `| tee`: a process that outlives mutmut keeps a
# pipe open and tee waits for it, so the budget would stop mutmut and still hold the step until the
# job's timeout (reproduced on 2026-10-02 with a backgrounded child, 5 s past a 1 s budget).
set +e
timeout --signal=INT --kill-after=60 "$BUDGET" mutmut run --max-children "$(nproc)" "${patterns[@]}" >"$run_log" 2>&1
status=$?
set -e
cat "$run_log"
budget_reached=0
if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
  budget_reached=1
elif [ "$status" -ne 0 ]; then
  if grep -q "nothing matches" "$run_log"; then
    {
      echo
      echo "The changed functions produced no mutants (mutmut skips some functions, for example ones it cannot rewrite). Nothing to report."
    } >>"$SUMMARY"
    exit 0
  fi
  exit 1
fi

# Results restricted to the targeted functions, so mutants that were never run do not show up as
# "not checked" in the counts.
targeted() {
  "$PYTHON" -c 'import fnmatch, sys
patterns = sys.argv[1:]
for line in sys.stdin:
    name = line.split(":", 1)[0].strip()
    if not patterns or any(fnmatch.fnmatchcase(name, p) for p in patterns):
        sys.stdout.write(line)' "${patterns[@]}"
}
mapfile -t survivors < <(mutmut results 2>/dev/null | targeted | awk -F': ' '$2 == "survived" {gsub(/^ +/, "", $1); print $1}')
mapfile -t notests < <(mutmut results --all true 2>/dev/null | targeted | awk -F': ' '$2 == "no tests" {gsub(/^ +/, "", $1); print $1}')
counts="$(mutmut results --all true 2>/dev/null | targeted | awk -F': ' '{n[$2]++} END {for (k in n) printf "%s %d, ", k, n[k]}')"

{
  echo
  echo "Mutated: $(printf '`%s` ' "${files[@]}")"
  if [ ${#patterns[@]} -gt 0 ]; then
    echo
    echo "Only the ${#patterns[@]} function(s) the change added code to: $(printf '`%s` ' "${patterns[@]}")"
  fi
  echo
  echo "Against ${#tests[@]} test file(s). Outcome: ${counts%, }."
  if [ "$budget_reached" -eq 1 ]; then
    echo
    echo "**Time budget reached** (MUTATION_BUDGET_SECONDS=$BUDGET): the run was stopped, so these counts are partial and the mutants shown as \`not checked\` did not run. For the rest, raise the budget or run \`MUTATE_FILES=\"...\" bash scripts/mutation_changed.sh\` on a Linux host."
  fi
  type_filter="$("$PYTHON" scripts/mutmut_type_filter.py --summary "$MUTMUT_TYPE_FILTER_REPORT")"
  if [ -n "$type_filter" ]; then
    echo
    echo "$type_filter"
  fi
  if [ ${#skipped[@]} -gt 0 ]; then
    echo
    echo "Not run (over MAX_FILES=$MAX_FILES): $(printf '`%s` ' "${skipped[@]}")"
  fi
  if [ ${#dropped_tests[@]} -gt 0 ]; then
    echo
    echo "${#dropped_tests[@]} importing test file(s) not run (over MAX_TESTS_PER_FILE=$MAX_TESTS_PER_FILE per module); a survivor here may be killed by one of them."
  fi
  if [ ${#notests[@]} -gt 0 ]; then
    echo
    echo "${#notests[@]} mutant(s) reached by no selected test (often: covered only by database tests, which skip here)."
  fi
  echo
  if [ ${#survivors[@]} -eq 0 ]; then
    echo "No survivors."
  else
    echo "### ${#survivors[@]} survivor(s)"
    echo
    echo "Each is a change to the code that every selected test still passed. Shown up to 25."
    for name in "${survivors[@]:0:25}"; do
      echo
      echo "<details><summary><code>$name</code></summary>"
      echo
      echo '```diff'
      mutmut show "$name" 2>/dev/null | head -40
      echo '```'
      echo "</details>"
    done
  fi
} >>"$SUMMARY"
