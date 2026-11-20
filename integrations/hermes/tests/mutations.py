"""Red proof: apply each mutation, run its test, require a failure in the test's own assertion.

Run from the repository root: `python tests/mutations.py`. Each row breaks the production line the
named test protects; the script restores the file afterwards whatever happens, and prints, per
mutation, RED with the first failing assertion line, or SURVIVED. Exit status 1 if any survived or
failed for a reason other than an assertion (an ImportError or a collection error is not red).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = "re_call_hermes/provider.py"
R = "re_call_hermes/render.py"
C = "re_call_hermes/config.py"
M = "re_call_hermes/mcp_client.py"
INIT = "re_call_hermes/__init__.py"
S = "re_call_hermes/surface.py"

MUTATIONS = [
    ("RENDER-CITABLE", R, 'citable_ids is None or str(hit.get("chunk_id")) in citable_ids', "True",
     "tests/test_render.py::test_only_citable_hits_are_presented_as_current"),
    ("RENDER-RETIRED", R, "if verdict in RETIRED_VERDICTS:", "if verdict in ():",
     "tests/test_render.py::test_superseded_hits_are_flagged_with_their_replacement_and_never_counted"),
    ("RENDER-ABSTAIN", R, 'elif not abstained and verdict == "ok"', 'elif verdict == "ok"',
     "tests/test_render.py::test_abstention_says_so_and_cites_nothing"),
    ("PREFETCH-STATUS", P, "        self._status = None\n        if self._settings is None or is_trivial_prompt",
     "        if self._settings is None or is_trivial_prompt",
     "tests/test_provider.py::test_prefetch_injects_citable_memory_and_reports_the_count"),
    ("TRIVIAL", P, "if self._settings is None or is_trivial_prompt(query):", "if self._settings is None:",
     "tests/test_provider.py::test_trivial_prompts_never_reach_the_server"),
    ("INIT-WAITS", P, "            starter.join()", "            pass",
     "tests/test_provider.py::test_initialize_returns_only_once_the_server_is_ready"),
    ("COLD-START-WAIT", P, "                self._start_done.wait(PREFETCH_BUDGET_S - MIN_SEARCH_S)",
     "                pass",
     "tests/test_provider.py::test_the_first_prefetch_waits_for_a_starting_server"),
    ("FAILED-START-NO-WAIT", P,
     "            if self._starting:\n                self._start_done.wait(PREFETCH_BUDGET_S - MIN_SEARCH_S)",
     "            if True:\n                self._ready.wait(PREFETCH_BUDGET_S - MIN_SEARCH_S)",
     "tests/test_provider.py::test_a_failed_start_is_not_waited_on"),
    ("DEADLINE", P, "        timeout = min(PREFETCH_DEADLINE_S, PREFETCH_BUDGET_S - (time.monotonic() - started))",
     "        timeout = TOOL_TIMEOUT_S",
     "tests/test_provider.py::test_prefetch_gives_up_before_hermes_does"),
    ("TOOL-ERROR", P, "        except ToolError as exc:\n            return render_unavailable(str(exc)).text\n", "",
     "tests/test_provider.py::test_a_refused_search_is_reported_not_hidden"),
    ("ENV", C, "env = {k: base[k] for k in PASSTHROUGH_ENV if k in base}", "env = dict(base)",
     "tests/test_provider.py::test_the_child_gets_a_minimal_environment_and_the_data_directory"),
    ("ENV-CWD", P, "cwd=settings.data_dir,  # no project .env", "cwd=Path.cwd(),  # no project .env",
     "tests/test_provider.py::test_the_child_gets_a_minimal_environment_and_the_data_directory"),
    ("TOOLS", P, "        if self._primary:\n            return list(TOOLS)", "        if True:\n            return list(TOOLS)",
     "tests/test_provider.py::test_write_tools_are_offered_only_to_the_primary_agent_on_lite"),
    ("FORWARD", P, 'if v not in (None, "")}', "}",
     "tests/test_provider.py::test_tool_calls_forward_to_the_server_tool_with_their_arguments"),
    ("MIRROR", P, '            arguments["supersedes"] = supersedes\n', "            pass\n",
     "tests/test_provider.py::test_builtin_memory_writes_are_mirrored_with_supersession"),
    ("MIRROR-GUARD", P, 'if action in ("replace", "remove") and previous is None:', "if False:",
     "tests/test_provider.py::test_a_destructive_mirror_without_the_entry_identity_is_skipped"),
    ("RESTART", P, "            self._maybe_restart()\n            raise NotSent(self._start_error",
     "            raise NotSent(self._start_error", "tests/test_provider.py::test_a_crashed_server_is_restarted"),
    ("STORE", C, "if store is not None and not store.is_file():", "if False:",
     "tests/test_provider.py::test_a_missing_store_makes_the_provider_unavailable_with_the_setup_command"),
    ("PASSWORD", C, "if not self.lite and (urlsplit(self.dsn).password or _DSN_PASSWORD.search(self.dsn)):", "if False:",
     "tests/test_provider.py::test_a_dsn_with_a_password_is_refused"),
    ("REGISTER", INIT, "    ctx.register_memory_provider(ReCallMemoryProvider())",
     "    ctx.register_memory_provider(_SHARED)\n\n\n_SHARED = ReCallMemoryProvider()",
     "tests/test_provider.py::test_register_hands_out_a_new_instance_each_time"),
    ("LATE-ANSWER", M, "                    if message[\"id\"] in self._abandoned:", "                    if False:",
     "tests/test_mcp_client.py::test_a_late_answer_is_dropped_and_the_next_call_gets_its_own"),
    # Added after the 2026-11-08 bug review: one row per defect it found.
    ("MIRROR-ONCE", P, '            if action == "remove" or present:', '            if action == "remove":',
     "tests/test_provider.py::test_an_identical_entry_is_mirrored_once"),
    ("DAT-001-DUP-ADD", P, '            if action == "remove" or present:', '            if action == "remove":',
     "tests/test_provider.py::test_a_duplicate_add_does_not_keep_a_removed_entry_alive"),
    ('MIRROR-NO-RETRY', P, '        answer = self._call("recall_remember", arguments, TOOL_TIMEOUT_S)\n',
     '        try:\n            answer = self._call("recall_remember", arguments, TOOL_TIMEOUT_S)\n        except McpError:\n            answer = self._call("recall_remember", arguments, TOOL_TIMEOUT_S)\n',
     'tests/test_provider.py::test_a_lost_remember_answer_is_preserved_without_retry'),
    ("WRITER-SURVIVES", P, "            except Exception:  # one bad write", "            except McpError:  # one bad write",
     "tests/test_provider.py::test_a_corrupt_mirror_file_does_not_end_mirroring"),
    ("CLOSED-DURING-START", P,
     "            if self._closed:  # shutdown() ran while this child was starting\n                client.close()\n                return\n",
     "", "tests/test_provider.py::test_shutdown_while_the_server_starts_leaves_no_server"),
    ("ONE-LAUNCH", P, "            if self._closed or self._starting:\n                return",
     "            if self._closed:\n                return",
     "tests/test_provider.py::test_two_launches_start_one_server"),
    ("STATUS-TOKEN", P, "if rendered.citable and self._prefetch_token is token:", "if rendered.citable:",
     "tests/test_provider.py::test_an_abandoned_prefetch_cannot_set_a_later_turns_status"),
    ("CONFIG-ERROR", C, "    except (ValueError, TypeError) as exc:  # reported by problem()",
     "    except TypeError as exc:  # reported by problem()",
     "tests/test_provider.py::test_a_corrupt_config_is_reported_not_raised"),
    ("WEDGED-SEND", M, "        self._outbox.put((json.dumps(message, separators=(\",\", \":\")) + \"\\n\").encode(\"utf-8\"))",
     "        proc.stdin.write((json.dumps(message) + \"\\n\").encode(\"utf-8\"))\n        proc.stdin.flush()",
     "tests/test_mcp_client.py::test_a_wedged_server_cannot_hold_a_caller_past_its_deadline"),
    ("WEDGED-CLOSE", M,
     "                self._thread_factory(lambda: _close_quietly(stdin), name=\"re-call-mcp-close\").start()",
     "                _close_quietly(stdin)",
     "tests/test_mcp_client.py::test_a_wedged_server_cannot_hold_a_caller_past_its_deadline"),
    ("MALFORMED", M, "        parts = content if isinstance(content, list) else []", "        parts = content",
     "tests/test_mcp_client.py::test_a_malformed_answer_is_an_mcp_error"),
    # Found by the contract test against real Hermes v0.23.0: tools are indexed before initialize().
    ("TOOLS-BEFORE-INIT", P, "        settings = self._settings or load_settings(self._home_for_probe())\n",
     "        settings = self._settings\n        if settings is None:\n            return []\n",
     "tests/test_provider.py::test_tools_are_declared_before_initialize"),
    # Added after the differential review of b3c7cc9.
    # CCA run 20261108T171507Z: one row per fix whose test is red-proved by mutation rather than
    # (or as well as) the pre-fix snapshot of `cca_checks tautology verify`.
    ("SEC-001-CWD", C, "            found = _which_on_path(argv[0])", "            found = __import__('shutil').which(argv[0])",
     "tests/test_audit_fixes.py::test_the_server_command_is_never_taken_from_the_working_directory"),
    ("SEC-001-REL", C, "        if whole.is_absolute() and whole.is_file():", "        if whole.is_file():",
     "tests/test_audit_fixes.py::test_a_relative_server_path_is_refused"),
    ("SEC-002", C, "(urlsplit(self.dsn).password or _DSN_PASSWORD.search(self.dsn))", "(urlsplit(self.dsn).password)",
     "tests/test_audit_fixes.py::test_a_password_in_any_dsn_form_is_refused"),
    ("ENV-001", C, '        mirror_builtin = _as_bool(values["mirror_builtin"])', '        mirror_builtin = bool(values["mirror_builtin"])',
     "tests/test_audit_fixes.py::test_mirror_builtin_false_from_the_setup_wizard_turns_mirroring_off"),
    ("ENV-002", C, "    return path if path.is_absolute() else base / path", "    return path",
     "tests/test_audit_fixes.py::test_relative_store_and_memo_paths_resolve_under_the_data_folder"),
    ("ENV-004", C, '    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",\n', "",
     "tests/test_audit_fixes.py::test_the_child_gets_proxy_and_model_cache_variables_and_a_profile_embedding_cache"),
    ("DOC-001", C, "            RECALL_EMBED_CACHE=str(self.data_dir / EMBED_CACHE_FILE),\n", "",
     "tests/test_audit_fixes.py::test_the_child_gets_proxy_and_model_cache_variables_and_a_profile_embedding_cache"),
    ("DEP-001", "pyproject.toml", '[project]\nname = "recall-hermes"', '[build-system]\nrequires = ["setuptools"]\n\n[project]\nname = "recall-hermes"',
     "tests/test_audit_fixes.py::test_the_plugin_declares_nothing_hermes_would_install"),
    ("DAT-002", P, "        content = content.strip()\n        previous = previous.strip() if previous is not None else None\n", "",
     "tests/test_audit_fixes.py::test_whitespace_around_an_entry_does_not_strand_its_memo"),
    ('MIRROR-DESTINATION', P, '            raise ToolError("mirror destination is changed or unknown; reconcile mirror.json before mirroring")',
     '            pass',
     'tests/test_audit_fixes.py::test_an_unbound_legacy_mirror_is_preserved_without_deletion'),
    ("DAT-004-REPLAY", P, "            try:\n                self._drain_pending()\n            except Exception:\n                logger.warning(\"re-call: could not replay",
     "            try:\n                pass\n            except Exception:\n                logger.warning(\"re-call: could not replay",
     "tests/test_audit_fixes.py::test_an_unsubmitted_journaled_write_is_completed_with_its_request_id"),
    ('MIRROR-INFLIGHT', P, '            if request_id in state["submitted"]:\n                raise McpError("mirror write is in flight or unresolved; replay refused, reconciliation required")\n',
     '',
     'tests/test_audit_fixes.py::test_an_inflight_write_cannot_be_replayed_by_another_instance'),
    ('MIRROR-UNKNOWN', P, '        except ToolError:\n            self._settle(path, request_id)',
     '        except (ToolError, McpError):\n            self._settle(path, request_id)',
     'tests/test_audit_fixes.py::test_an_unknown_write_is_retained_and_blocks_later_ones'),
    ("MIRROR-ORDER", P, '            if request_id in state["submitted"]:\n                raise McpError("mirror write is in flight or unresolved; replay refused, reconciliation required")\n', "",
     "tests/test_audit_fixes.py::test_an_unknown_write_blocks_a_later_replace"),
    ("DAT-004-RECHECK", P, "            if all(op[0] != request_id for op in state[\"pending\"]):\n                return  # another instance on this profile applied it already\n", "",
     "tests/test_audit_fixes.py::test_a_journaled_write_another_instance_applied_is_not_applied_again"),
    ("DAT-003-SETTLE", P, "            if all(op[0] != request_id for op in state[\"pending\"]):\n                return  # settled meanwhile by another instance, whose later writes this must not undo\n", "",
     "tests/test_audit_fixes.py::test_a_late_settle_does_not_undo_another_instances_later_write"),
    ("DAT-005", P,'        if isinstance(answer, dict) and answer.get("conflict") in TARGET_CONFLICTS and supersedes:', "        if False:",
     "tests/test_audit_fixes.py::test_a_refused_correction_still_stores_the_new_text"),
    ("DAT-005-FINAL", P, '        if isinstance(answer, dict) and answer.get("conflict") in TARGET_CONFLICTS and supersedes:',
     '        if isinstance(answer, dict) and answer.get("refused") and supersedes:',
     "tests/test_audit_fixes.py::test_a_refusal_that_is_not_a_target_conflict_is_final"),
    ("REFUSAL-NO-CONTENT", M, '        if result.get("isError"):  # a refusal stays a refusal', '        if False:  # a refusal stays a refusal',
     "tests/test_mcp_client.py::test_a_refusal_without_content_is_still_a_refusal"),
    # The shared per-user Lite store (owner decision 2026-11-13), and the profile's old store.
    ("SHARED-PRESENT", C, "    if (shared / STORE_FILE).is_file():\n        _notify", "    if False:\n        _notify",
     "tests/test_shared_store.py::test_the_shared_store_is_chosen_when_present"),
    ("SHARED-UNUSED-NOTICE", C, '        _notify(old, f"re-call: using the shared store', '        (old, f"re-call: using the shared store',
     "tests/test_shared_store.py::test_the_shared_store_is_chosen_when_present"),
    ("OLD-KEPT", C, "{data_dir} to {shared}\")\n    return data_dir", "{data_dir} to {shared}\")\n    return shared",
     "tests/test_shared_store.py::test_the_old_store_is_kept_and_the_notice_is_logged_once"),
    ("NOTICE-ONCE", C, "    if old not in _NOTIFIED:", "    if True:",
     "tests/test_shared_store.py::test_the_old_store_is_kept_and_the_notice_is_logged_once"),
    ("NEITHER-SHARED", C, "    if not old.is_file():\n        return shared", "    if not old.is_file():\n        return data_dir",
     "tests/test_shared_store.py::test_the_shared_default_applies_when_neither_exists"),
    ("SHARED-MEMOS", C, "memos or store_dir / MEMO_DIR", "memos or data_dir / MEMO_DIR",
     "tests/test_shared_store.py::test_the_shared_default_applies_when_neither_exists"),
    ("MEMO-WITH-STORE", C, "memos or store_dir / MEMO_DIR", "memos or shared_dir() / MEMO_DIR",
     "tests/test_shared_store.py::test_the_store_and_the_memo_root_resolve_together"),
    ("ASSISTANT-SUBFOLDER", C, 'ASSISTANT_SUBFOLDER = "hermes"', 'ASSISTANT_SUBFOLDER = "assistant"',
     "tests/test_shared_store.py::test_the_store_and_the_memo_root_resolve_together"),
    ("XDG-FALLBACK", C, 'Path.home() / ".config") / DATA_DIR', "Path.home()) / DATA_DIR",
     "tests/test_shared_store.py::test_the_shared_default_applies_when_neither_exists"),
    ("XDG-RELATIVE", C, "(Path(xdg) if os.path.isabs(xdg) else", "(Path(xdg) if xdg else",
     "tests/test_shared_store.py::test_the_shared_default_applies_when_neither_exists"),
    ("DSN-WINS", C, '        store = _absolute(Path(dsn[len("sqlite:///"):]), data_dir) if dsn else store_dir / STORE_FILE',
     "        store = store_dir / STORE_FILE", "tests/test_shared_store.py::test_an_explicit_dsn_wins"),
    ("DSN-MEMOS", C, "    store_dir = data_dir if dsn or custom_memos else _store_dir(data_dir)", "    store_dir = _store_dir(data_dir)",
     "tests/test_shared_store.py::test_an_explicit_dsn_wins"),
    ("PER-PROFILE-DATA", C, "        data_dir=data_dir,\n", "        data_dir=store_dir,\n",
     "tests/test_shared_store.py::test_config_and_logs_stay_per_profile"),
    ("PER-PROFILE-CONFIG", C, "read_json(data_dir / CONFIG_FILE)", "read_json(shared_dir() / CONFIG_FILE)",
     "tests/test_shared_store.py::test_config_and_logs_stay_per_profile"),
    ("PER-PROFILE-SAVE", C, "    path = Path(hermes_home) / DATA_DIR / CONFIG_FILE\n", "    path = shared_dir() / CONFIG_FILE\n",
     "tests/test_shared_store.py::test_config_and_logs_stay_per_profile"),
    ("PER-PROFILE-LOG", P, 'stderr_path=settings.data_dir / "server.log",', 'stderr_path=settings.store_path.parent / "server.log",',
     "tests/test_shared_store.py::test_config_and_logs_stay_per_profile"),
    ("MIRROR-REBIND", P, '            if found is None:\n                raise ToolError("mirror journal names',
     '            if True:\n                raise ToolError("mirror journal names',
     "tests/test_shared_store.py::test_the_mirror_journal_follows_the_old_store_moved_as_the_notice_says"),
    ("MAPPING-CONTRADICTION", P, '        if _digest(key.rpartition(":")[0], text) != key:\n            return None',
     "        if False:\n            return None",
     "tests/test_shared_store.py::test_a_journal_does_not_follow_a_deleted_store_to_another_one"),
    ("IDENTITY-TRUSTED", P, '        changed = recorded != destination or state["store"] not in (None, identity)',
     '        changed = state["store"] not in (None, identity)',
     "tests/test_shared_store.py::test_a_reused_file_number_is_no_evidence_of_a_move"),
    ("SAME-PATH", P, '        changed = recorded != destination or state["store"] not in (None, identity)',
     "        changed = recorded != destination",
     "tests/test_shared_store.py::test_another_store_at_the_same_path_is_checked"),
    ("MIRROR-STAMP", P, "            atomic_json_write(path, {**raw, **binding})  # records the identity a later move is recognised by\n",
     "            pass\n", "tests/test_shared_store.py::test_another_store_at_the_same_path_is_checked"),
    ("MAPPING-FOUND", P, "    return (held, unread) if held or not memos else None", "    return (held, unread)",
     "tests/test_shared_store.py::test_a_store_moved_without_its_memo_folder_is_not_followed"),
    ("MEMO-OWN-STORE", C, "    store_dir = data_dir if dsn or custom_memos else _store_dir(data_dir)",
     "    store_dir = data_dir if dsn else _store_dir(data_dir)",
     "tests/test_shared_store.py::test_a_custom_memo_folder_keeps_a_store_of_its_own"),
    ("DSN-SHARED-MEMOS", C, "            store_dir = store.parent  # a dsn naming the shared store", "            pass  # a dsn naming the shared store",
     "tests/test_shared_store.py::test_the_shared_store_and_the_shared_memo_folder_go_together"),
    ("SHARED-MEMOS-AS-BLANK", C,
     "    custom_memos = memos is not None and (shared is None or not _same_path(memos, shared / MEMO_DIR))",
     "    custom_memos = memos is not None",
     "tests/test_shared_store.py::test_the_shared_store_and_the_shared_memo_folder_go_together"),
    ("SHARED-MEMO-ROOT-GUARD", C,
     "        if store is not None and shared is not None and _same_path(store, shared / STORE_FILE) != _same_path(",
     "        if False and _same_path(store, shared / STORE_FILE) != _same_path(",
     "tests/test_shared_store.py::test_the_shared_store_and_the_shared_memo_folder_go_together"),
    ("GUARD-BOTH-WAYS", C, "_same_path(store, shared / STORE_FILE) != _same_path(", "_same_path(store, shared / STORE_FILE) > _same_path(",
     "tests/test_shared_store.py::test_the_shared_store_and_the_shared_memo_folder_go_together"),
    ("STAMP-KEEPS-RAW", P, "atomic_json_write(path, {**raw, **binding})", "atomic_json_write(path, {**state, **binding})",
     "tests/test_shared_store.py::test_recording_the_identity_keeps_the_journal_as_read"),
    ("REBIND-NEEDS-NO-STAMP", P, "            if found is None:\n", '            if found is None or state["store"] is None:\n',
     "tests/test_shared_store.py::test_a_move_needs_no_earlier_start_on_the_old_store"),
    ("NONE-KEEPS-IDENTITY", P, '        state["store"] = identity if identity is not None else state["store"]',
     '        state["store"] = identity', "tests/test_shared_store.py::test_a_missing_identity_keeps_the_recorded_one"),
    ("PRUNE-ABSENT", P, '            state["memos"] = held  # so only', "            pass  # so only",
     "tests/test_shared_store.py::test_a_journaled_write_that_rebinds_keeps_the_drop"),
    ("PRUNE-PERSIST", P, '            binding["memos"] = held\n', "            pass\n",
     "tests/test_shared_store.py::test_a_rebound_journal_drops_mappings_whose_memo_is_absent"),
    ("UNCONFIRMED-USED", P, "                    memos[old_key] = candidate  # confirmed", "                    pass  # confirmed",
     "tests/test_shared_store.py::test_a_memo_in_place_after_the_rebind_is_still_managed"),
    ("UNCONFIRMED-PERSIST", P, '            binding["unconfirmed"] = state["unconfirmed"]\n', "",
     "tests/test_shared_store.py::test_a_memo_in_place_after_the_rebind_is_still_managed"),
    ("UNCONFIRMED-CHECKED", P, "                elif _digest(target, text) == old_key:\n                    memos[old_key] = candidate",
     "                else:\n                    memos[old_key] = candidate",
     "tests/test_shared_store.py::test_a_rebound_journal_drops_mappings_whose_memo_is_absent"),
    # 2026-11-20 review of the shared store commits (tests/test_move_review_fixes.py).
    ("SWITCH-FRESH-MAPPING", P,
     "if found is None and self._switched_to_shared(recorded):", "if False:",
     "tests/test_move_review_fixes.py::test_a_switch_to_the_shared_store_keeps_mirroring"),
    ("RETIRE-WHEN-READ", P, 'state["retire_when_read"][old_key] = candidate',
     'state["unconfirmed"][old_key] = candidate',
     "tests/test_move_review_fixes.py::test_an_entry_removed_before_its_memo_is_in_place_is_forgotten_once_it_is"),
    ("NO-HOME-REASON", C, "    except RuntimeError:\n        return None", "    except ValueError:\n        return None",
     "tests/test_move_review_fixes.py::test_no_home_directory_is_a_reason_not_an_exception"),
    ("WRITER-STARTED-FIRST", P, "            writer.start()\n            self._writer = writer",
     "            self._writer = writer\n            writer.start()",
     "tests/test_move_review_fixes.py::test_a_shutdown_before_the_writer_starts_does_not_join_it"),
]

# Run only with HERMES_SRC set: without a real Hermes these tests skip, which proves nothing.
CONTRACT_MUTATIONS = [
    ("CONTRACT-ABC", P, "    def get_tool_schemas(self) -> list[dict[str, Any]]:\n",
     "    def _get_tool_schemas(self) -> list[dict[str, Any]]:\n",
     "tests/test_contract.py::test_the_provider_is_a_concrete_subclass_of_the_real_abc"),
    ("CONTRACT-LOADER", "__init__.py", "    from .re_call_hermes import ReCallMemoryProvider, register\n",
     "    pass\n", "tests/test_contract.py::test_hermes_loads_the_directory_install_by_name"),
    ("CONTRACT-MANAGER", P, "        if self._writer is None:\n            return\n        # One id per write",
     "        return\n        # One id per write",
     "tests/test_contract.py::test_memory_manager_routes_prefetch_tools_and_writes"),
]

# Run only with RECALL_HERMES_E2E=1 and RECALL_E2E_BIN set, on a host where local embedding is acceptable.
E2E_MUTATIONS = [
    ("E2E-SUPERSEDES-DROPPED", P, 'if v not in (None, "")}',
     'if v not in (None, "") and k != "supersedes"}',
     "tests/test_e2e_lite.py::test_supersession_and_abstention_through_a_real_lite_store"),
    ("SURFACE-SCHEMA-DRIFT", S, '"title": "Max Items"', '"title": "Max items"',
     "tests/test_e2e_lite.py::test_the_provider_serves_recall_mcps_own_six_tool_schemas"),
    ("SURFACE-DESCRIPTION-DRIFT", S, "Return the tenant scoped current projection", "Return the current projection",
     "tests/test_e2e_lite.py::test_the_provider_serves_recall_mcps_own_six_tool_schemas"),
]


def run(name: str, path: str, old: str, new: str, test: str) -> bool:
    target = ROOT / path
    original = target.read_bytes()
    text = original.decode("utf-8")
    if text.count(old) != 1:
        print(f"{name}: BAD MUTATION (anchor found {text.count(old)} times)")
        return False
    target.write_bytes(text.replace(old, new).encode("utf-8"))
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test],
                             cwd=ROOT, capture_output=True, text=True, timeout=300)
    finally:
        target.write_bytes(original)
    lines = out.stdout.splitlines()
    failed = out.returncode == 1 and any(" failed" in line for line in lines[-3:])
    reason = next((line.strip() for line in lines if line.startswith("E ")), "")
    body = reason[1:].strip()
    # `assert x` and `assert x, "message"` both count; any other exception type does not.
    by_assertion = failed and (body.startswith("assert") or body.startswith("AssertionError"))
    if failed and by_assertion:
        print(f"{name}: RED  {test.split('::')[1]}  {reason[:160]}")
        return True
    print(f"{name}: {'NOT AN ASSERTION' if failed else 'SURVIVED'}  {test}  {reason[:160]}")
    return False


if __name__ == "__main__":
    import os

    rows = MUTATIONS + (CONTRACT_MUTATIONS if os.environ.get("HERMES_SRC") else [])
    rows += E2E_MUTATIONS if os.environ.get("RECALL_HERMES_E2E") == "1" else []
    if len(sys.argv) > 1:
        rows = [row for row in rows if row[0] in sys.argv[1:]]
    ok = all([run(*row) for row in rows])
    sys.exit(0 if ok else 1)
