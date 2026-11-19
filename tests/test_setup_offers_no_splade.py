"""`recall setup` does not offer a sparse backend that no search would use.

Invariant: setup offers only choices the runtime reads. Failure mode it catches: setup listed
SPLADE and wrote `RECALL_SPARSE=splade`, but no indexing or search path reads that variable
(`trusted_search` builds `HybridRetriever` without `sparse_backend`, and no production `Indexer`
gets a `sparse_encoder`), so every search ran the lexical leg while the wizard said otherwise.
The owner chose, on 2026-11-19, to stop offering it; SPLADE stays a library option through
`HybridRetriever(sparse_backend=...)`.

Red proof receipt ``setup-no-splade-offer-01``, 2026-11-19: against `a294f49bd` (setup with its
"Choose the sparse retrieval backend:" menu), the first assertion failed because the menu listed
`splade`. Green after the menu was removed.
"""

from __future__ import annotations

import io

from recall.setup import HardwareProbe, run_setup_wizard


def test_setup_neither_offers_nor_writes_splade_on_a_machine_that_could_run_it(
    tmp_path, monkeypatch
) -> None:
    probe = HardwareProbe(
        cpu_count=8,
        gpu="nvidia",
        cuda_available=True,
        free_bytes=100 * 2**30,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    output = io.StringIO()
    last = {"text": ""}

    def _print(*args, **kwargs) -> None:
        last["text"] = " ".join(str(a) for a in args)
        print(*args, **kwargs, file=output)

    def _answer(_prompt: str = "") -> str:
        # Answer by the prompt just printed, so the run completes whatever questions it asks.
        prompt = last["text"]
        if prompt.endswith("Select a number: "):
            return "1"
        if prompt.rstrip().endswith("]"):
            return "n"
        return ""

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        input_fn=_answer,
        print_fn=_print,
    )

    assert "splade" not in output.getvalue().lower()
    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "RECALL_SPARSE=splade" not in env
    assert "RECALL_SPARSE=fts" in env
