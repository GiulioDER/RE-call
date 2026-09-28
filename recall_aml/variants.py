"""Registered RE-call Hosted behaviors used by local attribution runs."""

from __future__ import annotations

from dataclasses import dataclass


REPOSITORY_KINDS = frozenset(
    {"architectural decision", "constraint", "repository fact"}
)
#: What ``HostedVariant.multimodal_scope`` may be; see that field.
MULTIMODAL_SCOPES = ("route", "preserve", "dual")
EXPERIENCE_KINDS = frozenset(
    {
        "symptom",
        "root cause",
        "failed attempt",
        "successful repair",
        "procedure",
        "validation",
    }
)


@dataclass(frozen=True)
class HostedVariant:
    name: str
    raw: bool
    compiler: bool
    facets: bool
    reranker: bool
    pack: bool
    context_chars: int | None
    learned_sparse: bool = False
    task_conditioned: bool = False
    code_aware: bool = False
    anchor_compiler: bool = False
    anchor_compiler_version: int = 2
    raw_rescue_tail: bool = False
    compiled_kinds: frozenset[str] | None = None
    drop_compiler_fallback: bool = False
    multimodal_preserve: bool = False
    multimodal_native: bool = False
    graph_sidecar: bool = False
    embedding_profile: str = "voyage-context-4-v1"
    canonical_bm25: bool = False
    word_window_size: int | None = None
    word_window_stride: int | None = None
    exact_dense: bool = False
    stable_window_order: bool = False
    content_only_windows: bool = False
    #: Choose the window renderer per Add: content-only for a coding trajectory, timestamp, role
    #: and content otherwise (``recall_aml.window_format.looks_like_coding``).
    per_track_windows: bool = False
    #: Prefix each returned text item with its own ``created_at`` at Search time; nothing stored
    #: or ranked changes (``recall_aml.window_format.dated_items``).
    dated_search_content: bool = False
    #: Also date each image-bearing item, whose content is a list of parts, by one leading text
    #: part (``recall_aml.window_format.dated_multimodal_items``). ``RECALL_AML_DATED_MULTIMODAL``
    #: overrides it for an experiment.
    dated_multimodal_content: bool = False
    #: Which Searches may return a multimodal memory's images (``MULTIMODAL_SCOPES``): ``route``
    #: only on the multimodal route; ``preserve`` also attaches the images of whatever text
    #: retrieval found; ``dual`` also runs the visual leg on every query.
    #: ``RECALL_AML_MULTIMODAL_SCOPE`` overrides it for an experiment.
    multimodal_scope: str = "route"
    #: T-1: resolve relative time phrases in returned text items against each item's own
    #: date (``recall_aml.temporal_render``), on every route except ``code``.
    #: ``RECALL_AML_RESOLVE_RELATIVE_TIMES`` overrides it.
    resolved_relative_times: bool = False
    #: W2: speaker marks in returned windows (``recall_aml.window_compose``).
    #: ``RECALL_AML_SPEAKER_MARKS`` overrides it.
    speaker_marks: bool = False
    #: W2: the returned windows of one Add as one item in source order.
    #: ``RECALL_AML_SESSION_COALESCE`` overrides it.
    session_coalesce: bool = False
    #: W4: where T-1 applies. ``route``: every route except ``code`` (served since 2026-09-26).
    #: ``content``: every route, skipping items and phrases that look like code
    #: (``recall_aml.temporal_render.looks_like_code``). ``RECALL_AML_T1_GATE`` overrides it.
    relative_times_gate: str = "route"
    #: W4: T-1's rendering (``recall_aml.temporal_render.RENDER_VERSIONS``); ``v2`` keeps week
    #: expressions relative and anchored. ``RECALL_AML_T1_RENDER`` overrides it.
    relative_times_render: str = "v1"
    #: LW-1: after the top 10, return each retrieved session's last window
    #: (``recall_aml.last_window``). ``RECALL_AML_LAST_WINDOW`` overrides it.
    last_window_append: bool = False
    #: K-2: set same-subject, different-day items side by side, newest first, inside the top
    #: 30 (``recall_aml.conflict_order``). ``RECALL_AML_SAME_SUBJECT_ORDER`` overrides it.
    same_subject_order: bool = False
    #: MM-4: read each image at Add into a sidecar (``recall_aml.image_text``); use the sidecars
    #: as a retrieval leg; append them, labelled, to returned image messages. Overrides:
    #: ``RECALL_AML_IMAGE_TEXT_BUILD``, ``RECALL_AML_IMAGE_TEXT_LEG``, ``RECALL_AML_IMAGE_TEXT_SHOWN``.
    image_text_build: bool = False
    image_text_leg: bool = False
    image_text_shown: bool = False
    #: What an anchored compile sends of the session's earlier compiled records
    #: (``recall_aml.compiler.PRIOR_RECORD_MODES``).
    anchor_prior_records: str = "with-ids"
    #: Bound the v3 anchored compile (``anchor_compiler_version`` 3 only) to this many encoded
    #: characters of the Add's own anchors (None: no limit). Over it, the Add is compiled from its
    #: first and last anchors fitted to the limit (since 2026-09-28; until then it was skipped).
    #: Only an Add whose first anchor alone does not fit is skipped, which is a compiler fallback:
    #: raw windows and atomic views are stored as usual, and fallback records are built unless
    #: ``drop_compiler_fallback`` is set.
    anchor_compile_max_payload_chars: int | None = None
    #: What an anchored compile asks the model to write (``recall_aml.compiler.ANCHOR_OUTPUT_MODES``).
    anchor_compile_output: str = "full"
    #: Send only the newest prior records whose encoded size fits this many characters (None:
    #: count bound only). ``recall_aml.compiler.fit_prior_records``. C9's 145,000 is the largest
    #: round value that keeps its anchors-only limit plus this budget under
    #: ``ANCHOR_PAYLOAD_BUDGET_CHARS``; 24 records that each cite one full anchor (about 100,000)
    #: fit whole. It was 40,000 until 2026-09-27, which trimmed ordinary Coding sessions. It bounds
    #: characters, not tokens: escaped CJK or base64 inside it can still pass the model's window,
    #: and the compiler then resends once without prior records (``_compile_anchored``).
    anchor_prior_records_max_chars: int | None = None
    #: Resend a compile whose answer stopped at ``max_tokens``. False (C9) raises at the first
    #: cut-off answer instead, since a resend rarely recovers it and costs the full prompt again.
    compile_resend_truncated: bool = True
    #: When the compile provider answers HTTP 402 (out of credit), fail the Add with 503 and
    #: ``Retry-After`` instead of storing it raw-only. AML retries an Add 503 up to 32 times with
    #: the same request, so a top-up within about 30 minutes lets the run continue, and after
    #: that the run stops and is resumed from AML's site. The Textual Full of 2026-09-25 to 27
    #: stored about 71% of its bulk Adds without compiled records while the credit was out.
    stop_on_credit_exhausted: bool = False
    context_specialist: bool = False
    context_embedding_profile: str = "voyage-context-4-v1"
    atomic_rescue: bool = False
    #: Build atomic views inside Add and select from them at Search, instead of loading a
    #: hand-built, corpus-fingerprint-bound file artifact that the official Add then Search flow
    #: never gives a chance to exist (``recall_aml.atomic_views``).
    atomic_views_at_add: bool = False
    #: What ``RECALL_ATOMIC_RESCUE_MODE`` and ``RECALL_ATOMIC_RESCUE_PLACEMENT`` mean when unset.
    atomic_rescue_default_mode: str = "off"
    atomic_rescue_default_placement: str = "dense"


ATTRIBUTION_VARIANTS = (
    HostedVariant("A0_raw", True, False, False, False, False, None),
    HostedVariant("A1_compiler", True, True, False, False, False, None),
    HostedVariant("A2_facets", True, True, True, False, False, None),
    HostedVariant("A3_rerank", True, True, True, True, False, None),
    HostedVariant("A4_pack_5000", True, True, True, True, True, 5_000),
    HostedVariant("A4_pack_7000", True, True, True, True, True, 7_000),
    HostedVariant("A4_pack_9000", True, True, True, True, True, 9_000),
)
EXPERIENCE_VARIANTS = (
    HostedVariant("E0_raw", True, False, False, False, False, None),
    HostedVariant("E1_compiled", False, True, False, False, False, None),
    HostedVariant("E2_compiled_raw", True, True, False, False, False, None),
)
CODING_MATRIX_VARIANTS = (
    HostedVariant("C0_raw_lexical", True, False, False, False, False, None),
    HostedVariant("C1_splade", True, False, False, False, False, None, learned_sparse=True),
    HostedVariant("C2_procedure", True, True, False, False, False, None, learned_sparse=True),
    HostedVariant("C3_rerank", True, True, False, True, False, None, learned_sparse=True),
    HostedVariant(
        "C4_task_pack",
        True,
        True,
        True,
        True,
        True,
        7_000,
        learned_sparse=True,
        task_conditioned=True,
    ),
)
CODE4_OFFICIAL_VARIANTS = (
    HostedVariant(
        "C5_code4_bm25",
        True,
        False,
        False,
        False,
        False,
        None,
        embedding_profile="voyage-code-4-v1",
        canonical_bm25=True,
        word_window_size=160,
        word_window_stride=120,
    ),
    HostedVariant(
        "C6_code4_exact_bm25",
        True,
        False,
        False,
        False,
        False,
        None,
        embedding_profile="voyage-code-4-v1",
        canonical_bm25=True,
        word_window_size=160,
        word_window_stride=120,
        exact_dense=True,
        stable_window_order=True,
        content_only_windows=True,
    ),
)
SPECIALIST_VARIANTS = (
    HostedVariant(
        "C7_routed_specialists",
        True,
        False,
        False,
        False,
        False,
        None,
        multimodal_preserve=True,
        multimodal_native=True,
        embedding_profile="voyage-code-4-v1",
        canonical_bm25=True,
        word_window_size=160,
        word_window_stride=120,
        exact_dense=True,
        stable_window_order=True,
        content_only_windows=True,
        context_specialist=True,
    ),
    HostedVariant(
        "C8_routed_specialists_grounded_graph",
        True,
        True,
        False,
        False,
        False,
        None,
        anchor_compiler=True,
        anchor_compiler_version=3,
        drop_compiler_fallback=True,
        graph_sidecar=True,
        multimodal_preserve=True,
        multimodal_native=True,
        embedding_profile="voyage-code-4-v1",
        canonical_bm25=True,
        word_window_size=160,
        word_window_stride=120,
        exact_dense=True,
        stable_window_order=True,
        content_only_windows=True,
        context_specialist=True,
        atomic_rescue=True,
    ),
    # C8 with its atomic stage made servable under the official contract: views are built and
    # persisted inside Add, so every Search, including a streaming one between two Adds, can use
    # them. Active by default and placed after fusion (user decision 2026-09-23: the fused top
    # five must not change), so an unset environment cannot silently switch the stage off.
    HostedVariant(
        "C9_routed_specialists_grounded_graph_atomic",
        True,
        True,
        False,
        False,
        False,
        None,
        anchor_compiler=True,
        anchor_compiler_version=3,
        drop_compiler_fallback=True,
        graph_sidecar=True,
        multimodal_preserve=True,
        multimodal_native=True,
        embedding_profile="voyage-code-4-v1",
        canonical_bm25=True,
        word_window_size=160,
        word_window_stride=120,
        exact_dense=True,
        stable_window_order=True,
        content_only_windows=True,
        dated_search_content=True,
        # T-1, owner decision 2026-09-26: on for the second Textual Full (BEAM and LongMemEval-S
        # inside noise), off on the code route. LoCoMo temporal: +14.7 points measured on every
        # route, +12.2 [+6.9, +17.5] as served behind this gate (recomputed from the stored
        # labels 2026-09-27, audit cca789b STAKES-001). Real Textual traffic routes to code far
        # more than LoCoMo does, so its served reach there is smaller and unmeasured.
        resolved_relative_times=True,
        anchor_prior_records="without-ids",
        anchor_compile_max_payload_chars=150_000,
        # CO-1, owner decision 2026-09-27 ("add if positive"): ask the compiler only for what
        # survived in 76% of full records anyway (kind and cited anchors). Coding collect: MRR
        # 0.8627 unchanged, source recall 34 of 34, Add median 4.4 s against 11.0 s. Textual
        # (LoCoMo, 1,535 questions) check passed its rule: S - F +0.20 turn_hit@10, +0.13
        # turn_hit@20, +0.26 session_hit@10, ingest time 0.48 of full (recall-lab d92273f).
        anchor_compile_output="select",
        anchor_prior_records_max_chars=145_000,
        compile_resend_truncated=False,
        # Owner decision 2026-09-28: stop (retryably) rather than store Adds raw-only.
        stop_on_credit_exhausted=True,
        context_specialist=True,
        atomic_rescue=True,
        atomic_views_at_add=True,
        atomic_rescue_default_mode="active",
        atomic_rescue_default_placement="fused",
    ),
)
CLEAN_RERANK_VARIANTS = (
    HostedVariant("B0_raw", True, False, False, False, False, None),
    HostedVariant("B1_raw_rerank", True, False, False, True, False, None),
)
CODE_AWARE_VARIANTS = (
    HostedVariant("M0_raw", True, False, False, False, False, None),
    HostedVariant(
        "M1_code_neighbors",
        True,
        False,
        False,
        False,
        False,
        None,
        code_aware=True,
    ),
)
ANCHOR_COMPILER_VARIANTS = (
    HostedVariant("V2_raw", True, False, False, False, False, None),
    HostedVariant(
        "V2_anchor_raw",
        True,
        True,
        False,
        False,
        False,
        None,
        anchor_compiler=True,
    ),
    HostedVariant("V3_raw", True, False, False, False, False, None),
    HostedVariant(
        "V3_anchor_raw",
        True,
        True,
        False,
        False,
        False,
        None,
        anchor_compiler=True,
        anchor_compiler_version=3,
        raw_rescue_tail=True,
    ),
)
MULTIVIEW_RETRIEVAL_VARIANTS = (
    HostedVariant("M0_multiview_raw", True, False, False, False, False, None),
    HostedVariant(
        "M2_repository_raw",
        True,
        True,
        False,
        False,
        False,
        None,
        anchor_compiler=True,
        compiled_kinds=REPOSITORY_KINDS,
        drop_compiler_fallback=True,
    ),
    HostedVariant(
        "M3_experience_raw",
        True,
        True,
        False,
        False,
        False,
        None,
        anchor_compiler=True,
        compiled_kinds=EXPERIENCE_KINDS,
        drop_compiler_fallback=True,
    ),
)
MULTIMODAL_VARIANTS = (
    HostedVariant("MM0_caption", True, False, False, False, False, None),
    HostedVariant(
        "MM1_preserve",
        True,
        False,
        False,
        False,
        False,
        None,
        multimodal_preserve=True,
    ),
    HostedVariant(
        "MM2_dual",
        True,
        False,
        False,
        False,
        False,
        None,
        multimodal_preserve=True,
        multimodal_native=True,
    ),
)
GRAPH_VARIANTS = (
    HostedVariant("G0_raw", True, False, False, False, False, None),
    HostedVariant(
        "G1_grounded_graph",
        True,
        True,
        False,
        False,
        False,
        None,
        anchor_compiler=True,
        anchor_compiler_version=3,
        drop_compiler_fallback=True,
        graph_sidecar=True,
    ),
)
VARIANTS = (
    ATTRIBUTION_VARIANTS
    + EXPERIENCE_VARIANTS
    + CODING_MATRIX_VARIANTS
    + CODE4_OFFICIAL_VARIANTS
    + SPECIALIST_VARIANTS
    + CLEAN_RERANK_VARIANTS
    + CODE_AWARE_VARIANTS
    + ANCHOR_COMPILER_VARIANTS
    + MULTIVIEW_RETRIEVAL_VARIANTS
    + MULTIMODAL_VARIANTS
    + GRAPH_VARIANTS
)
DEFAULT_VARIANT = "A4_pack_7000"


def variant(name: str) -> HostedVariant:
    for item in VARIANTS:
        if item.name == name:
            return item
    raise ValueError(f"unknown AML Hosted variant {name!r}")
