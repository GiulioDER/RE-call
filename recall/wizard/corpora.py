"""The three tenants the wizard configures, and the couplings that make them coherent.

The design describes this as a table with independent columns. They are not independent, and the
dependencies are the whole reason this module exists rather than a dict literal in the pipeline:

* **`RECALL_ENV=production` is the only switch that routes search through `GenerationStore`**, so it
  is the only way a calibrated corpus is actually served as calibrated. Verified: `cli.py` selects
  that store on this switch in three places and `recall_mcp/server.py` in one, and nothing else does.
* **`recall index` and `recall setup`'s auto-index refuse under that switch, but `Indexer` itself
  does not.** So calibrated and writable are mutually exclusive for anyone going through the CLI,
  and the wizard has to impose the same rule on ITSELF, because it drives `Indexer` in process and
  nothing would stop it. An earlier version of this note said "the same switch refuses local
  filesystem indexing", which is true of the command and false of the code this module calls.
* **An uncalibrated corpus cannot be served under production at all.** There is no generation, so
  `GenerationStore.snapshot` raises `NoActiveGeneration` from OUTSIDE `trusted_search`'s try block:
  the caller gets a raw exception rather than a refusal, with no failure code, no advice and no
  trust state. That is worse than any of the refusals below, which is why it is refused here.
* **An uncalibrated corpus cannot be served strictly**, and the refusal it produces is
  `INDEX_NOT_READY`, not `CALIBRATION_MISSING`. `trust.py` overwrites the calibration verdict
  whenever the binding carries no `generation_id`, and a legacy-`chunks` tenant has no generation
  binding at all, so the calibration verdict is outranked before it is ever reported. An earlier
  version of this module named the wrong code, in the one direction guaranteed to mislead: the
  operator would have gone looking at the calibration.

Written as constructor invariants rather than as prose, because every one of them otherwise surfaces
several steps later as a refusal about something else: a promoted generation that will not calibrate,
a tenant that answers `INDEX_NOT_READY` while its rows are visibly in the table, or an uncaught
`NoActiveGeneration`.

**The embedder is not a per-corpus field.** All three tenants share one database and one
`chunks`-family schema, and the vector dimension is welded to the table, so two embedders across
corpora is not a configuration to validate — it is one the API should be unable to express.
`CorpusPlan` therefore takes a single embedder for the whole plan.

Nothing here touches a database or reads an environment variable. It is the answer to "what am I
about to build", which the pipeline then executes and the resumable state records.

⚠️ **One exception, stated because an earlier version of this paragraph claimed purity it did not
have:** `CorpusSpec.__post_init__` stats `root`. Three auditors caught the contradiction
independently. The stat is worth its cost, since a root that is a file should not survive
construction, but it has two consequences a reader should not have to discover. On a UNC path it
resolves through the network redirector, so a dead share can block construction with no timeout,
and `recall.manifest` treats UNC roots as a supported shape. And it is a snapshot, not an
invariant: the directory can vanish between constructing the spec and the pipeline reading it, so
the build step must re-check rather than trust this.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from recall.generation_build import BuildRequest
from recall.index import ChunkerKind
from recall.lint import DEFAULT_GLOB
from recall.trust_policy import TrustMode

#: The environments `GenerationManager` accepts. Mirrored as a `Literal` rather than imported,
#: because `generations.py` validates against a bare set literal with no name to import. Read with
#: `get_args` at the one place that checks it, rather than repeated as a second tuple beside it:
#: that is the shape that let a widened `Literal` and a stale tuple disagree, and it is the mistake
#: this branch already fixed once for `ChunkerKind`.
ServingEnvironment = Literal["development", "test", "production"]

#: Why a corpus root must be absolute. One constant rather than the same sentence written out in
#: two modules: `load_config` names the config KEY at fault and `CorpusSpec` names the PATH, which
#: is a deliberate duplication of the CHECK, but the explanation behind it must not be able to
#: drift into two different reasons for one rule.
RELATIVE_ROOT = (
    "a relative root resolves against the wizard's own working directory, so the commit stamped "
    "on every chunk would be the wizard's rather than the corpus's"
)

#: Python is the code corpus's default. Not `DEFAULT_GLOB`, which is markdown: a code tenant globbed
#: for markdown would build, promote and calibrate happily over the repository's documentation and
#: report success, which is worse than failing.
CODE_GLOB = "**/*.py"

__all__ = [
    "CODE_GLOB",
    "CorpusPlan",
    "CorpusSpec",
    "ServingEnvironment",
    "code_corpus",
    "default_plan",
    "docs_corpus",
    "memory_corpus",
]


@dataclass(frozen=True)
class CorpusSpec:
    """One tenant: where its content is, how it is chunked, and how it is served."""

    tenant: str
    root: Path
    glob: str
    chunker: ChunkerKind
    #: Built as an immutable generation and bound to a published calibration, rather than indexed
    #: into the legacy `chunks` table.
    calibrated: bool
    serving_environment: ServingEnvironment
    trust_mode: TrustMode
    #: Accepts writes after install, which only the legacy indexing path allows.
    writable: bool

    def __post_init__(self) -> None:
        if not self.tenant.strip():
            raise ValueError("tenant must be non-empty")
        if self.serving_environment not in get_args(ServingEnvironment):
            raise ValueError(
                f"serving_environment must be one of {get_args(ServingEnvironment)}, "
                f"not {self.serving_environment!r}"
            )
        # Checked here, and not left to `BuildRequest`. Both are runtime-erased `Literal`s, but the
        # uncalibrated tenant never builds a `BuildRequest` at all, so on that path nothing would
        # ever validate the chunker and a typo would pick the text chunker in silence.
        if self.chunker not in get_args(ChunkerKind):
            raise ValueError(
                f"chunker must be one of {get_args(ChunkerKind)}, not {self.chunker!r}"
            )
        # COERCED, not merely checked, and this is the third erased-type field in a row for the
        # same reason. `TrustMode` is a `StrEnum`, so the plain string "strict" compares EQUAL to
        # `TrustMode.STRICT` while failing `is`. The uncalibrated-and-strict guard below used `is`,
        # so passing the string sailed past it and produced exactly the tenant that answers
        # INDEX_NOT_READY forever. Measured: that spec constructed with `type(trust_mode) is str`.
        # Coercing here means the guard cannot be bypassed by spelling, and every later reader gets
        # a real enum member rather than something that merely prints like one.
        try:
            object.__setattr__(self, "trust_mode", TrustMode(self.trust_mode))
        except ValueError:
            raise ValueError(
                f"trust_mode must be one of {[m.value for m in TrustMode]}, not "
                f"{self.trust_mode!r}"
            ) from None
        # Absolute, because `commit_root` is derived from it and `head_commit` shells out to
        # `git -C <path>`, which resolves a relative path against the WIZARD's working directory and
        # then walks upward. Measured: `docs_corpus(Path("docs"))` stamped this repository's own HEAD
        # onto what is nominally the user's corpus — exactly the false provenance `commit_root`
        # exists to prevent, reintroduced one layer up.
        #
        # ⚠️ This closes ONE vector, not the class, and an earlier version of this comment read as
        # though it closed the class. `git -C <root>` walks UPWARD to the nearest `.git`, so an
        # ABSOLUTE root that merely sits inside an unrelated checkout still records that
        # repository's HEAD. Measured: an absolute scratch directory inside this worktree was
        # accepted and stamped `e5d8435c`, this worktree's own commit. A scratch folder under a
        # monorepo, a vendored subtree, anything beneath `node_modules` does the same.
        #
        # It is not fixable by refusing non-repository roots, because the common case is exactly a
        # subdirectory of a repository: a `docs/` folder genuinely WAS produced by its repo's
        # commit, and that is the provenance you want. Distinguishing "this repo's docs" from "a
        # directory that happens to sit inside a checkout" needs the pipeline to ask whether the
        # root holds files that repository actually tracks, which is a step this layer does not
        # have. Recorded here so the residual is visible rather than implied to be handled.
        #
        # Existence is deliberately NOT required: the wizard creates the memory directory. A root
        # that is absolute and absent stamps no commit, which is an absent record rather than a false
        # one, and absent is safe.
        if not self.root.is_absolute():
            raise ValueError(f"root must be absolute, not {str(self.root)!r}: {RELATIVE_ROOT}")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(f"root {str(self.root)!r} exists and is not a directory")
        if self.calibrated and self.writable:
            raise ValueError(
                f"corpus {self.tenant!r} cannot be both calibrated and writable: "
                "RECALL_ENV=production routes search through GenerationStore and refuses local "
                "filesystem indexing with the same switch"
            )
        if self.calibrated and self.serving_environment != "production":
            raise ValueError(
                f"a calibrated corpus is served under production, not "
                f"{self.serving_environment!r}: it is the only environment that reads the "
                "generation store, so anywhere else the calibration is simply not consulted"
            )
        # Before the uncalibrated-in-production guard below, so a WRITABLE corpus gets the message
        # about writing rather than the one about generations. Both are reachable: writable implies
        # uncalibrated here, so without this ordering the guard below would absorb every case.
        if self.writable and self.serving_environment == "production":
            raise ValueError(
                f"corpus {self.tenant!r} is writable, and `recall index` refuses to run under "
                "production, so nothing could ever write to it"
            )
        if not self.calibrated and self.serving_environment == "production":
            raise ValueError(
                f"an uncalibrated corpus cannot be served under production: {self.tenant!r} has no "
                "generation, so GenerationStore raises NoActiveGeneration from outside "
                "trusted_search's try block. The caller gets a raw exception rather than a refusal, "
                "with no failure code and no advice, which is worse than any refusal."
            )
        # `==`, not `is`. Identity on a `StrEnum` was half the bypass above; the coercion is the
        # other half, and either alone would leave the guard spellable-past.
        if not self.calibrated and self.trust_mode == TrustMode.STRICT:
            raise ValueError(
                f"an uncalibrated corpus cannot be served strictly: {self.tenant!r} would refuse "
                "every query with INDEX_NOT_READY, which outranks the calibration verdict because "
                "a legacy-chunks tenant has no generation binding. Use development trust, and say "
                "so to the user rather than letting them infer it from a degraded answer."
            )
        # The converse, which was missing. Every guard above refused a corpus claiming MORE than it
        # could deliver; this one refuses a calibrated corpus claiming LESS. The module's own
        # docstring says a calibrated corpus is served strictly, and nothing enforced it: a
        # `calibrated=True` spec with development trust constructed happily and would have thrown
        # away the refusal guarantee that calibrating the corpus was the whole point of buying.
        if self.calibrated and self.trust_mode != TrustMode.STRICT:
            raise ValueError(
                f"a calibrated corpus is served strictly, so {self.tenant!r} cannot ask for "
                f"{self.trust_mode.value!r} trust: calibration exists to make a confident answer "
                "refusable, and development trust degrades instead of refusing"
            )

    def build_request(self, *, project: str | None = None) -> BuildRequest:
        """The `generation_build` request for this corpus.

        `commit_root` is this corpus's own root, not the process working directory. The CLI stamps
        `head_commit(".")`, which is right for an operator running a command inside the repository
        being built and wrong for a wizard indexing somebody else's: it would record the wizard's
        commit on every chunk of the user's corpus, which is provenance that is present,
        well-formed and false.
        """
        if not self.calibrated:
            raise ValueError(
                f"corpus {self.tenant!r} is not calibrated, so it has no generation to build: it "
                "is indexed into the legacy chunks table. Building one would produce a promoted "
                "generation that nothing can calibrate."
            )
        # No `max_chars` / `overlap` override. There was one, it had no caller and no test, and it
        # survived mutation: replacing the whole line with `return request` left every test green.
        # Exposing `max_chars` alone would also be wrong rather than merely unused, because the
        # effective overlap is clamped to `max_chars // 4`, so changing one silently changes the
        # recorded chunker identity and therefore the fingerprint a calibration binds to. When a
        # caller needs it, it takes both, with a test.
        return BuildRequest(
            chunker=self.chunker,
            project=project,
            commit_root=str(self.root),
        )


def docs_corpus(root: Path) -> CorpusSpec:
    """Markdown documentation, calibrated and read-only."""
    return CorpusSpec(
        tenant="docs",
        root=Path(root),
        glob=DEFAULT_GLOB,
        chunker="text",
        calibrated=True,
        serving_environment="production",
        trust_mode=TrustMode.STRICT,
        writable=False,
    )


def code_corpus(root: Path) -> CorpusSpec:
    """A source repository, calibrated and read-only, chunked as code rather than prose."""
    return CorpusSpec(
        tenant="code",
        root=Path(root),
        glob=CODE_GLOB,
        chunker="code",
        calibrated=True,
        serving_environment="production",
        trust_mode=TrustMode.STRICT,
        writable=False,
    )


def memory_corpus(root: Path) -> CorpusSpec:
    """The project's own memory directory: writable, and deliberately never calibrated.

    A fresh memory directory holds one file, so it cannot meet the 20-per-class certification floor
    and a calibrated memory tenant would answer `CALIBRATION_MISSING` forever. It is also the one
    corpus that must accept writes after install, which production mode refuses. Its server
    therefore carries `RECALL_TRUST_MODE=development`, and the wizard says so rather than letting
    the user infer it from a degraded answer.
    """
    return CorpusSpec(
        tenant="memory",
        root=Path(root),
        glob=DEFAULT_GLOB,
        chunker="text",
        calibrated=False,
        serving_environment="development",
        trust_mode=TrustMode.DEVELOPMENT,
        writable=True,
    )


@dataclass(frozen=True)
class CorpusPlan:
    """Every corpus this install will configure, under one embedder."""

    embedder: str
    corpora: tuple[CorpusSpec, ...]

    def __post_init__(self) -> None:
        if not self.embedder.strip():
            raise ValueError("embedder must be non-empty")
        # Coerced, because the annotation is not a runtime constraint. A list was accepted, and
        # `frozen=True` blocks rebinding rather than mutation, so `plan.corpora.append(...)` then
        # reinstated the duplicate tenant this class exists to refuse — after construction, past the
        # check below. It also cost the record its hashability.
        object.__setattr__(self, "corpora", tuple(self.corpora))
        if not all(isinstance(corpus, CorpusSpec) for corpus in self.corpora):
            # A `str` is iterable, so `corpora="docs"` survived the coercion above and failed three
            # lines down with `AttributeError: 'str' object has no attribute 'tenant'` — an error
            # that names neither the argument nor the mistake.
            raise TypeError(
                "corpora must be an iterable of CorpusSpec, not "
                f"{[type(c).__name__ for c in self.corpora]}"
            )
        if not self.corpora:
            raise ValueError("a plan needs at least one corpus; an empty install is not a no-op")
        seen: set[str] = set()
        for corpus in self.corpora:
            if corpus.tenant in seen:
                # 🔁 The reason here was WRONG, and wrong in a way that would have sent an operator
                # looking for deleted rows. It said re-indexing a tenant prunes sources absent from
                # the new manifest, so the second corpus deletes the first. Neither half holds:
                # `_prune_vanished` is scoped to the indexed ROOT, and its own docstring says the
                # scoping exists precisely so that "a corpus may be indexed in several roots;
                # pruning everything absent from THIS glob would delete the others' rows on every
                # run". It also prunes only what is gone from DISK, never what is missing from a
                # manifest.
                #
                # The real hazard is supersession, not deletion: one tenant has one
                # `active_generation_id`, so two calibrated corpora on one tenant produce two
                # generations of which only one can be active. Promoting the second does not delete
                # the first, it makes it unsearchable, and `rollback()` exists because the rows are
                # still there. Silently serving half the corpus you asked for is a quieter failure
                # than deletion, not a smaller one.
                raise ValueError(
                    f"tenant {corpus.tenant!r} appears twice; a tenant has one active generation, "
                    "so promoting the second corpus would leave the first indexed but unsearchable"
                )
            seen.add(corpus.tenant)

    @property
    def calibrated(self) -> tuple[CorpusSpec, ...]:
        """The corpora that go through the generation pipeline, in plan order."""
        return tuple(c for c in self.corpora if c.calibrated)


def default_plan(
    *,
    embedder: str,
    docs_root: Path,
    code_root: Path,
    memory_root: Path,
) -> CorpusPlan:
    """The three-tenant layout the design specifies, under one embedder."""
    return CorpusPlan(
        embedder=embedder,
        corpora=(
            docs_corpus(docs_root),
            code_corpus(code_root),
            memory_corpus(memory_root),
        ),
    )
