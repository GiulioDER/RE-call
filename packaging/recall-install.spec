# PyInstaller spec for the graphical installer.
#
# ⛔ **onedir, not onefile, and that is not a preference.** A onefile build unpacks its entire
# payload to a temporary directory on EVERY launch. This bundle carries PySide6 and the ONNX
# runtime, which is hundreds of megabytes, so onefile costs a long unexplained pause before the
# window appears — on an installer, whose entire job is to reassure somebody that something is
# happening. It also leaves the payload in `%TEMP%` for antivirus to scan each time.
#
# ⚠️ **`recall` is collected wholesale via `collect_submodules`, and that is load-bearing.** This
# codebase imports lazily almost everywhere: `from recall.wizard.headless import run_headless`
# inside a function, `from recall.calibration_v2 import ...` inside a method. PyInstaller's static
# analysis cannot see any of those, so a bundle built from the entry point's visible imports alone
# starts fine and dies with `ModuleNotFoundError` at the moment the user presses Install — the
# worst possible time and the hardest to diagnose from a shipped binary.
#
# The same applies to the data files of `fastembed` and `tokenizers`, which load model
# configuration from package resources rather than from code.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = (
    collect_submodules("recall")
    + collect_submodules("recall_mcp")
    + collect_submodules("psycopg")
    # psycopg's binary driver is imported by name at runtime, so nothing references it statically.
    + ["psycopg_binary", "psycopg_pool"]
)

datas = collect_data_files("recall") + collect_data_files("fastembed")

# ⚠️ Excluded deliberately, and MEASURED rather than guessed. `collect_submodules("recall")` above
# sweeps in the evaluation and benchmark modules, which drag sklearn, pyarrow, narwhals and friends
# behind them — none of which the installer ever runs. Verified by importing the actual install
# path (`wizard.queryset`, `wizard.headless`, `wizard.pipeline`, `wizard.uninstall`) and listing
# the heavy top-level packages it pulls in: **none**.
#
# The distinction matters because the wholesale collect is the right call for recall's own lazily
# imported modules, and the wrong call for their optional third-party dependencies. Keeping the
# first and dropping the second is what makes the bundle a download somebody finishes.
#
# ⛔ `onnxruntime`, `tokenizers` and `numpy` are NOT here: fastembed needs all three to embed
# anything, and an installer that cannot embed cannot install.
excludes = [
    "torch",
    "transformers",
    "sentence_transformers",
    "sklearn",
    "pyarrow",
    "narwhals",
    "datasets",
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "pytest",
]

a = Analysis(
    ["entry_install.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="recall-install",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # ⚠️ `console=False` would hide every traceback a failed install produces. This is a tool people
    # run when something is not working; a window that closes with no output is the one outcome
    # worse than an ugly console. The GUI still opens.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="recall-install",
)
