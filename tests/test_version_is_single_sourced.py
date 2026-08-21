"""`pyproject.toml` and `recall.__version__` must agree, because two things read different ones.

⛔ **This is not bookkeeping tidiness.** The two values feed different machinery:

* `pyproject.toml` builds the wheel, so it decides what version exists on PyPI.
* `recall.__version__` is what `recall/wizard/stack.py` pins into the Dockerfile it generates
  (`recall-rag[...]=={version}`) and what `_default_image` scopes the image tag to.

So a drift does not produce a cosmetic mismatch. It produces an installer that provisions a
container pinned to a version that either does not exist on PyPI — a stack that fails to build,
during somebody's first install — or exists and is a DIFFERENT recall than the one building the
generations, which is the silent version skew `dockerfile_text`'s own docstring says the pin exists
to prevent.

Found while releasing 0.9.7: bumping `pyproject.toml` left `__init__.py` at 0.9.6, and nothing in
the suite would have said so.
"""

from __future__ import annotations

import pathlib
import tomllib

import recall


def _declared() -> str:
    root = pathlib.Path(__file__).resolve().parent.parent
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(document["project"]["version"])


def test_the_package_and_the_project_declare_the_same_version() -> None:
    assert recall.__version__ == _declared(), (
        f"recall.__version__ is {recall.__version__} and pyproject.toml says {_declared()}. The "
        "wheel is built from pyproject, and the installer pins recall.__version__ into the "
        "Dockerfile it writes, so this drift ships a stack that installs the wrong recall or none."
    )


def test_the_generated_dockerfile_pins_the_version_that_will_be_published() -> None:
    """The consequence, asserted rather than described.

    `dockerfile_text` takes the version from `recall.__version__`. If that is not the version
    `pyproject.toml` publishes, the generated stack pins something PyPI may not have.
    """
    from recall.wizard.stack import dockerfile_text

    assert "recall-rag[" in dockerfile_text()
    assert f"=={_declared()}" in dockerfile_text(), (
        "the Dockerfile the installer writes must pin the version this project publishes"
    )
