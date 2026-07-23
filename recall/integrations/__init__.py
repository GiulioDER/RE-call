"""Optional framework integrations for RE-call.

Each submodule targets one host framework and imports that framework at module load, so it raises
a clear error when the corresponding extra is not installed. Importing ``recall.integrations``
itself pulls in nothing heavy — pick the submodule you need, e.g. ``recall.integrations.langchain``.
"""
