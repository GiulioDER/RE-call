"""Compiled graph: an opt-in, LLM-built evidence graph over a tenant's sources.

A model reads each source as numbered anchors and proposes records that cite them. The server keeps
only text copied verbatim from the cited anchors and resolves every edge itself, so the model never
names a graph endpoint. These are pure functions: nothing here reads configuration or calls a model.
"""
