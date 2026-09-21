# C8 scope and fingerprint artifact layout

## Supersedes

This record supersedes the registry layout statement in
`2026-09-21-aml-c8-scope-bound-atomic-qualification.md`. Its core prediction about exact opaque
scope binding remains unchanged.

## Question

Can C8 select an optional atomic artifact only by the exact opaque serving scope, generation
identity, and corpus fingerprint, with the same fingerprint also independently validated from the
immutable manifest?

## Frozen apparatus

The registry layout is
`<artifact-root>/<opaque-scope>/<generation-id>/<corpus-fingerprint>/manifest.json`. The service
receives the scope and fingerprint from its derived corpus status, never from an artifact request.
All three values must be one safe registry component. The fingerprint must be a lower case SHA256
hexadecimal digest.

## Predictions and pass criteria

1. Two frozen corpus fingerprints under one opaque scope and generation identity resolve to two
   distinct manifest paths and do not share a process cache entry.
2. A valid manifest from another fingerprint path fails closed before artifact loading.
3. A manifest copied to the requested path but carrying another fingerprint still fails its lineage
   validation.
4. Invalid scope, generation, or fingerprint path components are refused before a file is read.

## Interpretation

Passing establishes an unambiguous scope, generation, and corpus artifact selection contract. It
does not authorize an official AML Smoke.
