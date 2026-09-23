# Pre-registration: RE-call's own Voyage HTTP client in place of the voyageai SDK

**Date:** 2026-09-23   **Status:** predicted, not yet measured

## The question

Does building Voyage embedders and the Voyage reranker on `recall._voyage_http.Client` instead of
`voyageai.Client` (a) return the same vectors and rerank results bit for bit, and (b) remove the
cost of importing the SDK, measured as wall time and peak resident memory from process start to a
constructed embedder?

## What I predict

1. **Parity.** For every one of the 7 cases in `scripts/measure_voyage_http_client.py parity`
   (voyage-code-3 untyped, document and query; voyage-context-4 document group and query;
   voyage-4 untyped; rerank-2.5), wherever the SDK agrees with itself bit for bit, the HTTP client
   agrees with the SDK bit for bit (`sdk_vs_http_bit_equal` true). I expect the SDK to agree with
   itself on all 7, so I predict 7 of 7 bit-equal.
2. **Cost, this workstation** (has `torch` and `transformers`, like VPS2). Over 5 fresh-process
   samples per arm, alternating:
   * the `http` arm loads none of `voyageai`, `torch`, `transformers` in any sample (0 of 5);
   * the `sdk` arm loads `torch` in every sample (5 of 5);
   * median wall time falls by at least 3 s and by at least 70%;
   * median peak memory falls by at least 250 MB.

   These are deliberately below the ceiling. VPS2 measured 9.97 s and 11.43 s and 778 MB for
   `import voyageai` alone on 2026-09-22, and an earlier session measured 45 s warm for the same
   import on this workstation inside pytest, but I have a documented habit of over-predicting
   benefits by two to four times, so I predict roughly a
   third to a half of what those numbers suggest.
3. **Cost, VPS3** (no `torch`, no `transformers`). Same command. The gain is small there, because
   the SDK's heavy imports are absent: median wall time falls by less than 1.5 s and median peak
   memory by less than 100 MB, and neither arm loads `torch`.

## What would falsify this

* Parity: any case where the SDK agrees with itself but not with the HTTP client, or a length
  mismatch. That would mean the client's request or decoding differs from the SDK's, and it must
  not ship.
* Cost, workstation: the `http` arm loading `torch` or `voyageai` in any sample; or a median
  wall-time fall under 3 s or under 70%; or a peak memory fall under 250 MB.
* Cost, VPS3: a fall above either bound would mean something other than torch dominated the SDK's
  cost there, and my model of where the cost lives is wrong.

## How it will be measured

* Parity, on VPS3 from a session-owned copy of this branch, with the testbench's
  `VOYAGE_API_KEY` (21 small API calls):
  `python scripts/measure_voyage_http_client.py parity`
  Metric: `sdk_vs_http_bit_equal` per case, conditioned on `sdk_vs_sdk_bit_equal`.
* Cost, on this workstation and on VPS3:
  `python scripts/measure_voyage_http_client.py cost --samples 5`
  Metrics: `seconds_median` and `peak_mb_median` per arm (n = 5 each), their ranges, and
  `loaded_torch_in` / `loaded_voyageai_in` (counts out of 5). Peak memory is
  `PeakWorkingSetSize` on Windows and `ru_maxrss` on Linux, for the whole process.

Not measured here, and not claimed: the MCP server's end-to-end startup handshake, and VPS2
itself, which is off limits while the official AML run is in progress.

## What I already know

* VPS2, 2026-09-22, two samples of `import voyageai` alone: 11.43 s and 9.97 s, peak 778 MB in
  both, `torch` and `transformers` loaded (this session).
* `CLAUDE.md`, Testing: moving `voyageai` out of module scope in one test file took it from
  45.33 s to 1.04 s, because the import drags in `transformers` and `torch`.
* The client's request bodies, headers, base URL, decoding and error mapping already match the
  SDK against a fake transport (`tests/test_voyage_http_client.py`); this measurement is the live
  check that no server-side behaviour depends on anything the fake cannot see.

## Confounds I can name now

* Page cache: the first `torch` import of the day is much slower than later ones. Alternating arms
  and taking medians over 5 each limits this; a cold first sample would inflate the `sdk` arm's
  range, not its median.
* This workstation is shared with other sessions, so wall time is noisy. Memory is the more
  stable of the two metrics.
* Provider nondeterminism: if the SDK does not agree with itself on a case, bit parity cannot be
  asked of any client there, and that case is reported, not counted.
* The cost arms measure the constructor path only. A server imports more than this, so the
  saving in a real server process is at most the measured difference, not a multiple of it.

## Result (2026-09-23)
**Status:** measured

Commit measured: `4a2d815c` (this branch), shipped to a session-owned directory on VPS3 for the
parity and VPS3 cost runs.

**Apparatus correction, disclosed.** The first workstation cost run reported a peak of 0 MB for
every sample in both arms: the Windows `GetProcessMemoryInfo` call had no ctypes `argtypes` or
`restype`, so it failed silently. Its wall times were valid (median 44.14 s `sdk`, 1.38 s `http`,
n = 5 each) and are recorded here, but its memory numbers were not measurements. The call was
fixed in `4a2d815c`, verified against a known answer (touching a 200 MB bytearray moved the
reading from 16 MB to 216 MB), and the whole cost run was repeated. The numbers below are from
the repeat. The predictions above were not edited.

### 1. Parity (VPS3, live API, 21 calls)

| case | values | SDK vs SDK bit-equal | SDK vs HTTP bit-equal | SDK vs SDK max abs |
|---|---:|---|---|---:|
| embed voyage-code-3 untyped | 4096 | yes | yes | 0 |
| embed voyage-code-3 document | 4096 | yes | yes | 0 |
| embed voyage-code-3 query | 1024 | **no** | yes | 7.7e-4 |
| context-4 document group | 3072 | yes | yes | 0 |
| context-4 query | 1024 | yes | yes | 0 |
| embed voyage-4 untyped | 4096 | yes | yes | 0 |
| rerank-2.5 (index, score) | 12 | yes | yes | 0 |

Measured: 6 of 7 cases were SDK-self-consistent, and on all 6 the HTTP client was bit-equal to
the SDK (max abs 0.0). Predicted: 7 of 7 self-consistent and 7 of 7 bit-equal.
**Gap:** the parity claim holds on every case it can be asked of (6 of 6). The prediction that the
SDK agrees with itself everywhere is falsified: two SDK calls with the same voyage-code-3 query
returned vectors differing by up to 7.7e-4, so the provider is not deterministic for that input,
and per the rule above the case is reported, not counted. (The HTTP call happened to be
bit-equal to the first SDK call there; that is one sample and proves nothing about it.)

### 2. Cost, this workstation (n = 5 per arm, alternating)

| arm | wall time median (range) | peak memory median (range) | `torch` loaded | `voyageai` loaded |
|---|---|---|---|---|
| `sdk` | 46.18 s (43.24 to 50.52) | 558.8 MB (558.5 to 559.1) | 5 of 5 | 5 of 5 |
| `http` | 1.29 s (1.22 to 1.45) | 52.3 MB (52.0 to 52.8) | 0 of 5 | 0 of 5 |

Measured: wall time down 44.9 s (97%), peak memory down 506.5 MB. Predicted: down at least 3 s
and at least 70%, and at least 250 MB; `http` loading neither in any sample; `sdk` loading
`torch` in every sample.
**Gap:** every prediction held, and the magnitudes were under-predicted by roughly 15x for time
and 2x for memory. That is the opposite direction to my usual error, and the reason is worth
keeping: I discounted from the ceiling as I do for an intervention's efficacy, but this effect is
structural (a module is either imported or not), so the ceiling was the answer.

### 3. Cost, VPS3 (no `torch`, no `transformers`; n = 5 per arm)

| arm | wall time median (range) | peak memory median (range) | `torch` loaded |
|---|---|---|---|
| `sdk` | 1.02 s (0.99 to 1.10) | 95.1 MB (95.0 to 95.1) | 0 of 5 |
| `http` | 0.47 s (0.43 to 0.49) | 47.4 MB (47.4 to 48.2) | 0 of 5 |

Measured: down 0.55 s and 47.7 MB. Predicted: down less than 1.5 s and less than 100 MB, neither
arm loading `torch`. **Gap:** held. Where the heavy libraries are absent the SDK still costs about
half a second and 48 MB, which is `voyageai` itself plus `langchain_text_splitters`.

Not measured, as stated above: VPS2 and the MCP server's end-to-end handshake. VPS2 has `torch`
and `transformers` installed, as this workstation does, so the workstation result is the relevant
one for it, but that is an inference until measured there.
