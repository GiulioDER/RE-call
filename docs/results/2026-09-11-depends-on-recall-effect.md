# Authored `depends_on` recall effect

Measured 2026-09-11 after preregistration commit `1ab30aea`.

Preregistration: [2026-09-11-depends-on-recall-effect.md](../preregistrations/2026-09-11-depends-on-recall-effect.md).

Command:

```text
python -c "import pytest; raise SystemExit(pytest.main(['tests/test_semantic_graph.py','-k','depends_on_one_hop_improves_paired_recall_without_control_regression','-s','-q']))"
```

The test used 8 dependent queries and 8 unrelated controls in a deterministic in memory corpus.
The baseline kept the same trusted retrieval without graph expansion. The treatment used the real
`_expand_semantic_graph` one hop path with the new authored `depends_on` projection and a maximum
five item evidence budget.

| Population | Baseline hit@5 | Treatment hit@5 | Baseline MRR | Treatment MRR | Baseline precision@5 | Treatment precision@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dependent queries, n=8 | 0.000 | 1.000 | 0.000 | 0.200 | 0.000 | 0.200 |
| Controls, n=8 | 1.000 | 1.000 | 0.500 | 0.500 | 0.250 | 0.250 |

The dependent treatment activated 8 relations, discovered 8 candidates, accepted 8 candidates,
and added 8 new trusted evidence items. Each dependent query added exactly one candidate.

Each dependent treatment query returned a five item context. The control context did not change.
This proves the new typed edge can improve recall on a corpus that
contains exercised authored dependencies. It does not establish a production quality lift until
a rebuilt production generation contains dependency edges and the same paired evaluation is run.
