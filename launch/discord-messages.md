# Discord messages

> Now that the LangChain + LlamaIndex adapters exist, you have a legitimate reason to post in these
> communities — you built something FOR their users. Find the right channel first (#showcase,
> #tools, #community-projects, or #integrations — NOT #general or #help). Post once, be around to
> answer. Frame it as "I built an integration + here's the interesting bit", never a cold ad.

---

## LangChain Discord (#showcase or #tools)

Hey folks — I shipped a LangChain retriever for RE-call, an open-source memory-retrieval engine I've been building. `RecallRetriever` is a drop-in `BaseRetriever`, but with one twist that's the whole point: **when its trust layer can't find something it actually trusts, it returns an empty result instead of a best-effort neighbour** — so your chain gets nothing rather than a stale/superseded memory it'd cite with confidence.

```python
from recall.integrations.langchain import RecallRetriever
retriever = RecallRetriever.from_store(store, embedder, k=5)
docs = retriever.invoke("...")   # verdict + confidence + provenance in each Document's metadata
```

The reason it abstains: I benchmarked it on LOCOMO's adversarial split (the 22.5% that most memory benchmarks drop — the "does it know when it doesn't know" part) and published the whole trade-off, including where it's still bad. Repo + write-up if useful: https://github.com/GiulioDER/RE-call

Happy to answer anything about the retriever or the eval.

---

## LlamaIndex Discord (#showcase or #integrations)

Hi all — just shipped a LlamaIndex retriever for RE-call (open-source memory retrieval). It's a normal `BaseRetriever` returning `NodeWithScore`s, with the trust signal (verdict/confidence/provenance) in each node's metadata — and it returns **no nodes** when its trust layer abstains, so a query engine synthesises from nothing rather than from a stale memory.

```python
from recall.integrations.llamaindex import RecallRetriever
retriever = RecallRetriever.from_store(store, embedder, k=5)
nodes = retriever.retrieve("...")
```

Background on why the abstention exists: I scored the LOCOMO adversarial questions (the part accuracy leaderboards skip) and published the full curve. Repo: https://github.com/GiulioDER/RE-call — glad to go into the eval details.
