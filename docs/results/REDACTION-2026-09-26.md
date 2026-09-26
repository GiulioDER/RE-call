# Redaction of private memory content from committed traces, 2026-09-26

**What happened.** Experiment traces and retrieval pools committed between 2026-08-25 and
2026-09-16 captured whole memos from the private memory store as retrieval-candidate text. That
text included infrastructure addresses, SSH host-key fingerprints, hosting server identifiers,
personal email addresses, and operational notes from another private project. The repository is
public, so all of it was published. A security review of pull request #776 found it on master.

**What changed.** `scripts/redact_memo_traces.py` rewrote the 60 affected JSON files:

- every memo content field (`text`, `payload`, `answer`, `answer_span`, `original_prompt`,
  `label_note`) now holds `[redacted 2026-09-26: private memo content, sha256 <prefix>, <n> chars]`;
- every path of the other project's memos is replaced by a stable pseudonym
  (`sentiment-agent/redacted-<hash>`), identical wherever the same path occurs, so joins between
  fields still hold;
- any remaining non-loopback IPv4 address, SSH fingerprint, hosting server ID or personal mailbox
  is replaced in place.

Ids, ranks, scores, labels, counts and structure are unchanged, so every reported number can still
be recomputed from these files. Two scripts and one test that carried a private address or an SSH
key name now take them from the environment.

**What it does to the records.** Five of the files are frozen pre-registration inputs (the memory
query set, its gold labels and three retrieval pools), and sixteen files have their SHA-256 cited
in pre-registrations and results. Those citations are left exactly as written: they describe the
files as they were when measured. The untouched originals are kept privately, outside this
repository, with a hash manifest, so the cited hashes can still be verified by their holder. The
redaction was the maintainer's explicit decision; the frozen-record rule was set aside for it and
for nothing else.

**What it does not do.** Git history still holds the original text; removing it needs a history
rewrite. Anything already fetched, cached or forked is outside this repository's reach, so every
value named above is treated as disclosed.
