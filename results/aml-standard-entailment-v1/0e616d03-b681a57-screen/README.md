# Artifacts

The three raw screen outputs are stored gzip-compressed (`gzip -n -9`, no embedded name or
timestamp) to keep the pull request diff reviewable. `SHA256SUMS` records the digests of the
original, uncompressed files exactly as measured on 2026-09-19. Verify with:

```bash
gunzip -k *.json.gz && sha256sum -c SHA256SUMS
```
