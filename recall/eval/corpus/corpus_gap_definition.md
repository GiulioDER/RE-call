# Corpus gap

A corpus gap is when the memory store lacks any strongly relevant answer to a query. We detect it
when the best candidate similarity falls below a cosine threshold of about one half, and tell the
caller to treat the hits as unreliable rather than answering confidently from noise.
