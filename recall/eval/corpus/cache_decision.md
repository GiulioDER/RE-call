# Read-through cache

We adopted a read-through cache in front of the pricing service, keyed by product id and
invalidated on every write, with a 5-minute TTL as a backstop. Median read latency dropped from
180ms to 22ms after rollout. Status: adopted, in production.
