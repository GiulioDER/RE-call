# Incident: stale feature flag

A stale feature flag left the new recommendations widget disabled for a subset of users for two
days. Root cause was a cache of flag values that was not invalidated on flag update. Fix: subscribe
the flag cache to the update stream.
