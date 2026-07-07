# Incident: checkout 500s (June)

On the June deploy the checkout service returned 500s for roughly eight minutes. Root cause was a
database connection pool exhausted by a slow migration holding a table lock. Mitigation: run
migrations in a maintenance window and add a statement timeout. Follow-up: pool saturation alerting.
