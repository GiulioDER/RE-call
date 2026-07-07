# Datastore choice

We chose PostgreSQL over a document store for the primary datastore because our access patterns are
relational and we need transactions across the orders and inventory tables. We revisit this yearly;
the last review reaffirmed it. Status: adopted.
