# API rate limits

Each client key is limited to 100 requests per second. Requests beyond the limit receive
HTTP 429 with a Retry-After header. Status: adopted.
