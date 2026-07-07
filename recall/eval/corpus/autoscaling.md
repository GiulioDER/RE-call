# Autoscaling

Production runs three application replicas behind a load balancer with readiness health checks.
Autoscaling adds a replica when CPU crosses seventy percent for five minutes.
