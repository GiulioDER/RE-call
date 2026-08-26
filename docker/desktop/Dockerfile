FROM python:3.14-slim

WORKDIR /opt/recall
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-calc \
        libreoffice-impress \
        libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*
COPY . /opt/recall
RUN pip install --no-cache-dir ".[mcp,fastembed,documents]"

# Everything below drops the runtime out of root.
#
# This image installs LibreOffice, and LibreOffice is where UNTRUSTED uploaded documents are
# parsed: `recall_ingest` accepts a file from a client, `stage_uploads` writes it under
# /opt/recall/uploads, and `recall.extraction._run_libreoffice` hands it to `soffice`. That is a
# large C++ parser surface reached by attacker-controlled bytes, so the question is not whether it
# will ever have a bug but what a bug gets. As root it got uid 0 inside the container, which is
# the starting position for every container-escape technique that needs privileged operations.
#
# A FIXED uid/gid, not a floating one. The uploads path below is a named volume shared by all four
# services in docker-compose.desktop.yml, so its ownership has to mean the same thing in every
# container that mounts it, and an implicit uid assigned by useradd would drift the day a base
# image changes its numbering. 10001 is above the distro range and below the 65534 nobody uses.
RUN groupadd --system --gid 10001 recall \
    && useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/recall recall

# Created HERE, in the image, and owned by the runtime user. Docker seeds a NEW named volume from
# the image's content and ownership at the mount path, so creating it now is what makes the shared
# `recall_desktop_uploads` volume land as recall:recall instead of root:root.
#
# ⛔ An EXISTING volume keeps the ownership it already has. A stack that ran a previous version of
# this image has a root-owned `recall_desktop_uploads`, and after `docker compose pull` the ingest
# path will fail with EACCES on mkdir rather than anything that names this change. That volume
# holds staging trees only, and in this stack's configuration (RECALL_ENV=development, no control
# plane, so legacy mode) `discard_staging` removes each one once its ingest completes, so it is
# safe to reset. One time, with the stack down:
#
#     docker volume rm recall_desktop_uploads
#
# The application code stays root-owned and is NOT chowned: the runtime user has no business
# rewriting the package it executes, and read+execute is all it needs.
RUN install -d -o recall -g recall -m 0755 /opt/recall/uploads

# `soffice` needs a writable HOME to bootstrap a user profile before -env:UserInstallation
# redirects it, and fastembed unpacks its ONNX models into `fastembed_cache` under the system temp
# directory. /tmp is world-writable, so only HOME has to be stated.
ENV HOME=/home/recall

USER recall
