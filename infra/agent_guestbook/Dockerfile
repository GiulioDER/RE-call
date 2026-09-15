FROM python@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

RUN addgroup --system --gid 10001 recallguestbook \
    && adduser --system --uid 10001 --gid 10001 --no-create-home \
        --disabled-login --disabled-password recallguestbook \
    && mkdir /data \
    && chown 10001:10001 /data

WORKDIR /app
COPY --chown=10001:10001 agent_guestbook_server.py /app/agent_guestbook_server.py

USER 10001:10001
ENTRYPOINT ["python3", "/app/agent_guestbook_server.py"]
CMD ["--host", "0.0.0.0", "--port", "8789", "--database", "/data/guestbook.sqlite3"]
