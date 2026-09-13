FROM python@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

WORKDIR /app
COPY agent_guestbook_server.py /app/agent_guestbook_server.py

ENTRYPOINT ["python3", "/app/agent_guestbook_server.py"]
CMD ["--host", "0.0.0.0", "--port", "8789", "--database", "/data/guestbook.sqlite3"]
