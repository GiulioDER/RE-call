"""Chunk storage."""


class Store:
    def __init__(self):
        self._rows = []

    def add(self, row):
        self._rows.append(row)
