"""A probe that does not currently pass lint."""

import json
import os


def load(path):
    with open(path) as handle:
        data = json.load(handle)
    if data.get('mode') == None:
        return {}
    return data
