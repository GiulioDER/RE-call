#!/bin/bash
# Entry point for CI to run the corpus chunk probe
# Prints the chunk count from the recall package

set -e

python -m benchmarks.probe
