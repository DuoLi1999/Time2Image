#!/bin/bash

PYTHON_SCRIPT_PATH="transformation.py"

for i in {1..129}
do
    echo "Running with index: $i"
    python "$PYTHON_SCRIPT_PATH" --index "$i" --std 1
done