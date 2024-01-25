#!/bin/bash

pytest -s --tb=short \
    --cov-report term-missing --cov-report html:htmlcov/tests \
    --cov=src/groundcrew \
    --cov=src/groundcrew/llm \
    --cov=tests \
    tests

if [ "$(uname)" == "Darwin" ]; then
    open htmlcov/tests/index.html
fi
