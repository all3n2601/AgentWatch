#!/usr/bin/env bash
# The model pipeline end to end, in the image or in any Linux checkout with uv, make and libgomp.
# Every step is a repository make target; the first failure stops the run.
set -euo pipefail

RELEASE=data/datasets/steps-v1.3+aafae73eb211
SPLIT=data/datasets/splits/steps-v1.3+aafae73eb211/split-v3+932b2df410a7.json
# The evaluated split-v3 (docs/data-card.md); the rebuilt file must be byte-identical.
EXPECTED_SPLIT_SHA256=32488a942ad66f567be84e5992c2fe0bb14c92b6f67a4238000b85644f06d7c3

step() { printf '\n== %s\n' "$1"; }

step "Pull dataset v1 from the public DVC remote (no login)"
make pull-dataset TARGET=data/datasets

step "Rebuild split-v3 and check it is the evaluated split"
uv run --frozen --package agentwatch-data python -m agentwatch_data.splits --release "$RELEASE"
echo "$EXPECTED_SPLIT_SHA256  $SPLIT" | sha256sum --check --strict

step "Train and evaluate on test"
make train-candidates SPLIT="$SPLIT"

step "Calibrate on validation, score on test"
make calibrate SPLIT="$SPLIT"

step "Bias check"
make bias-check SPLIT="$SPLIT"

step "Figures"
make model-analysis SPLIT="$SPLIT"

step "Push to the model registry"
make register-model
