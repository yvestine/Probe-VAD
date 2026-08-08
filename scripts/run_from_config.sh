#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "Usage: bash scripts/run_from_config.sh configs/<dataset>.env" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$1"
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Configuration file not found: ${CONFIG_PATH}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set +a

exec bash "${PROJECT_ROOT}/scripts/run_covas_vad_rebalanced.sh"

