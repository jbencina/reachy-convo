#!/usr/bin/env bash
# Build the app wheel and install it on a Reachy Mini over the local network.
#
# Usage: scripts/deploy_to_robot.sh [robot-host]
#
# The default SSH password on a stock wireless Reachy Mini is "root".
# Run `ssh-copy-id pollen@reachy-mini.local` once to skip password prompts.
set -euo pipefail

ROBOT_HOST="${1:-${ROBOT_HOST:-reachy-mini.local}}"
ROBOT_USER="${ROBOT_USER:-pollen}"
APPS_VENV="/venvs/apps_venv"

cd "$(dirname "$0")/.."

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)"
if [[ -z "$VERSION" ]]; then
    echo "error: could not read version from pyproject.toml" >&2
    exit 1
fi
WHEEL="dist/reachy_mini_conversation_app-${VERSION}-py3-none-any.whl"

echo "==> Building wheel (version ${VERSION})"
uv build --wheel
if [[ ! -f "$WHEEL" ]]; then
    echo "error: expected wheel not found: $WHEEL" >&2
    exit 1
fi

WHEEL_NAME="$(basename "$WHEEL")"
echo "==> Copying ${WHEEL_NAME} to ${ROBOT_USER}@${ROBOT_HOST}:/tmp/"
scp "$WHEEL" "${ROBOT_USER}@${ROBOT_HOST}:/tmp/"

echo "==> Installing into ${APPS_VENV} on the robot"
# First install resolves any new dependencies; the forced no-deps reinstall
# guarantees the app code is replaced even when the version number is unchanged.
ssh "${ROBOT_USER}@${ROBOT_HOST}" "
    set -euo pipefail
    ${APPS_VENV}/bin/python -m pip install /tmp/${WHEEL_NAME}
    ${APPS_VENV}/bin/python -m pip install --force-reinstall --no-deps /tmp/${WHEEL_NAME}
    rm -f /tmp/${WHEEL_NAME}
"

cat <<EOF
==> Installed ${WHEEL_NAME} on ${ROBOT_HOST}.

Restart the app for the new code to take effect:
  - from the dashboard: http://${ROBOT_HOST}:8000
  - or via the REST API:
      curl -X POST http://${ROBOT_HOST}:8000/api/apps/stop-current-app
      curl -X POST http://${ROBOT_HOST}:8000/api/apps/start-app/reachy_mini_conversation_app

Follow logs with:
  ssh ${ROBOT_USER}@${ROBOT_HOST} sudo journalctl -u reachy-mini-daemon -f
EOF
