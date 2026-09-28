#!/bin/bash
# Create the public GitHub repo and push this directory (run once).
#
# Prerequisites:
#   mamba run -n ghcli gh auth login    # or: export PATH=.../envs/ghcli/bin:$PATH
#   SSH key already works (git@github.com authenticates as priftisg)
#
# Usage (from this directory):
#   bash push_to_github.sh
set -euo pipefail
export PATH="${HOME}/conda/envs/ghcli/bin:/nas/rhome/gpriftis/conda/envs/ghcli/bin:${PATH}"

REPO_NAME="${REPO_NAME:-ascat-crps-paper-figures}"
OWNER="${OWNER:-priftisg}"
VISIBILITY="${VISIBILITY:-public}"

cd "$(dirname "$0")"

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not authenticated. Run:"
  echo "  mamba run -n ghcli gh auth login -h github.com -p ssh -w"
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote origin already set: $(git remote get-url origin)"
else
  gh repo create "${OWNER}/${REPO_NAME}" \
    --"${VISIBILITY}" \
    --source=. \
    --remote=origin \
    --description "ASCAT CRPS paper figures and regeneration scripts" \
    --push
  echo "Created and pushed https://github.com/${OWNER}/${REPO_NAME}"
  exit 0
fi

git push -u origin HEAD
echo "Pushed to $(git remote get-url origin)"
