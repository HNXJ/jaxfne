#!/usr/bin/env bash
# Sync repo-root skills/ (source of truth) to global agent skill directories.
# Run from repo root after merging skill changes: bash skills/SYNC_GLOBAL.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/skills"

FOLDERS=(
  catalog-glossary-jaxfne
  jaxfne-config
  jaxfne-modeling-optimization-schema
  jaxfne-neural-network
  jaxfne-neural-tensor
  jaxfne-notebook-release-gate
  jaxfne-objective-grammar
  jaxfne-paradigm-design
  jaxfne-release-mutation-guard
  jaxfne-sha256-artifact-integrity
  jaxfne-spectrolaminar-suite
  jaxfne-vis-modules
  jaxfne-worker-context-router
)

DESTS=(
  "${HOME}/.claude/skills"
  "${HOME}/.agents/skills"
)

STALE_EXTENSIONLESS=(
  jaxfne-objective-grammar
  jaxfne-cortical-column-default
  jaxfne-signals-probe-objective-chain
  jaxfne-configuration-fluent-api
)

# Folders merged away in the 2026-06-30 consolidation -- remove their old
# global copies outright (rsync --delete only cleans WITHIN a synced folder,
# it does not remove a destination folder that's no longer in FOLDERS above).
STALE_FOLDERS=(
  jaxfne-configuration-fluent-api
  jaxfne-cortical-column-default
  jaxfne-visualization-schema
  jaxfne-signals-probe-objective-chain
)

echo "Source: ${SRC}"
echo "Repo SHA: $(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"

for dest in "${DESTS[@]}"; do
  mkdir -p "${dest}"
  for stale in "${STALE_EXTENSIONLESS[@]}"; do
    if [[ -e "${dest}/${stale}" && ! -d "${dest}/${stale}" ]]; then
      rm -f "${dest}/${stale}"
      echo "removed stale extensionless ${dest}/${stale}"
    fi
  done
  for stale in "${STALE_FOLDERS[@]}"; do
    if [[ -d "${dest}/${stale}" ]]; then
      rm -rf "${dest}/${stale}"
      echo "removed merged-away folder ${dest}/${stale}"
    fi
  done
  for folder in "${FOLDERS[@]}"; do
    if [[ ! -d "${SRC}/${folder}" ]]; then
      echo "WARN: missing ${SRC}/${folder}" >&2
      continue
    fi
    rsync -a --delete "${SRC}/${folder}/" "${dest}/${folder}/"
    echo "synced ${folder} -> ${dest}/"
  done
done

echo "Done. Verify: ls ~/.claude/skills/ | grep jaxfne"
