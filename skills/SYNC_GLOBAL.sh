#!/usr/bin/env bash
# Sync repo-root skills/ (source of truth) to global agent skill directories.
# Run from repo root after merging skill changes: bash skills/SYNC_GLOBAL.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/skills"

FOLDERS=(
  catalog-glossary-jaxfne
  jaxfne-configuration-fluent-api
  jaxfne-cortical-column-default
  jaxfne-modeling-optimization-schema
  jaxfne-notebook-release-gate
  jaxfne-objective-grammar
  jaxfne-paradigm-design
  jaxfne-release-mutation-guard
  jaxfne-sha256-artifact-integrity
  jaxfne-signals-probe-objective-chain
  jaxfne-spectrolaminar-suite
  jaxfne-visualization-schema
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
