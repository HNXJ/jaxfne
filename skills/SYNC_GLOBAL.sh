#!/usr/bin/env bash
#
# Synchronize canonical repository skills to user-level mirrors.
# Default: read-only drift check.
# Apply: bash skills/SYNC_GLOBAL.sh --apply
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/skills"

FOLDERS=(
  catalog-glossary-jaxfne
  jaxfne-config
  jaxfne-harden
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

MODE="check"
case "${1:-}" in
  "") ;;
  --check) MODE="check" ;;
  --apply) MODE="apply" ;;
  --help|-h)
    printf '%s\n' \
      "Usage: bash skills/SYNC_GLOBAL.sh [--check|--apply]" \
      "  --check  report mirror drift without writing (default)" \
      "  --apply  copy repository skills to named mirrors"
    exit 0
    ;;
  *)
    printf 'unknown option: %s\n' "$1" >&2
    exit 2
    ;;
esac

if ! command -v rsync >/dev/null 2>&1; then
  printf '%s\n' "rsync is required for skill mirror checks" >&2
  exit 2
fi

printf 'canonical_source=%s\n' "${SRC}"
printf 'mode=%s\n' "${MODE}"

status=0
for dest in "${DESTS[@]}"; do
  if [[ "${MODE}" == "apply" && -d "${dest}/.git" ]]; then
    mirror_status="$(git -C "${dest}" status --porcelain)"
    if [[ -n "${mirror_status}" ]]; then
      printf 'refusing_dirty_mirror=%s\n' "${dest}" >&2
      status=1
      continue
    fi
  fi
  for folder in "${FOLDERS[@]}"; do
    source="${SRC}/${folder}/"
    target="${dest}/${folder}/"
    if [[ ! -d "${source}" ]]; then
      printf 'missing_source=%s\n' "${folder}" >&2
      status=1
      continue
    fi
    if [[ "${MODE}" == "apply" ]]; then
      mkdir -p "${dest}/${folder}"
      rsync -a --delete "${source}" "${target}"
      printf 'synced=%s:%s\n' "${dest}" "${folder}"
    else
      if [[ ! -d "${target}" ]]; then
        printf 'drift=%s:%s (missing mirror)\n' "${dest}" "${folder}"
        status=1
        continue
      fi
      dry_run="$(rsync -ain --delete "${source}" "${target}")"
      if [[ -n "${dry_run}" ]]; then
        printf 'drift=%s:%s\n' "${dest}" "${folder}"
        status=1
      else
        printf 'ok=%s:%s\n' "${dest}" "${folder}"
      fi
    fi
  done
done

if [[ "${MODE}" == "check" && "${status}" -ne 0 ]]; then
  printf '%s\n' "mirror drift detected; review the dry-run output before --apply" >&2
fi
exit "${status}"
