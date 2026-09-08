#!/usr/bin/env bash
# Download the release-dist artifact produced by release_ci.yml for one exact commit.
#
# Fails loudly rather than falling back to a build: if no successful release_ci
# run exists for this SHA, the commit was never validated and must not be
# published. Requires GH_TOKEN with actions:read.
set -euo pipefail

SHA="${1:?usage: fetch_release_artifacts.sh <commit-sha> <outdir>}"
OUT="${2:?usage: fetch_release_artifacts.sh <commit-sha> <outdir>}"

echo "Resolving successful release_ci run for ${SHA}"
RUN_ID="$(gh run list \
  --workflow release_ci.yml \
  --commit "${SHA}" \
  --status success \
  --limit 1 \
  --json databaseId \
  --jq '.[0].databaseId // empty')"

if [ -z "${RUN_ID}" ]; then
  echo "FAIL: no successful release_ci run for commit ${SHA}." >&2
  echo "      Refusing to publish artifacts that no release gate produced." >&2
  exit 1
fi
echo "Using release_ci run ${RUN_ID}"

mkdir -p "${OUT}"
gh run download "${RUN_ID}" --name release-dist --dir "${OUT}"

if [ ! -f "${OUT}/RELEASE_MANIFEST.json" ]; then
  echo "FAIL: release-dist artifact contains no RELEASE_MANIFEST.json." >&2
  exit 1
fi

MANIFEST_SHA="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['source_sha'])" "${OUT}/RELEASE_MANIFEST.json")"
if [ "${MANIFEST_SHA}" != "${SHA}" ]; then
  echo "FAIL: manifest source_sha ${MANIFEST_SHA} != publishing commit ${SHA}." >&2
  exit 1
fi

echo "Retained artifacts for ${SHA}:"
ls -l "${OUT}"
