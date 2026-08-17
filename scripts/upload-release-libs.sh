#!/usr/bin/env bash
#
# Publish the prebuilt XGBoost libraries in xgboost-sys/lib as GitHub release assets.
#
# The asset names and checksums are read from xgboost-sys/build.rs, which is the single source of
# truth: the build script resolves release assets as <base>/<platform>-<file> (a release's asset
# namespace is flat), so an asset uploaded under any other name is invisible to it. Every file is
# checked against its pinned SHA-256 before upload and re-downloaded and checked again afterwards,
# so a release can never end up serving bytes the build script would reject.
#
# Usage:
#   scripts/upload-release-libs.sh [--dry-run] [tag]
#
# The tag defaults to LIB_TAG in build.rs. Requires the `gh` CLI, authenticated with a token that
# can write releases.

set -euo pipefail

DRY_RUN=false
TAG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h | --help)
            sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            echo "unknown option: $arg" >&2
            exit 2
            ;;
        *) TAG="$arg" ;;
    esac
done

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_RS="$REPO_ROOT/xgboost-sys/build.rs"
LIB_DIR="$REPO_ROOT/xgboost-sys/lib"

[ -f "$BUILD_RS" ] || {
    echo "not found: $BUILD_RS" >&2
    exit 1
}
[ -d "$LIB_DIR" ] || {
    echo "not found: $LIB_DIR (the prebuilt libraries are absent from the published crate)" >&2
    exit 1
}

if [ -z "$TAG" ]; then
    TAG=$(sed -n 's/^const LIB_TAG: &str = "\(.*\)";$/\1/p' "$BUILD_RS")
    [ -n "$TAG" ] || {
        echo "could not read LIB_TAG from $BUILD_RS" >&2
        exit 1
    }
fi

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# Emits "<platform> <file> <sha256>" per pinned artifact, in build.rs order.
read_manifest() {
    awk '
        /^[[:space:]]*"[a-z0-9_]+" => &\[/ {
            match($0, /"[a-z0-9_]+"/)
            platform = substr($0, RSTART + 1, RLENGTH - 2)
            next
        }
        /^[[:space:]]*file: "/   { match($0, /"[^"]+"/); file = substr($0, RSTART + 1, RLENGTH - 2); next }
        /^[[:space:]]*sha256: "/ { match($0, /"[^"]+"/); print platform, file, substr($0, RSTART + 1, RLENGTH - 2) }
    ' "$BUILD_RS"
}

MANIFEST=$(read_manifest)
[ -n "$MANIFEST" ] || {
    echo "no artifacts parsed from $BUILD_RS" >&2
    exit 1
}

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

echo "Release tag: $TAG"
echo "Staging assets from $LIB_DIR"

count=0
while read -r platform file sha; do
    src="$LIB_DIR/$platform/$file"
    if [ ! -f "$src" ]; then
        echo "  MISSING  $platform/$file" >&2
        exit 1
    fi
    actual=$(sha256_of "$src")
    if [ "$actual" != "$sha" ]; then
        echo "  MISMATCH $platform/$file" >&2
        echo "           on disk: $actual" >&2
        echo "           pinned:  $sha" >&2
        echo "Refusing to upload: build.rs would reject this file." >&2
        exit 1
    fi
    # Must match `mirror_url` in build.rs for release URLs.
    cp "$src" "$STAGE/$platform-$file"
    echo "  ok       $platform-$file"
    count=$((count + 1))
done <<<"$MANIFEST"

echo "$count assets staged."

if [ "$DRY_RUN" = true ]; then
    echo "Dry run, nothing uploaded. Would upload to release $TAG:"
    ls -1 "$STAGE"
    exit 0
fi

command -v gh >/dev/null 2>&1 || {
    echo "gh CLI not found" >&2
    exit 1
}

# Report what gh actually said. A missing release, an expired login and a remote pointing at a
# fork all fail here, and they need different fixes.
if ! gh_err=$(gh release view "$TAG" 2>&1 >/dev/null); then
    echo "cannot read release $TAG:" >&2
    echo "  $gh_err" >&2
    echo >&2
    echo "repo gh resolved: $(gh repo view --json nameWithOwner -q .nameWithOwner 2>&1 || echo '<unknown>')" >&2
    echo "auth status:" >&2
    gh auth status 2>&1 | sed 's/^/  /' >&2
    echo >&2
    echo "If the release genuinely does not exist: gh release create $TAG" >&2
    exit 1
fi

echo "Uploading to release $TAG"
# shellcheck disable=SC2046  # staged names are controlled above and contain no whitespace
gh release upload "$TAG" $(ls -d "$STAGE"/*) --clobber

echo "Verifying published assets"
BASE="https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/download/$TAG"
failed=0
while read -r platform file sha; do
    url="$BASE/$platform-$file"
    tmp="$STAGE/.verify"
    if ! curl -fsSL "$url" -o "$tmp"; then
        echo "  FAILED   $platform-$file (download)" >&2
        failed=1
        continue
    fi
    actual=$(sha256_of "$tmp")
    if [ "$actual" != "$sha" ]; then
        echo "  MISMATCH $platform-$file (served $actual)" >&2
        failed=1
    else
        echo "  verified $platform-$file"
    fi
done <<<"$MANIFEST"

[ "$failed" -eq 0 ] || {
    echo "Some assets are not serving the pinned bytes." >&2
    exit 1
}
echo "All $count assets published and verified against build.rs."
