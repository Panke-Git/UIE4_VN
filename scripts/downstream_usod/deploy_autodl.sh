#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' 'Usage: deploy_autodl.sh --commit FULL_SHA --checkout NEW_OR_IDENTICAL_PATH [--origin URL]'
}

commit=""
checkout=""
origin="https://github.com/Panke-Git/UIE4_VN.git"
branch="feat/downstream-usod-tc-usod"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --commit|--checkout|--origin)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      case "$1" in
        --commit) commit="$2" ;;
        --checkout) checkout="$2" ;;
        --origin) origin="$2" ;;
      esac
      shift 2 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ "$commit" =~ ^[0-9a-f]{40}$ && -n "$checkout" ]] || { usage >&2; exit 2; }
case "$origin" in
  https://github.com/Panke-Git/UIE4_VN.git|git@github.com:Panke-Git/UIE4_VN.git) ;;
  *) printf '%s\n' 'Expected the published Panke-Git/UIE4_VN origin.' >&2; exit 2 ;;
esac
export GIT_TERMINAL_PROMPT=0

verify_checkout() {
  local path="$1" actual_root root head_branch head_sha actual_origin
  actual_root=$(git -C "$path" rev-parse --show-toplevel) || return 1
  root=$(cd "$path" && pwd -P) || return 1
  head_branch=$(git -C "$path" branch --show-current) || return 1
  head_sha=$(git -C "$path" rev-parse HEAD) || return 1
  actual_origin=$(git -C "$path" remote get-url origin) || return 1
  if [[ "$actual_root" != "$root" || "$head_branch" != "$branch" ||
        "$head_sha" != "$commit" || "$actual_origin" != "$origin" ]]; then
    printf 'Checkout differs from the published origin, branch or commit: %s\n' "$path" >&2
    return 1
  fi
  if [[ -n "$(git -C "$path" status --porcelain --untracked-files=normal)" ]]; then
    printf 'Checkout contains tracked or untracked changes: %s\n' "$path" >&2
    return 1
  fi
}

if [[ -e "$checkout" || -L "$checkout" ]]; then
  verify_checkout "$checkout"
else
  parent=$(dirname "$checkout")
  mkdir -p "$parent"
  stage=$(mktemp -d "$parent/.usod-deploy.XXXXXXXX")
  trap 'rmdir "$stage" 2>/dev/null || true' EXIT
  git clone --single-branch --no-tags --branch "$branch" -- "$origin" "$stage/repository"
  verify_checkout "$stage/repository"
  mv -T --no-clobber -- "$stage/repository" "$checkout"
  if [[ -e "$stage/repository" ]]; then
    printf 'Checkout path appeared during deployment. Staged clone is at %s\n' "$stage/repository" >&2
    exit 1
  fi
  verify_checkout "$checkout"
fi
printf 'Verified %s at %s in %s\n' "$branch" "$commit" "$checkout"
