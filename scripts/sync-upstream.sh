#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 AMD. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dynamo AMD Fork — Upstream Sync Script
#
# Syncs the amd-additive branch with the upstream Dynamo repository.
# Our patches are additive-only (99.6% additions), so rebase should be
# conflict-free in most cases.
#
# Usage:
#   ./scripts/sync-upstream.sh              # Sync and rebase
#   ./scripts/sync-upstream.sh --dry-run    # Check without applying
#   ./scripts/sync-upstream.sh --check      # Verify patch additiveness

set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-origin}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
AMD_BRANCH="${AMD_BRANCH:-amd-additive}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

check_additiveness() {
    log_info "Checking patch additiveness..."
    local total_add total_del ratio
    total_add=$(git diff "${UPSTREAM_BRANCH}..${AMD_BRANCH}" | grep "^+[^+]" | wc -l)
    total_del=$(git diff "${UPSTREAM_BRANCH}..${AMD_BRANCH}" | grep "^-[^-]" | wc -l)

    if [ "$total_del" -eq 0 ]; then
        ratio="100.0"
    else
        ratio=$(echo "scale=1; $total_add * 100 / ($total_add + $total_del)" | bc)
    fi

    echo "  Additions: +${total_add}"
    echo "  Deletions: -${total_del}"
    echo "  Additive ratio: ${ratio}%"
    echo ""

    if [ "$total_del" -gt 50 ]; then
        log_warn "Deletions exceed threshold (50). Review changes for upstream compatibility."
        echo ""
        echo "Files with deletions:"
        git diff "${UPSTREAM_BRANCH}..${AMD_BRANCH}" --numstat | awk '$2 > 0 {print "  -" $2 " " $3}' | sort -rn | head -10
        return 1
    fi

    log_info "Patch series is ${ratio}% additive — safe for rebase"
    return 0
}

show_status() {
    log_info "Current state:"
    echo "  AMD branch: ${AMD_BRANCH}"
    echo "  Upstream:   ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"

    local commits_ahead commits_behind
    commits_ahead=$(git rev-list --count "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}..${AMD_BRANCH}" 2>/dev/null || echo "?")
    commits_behind=$(git rev-list --count "${AMD_BRANCH}..${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" 2>/dev/null || echo "?")
    echo "  AMD patches: ${commits_ahead} commits ahead"
    echo "  Upstream:    ${commits_behind} commits behind"
    echo ""

    echo "  AMD patch list:"
    git log --oneline "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}..${AMD_BRANCH}" 2>/dev/null | sed 's/^/    /'
    echo ""
}

do_sync() {
    log_info "Fetching upstream..."
    git fetch "${UPSTREAM_REMOTE}" "${UPSTREAM_BRANCH}"
    echo ""

    show_status

    local behind
    behind=$(git rev-list --count "${AMD_BRANCH}..${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" 2>/dev/null || echo "0")

    if [ "$behind" -eq 0 ]; then
        log_info "Already up to date with upstream."
        return 0
    fi

    log_info "Upstream has ${behind} new commits. Rebasing..."
    echo ""

    local current_branch
    current_branch=$(git branch --show-current)

    if [ "$current_branch" != "$AMD_BRANCH" ]; then
        git checkout "$AMD_BRANCH"
    fi

    if git rebase "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"; then
        log_info "Rebase successful — no conflicts!"
        echo ""
        show_status
        check_additiveness
    else
        log_error "Rebase has conflicts."
        echo ""
        echo "Conflicting files:"
        git diff --name-only --diff-filter=U 2>/dev/null | sed 's/^/  /'
        echo ""
        echo "Options:"
        echo "  1. Resolve conflicts manually, then: git rebase --continue"
        echo "  2. Abort rebase: git rebase --abort"
        echo "  3. Skip conflicting commit: git rebase --skip"
        return 1
    fi
}

do_dry_run() {
    log_info "Dry run — checking without applying"
    echo ""

    git fetch "${UPSTREAM_REMOTE}" "${UPSTREAM_BRANCH}"
    show_status

    local behind
    behind=$(git rev-list --count "${AMD_BRANCH}..${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" 2>/dev/null || echo "0")

    if [ "$behind" -eq 0 ]; then
        log_info "Already up to date."
        return 0
    fi

    log_info "Would rebase ${behind} upstream commits."
    echo ""
    echo "New upstream commits:"
    git log --oneline "${AMD_BRANCH}..${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}" | head -20 | sed 's/^/    /'
    echo ""

    log_info "Checking for potential conflicts..."
    local amd_files upstream_files overlap
    amd_files=$(git diff --name-only "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}..${AMD_BRANCH}")
    upstream_files=$(git diff --name-only "${AMD_BRANCH}..${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}")
    overlap=$(comm -12 <(echo "$amd_files" | sort) <(echo "$upstream_files" | sort))

    if [ -z "$overlap" ]; then
        log_info "No overlapping files — rebase will be conflict-free."
    else
        log_warn "Overlapping files (potential conflicts):"
        echo "$overlap" | sed 's/^/  /'
    fi

    echo ""
    check_additiveness
}

case "${1:-}" in
    --dry-run)
        do_dry_run
        ;;
    --check)
        check_additiveness
        ;;
    --help|-h)
        echo "Usage: $0 [--dry-run|--check|--help]"
        echo ""
        echo "  (no args)   Fetch upstream and rebase AMD patches"
        echo "  --dry-run   Check for conflicts without applying"
        echo "  --check     Verify patch series is additive"
        echo ""
        echo "Environment variables:"
        echo "  UPSTREAM_REMOTE  Git remote name (default: origin)"
        echo "  UPSTREAM_BRANCH  Upstream branch (default: main)"
        echo "  AMD_BRANCH       AMD patches branch (default: amd-additive)"
        ;;
    "")
        do_sync
        ;;
    *)
        log_error "Unknown option: $1"
        exit 1
        ;;
esac
