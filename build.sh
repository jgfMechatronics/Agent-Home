#!/usr/bin/env bash
#
# Agent Home Docker Image Builder
#
# Builds agent-home:production or agent-home:experimental based on git state.
# - main branch + clean tree → production (requires 'y')
# - anything else → experimental (Enter to proceed, or type 'production' to override)
# - dirty working tree → requires 'y' confirmation
#
# Images are also tagged with short commit hash for traceability.

set -euo pipefail

# --- Colors ---
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# --- Setup ---
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$PWD"

echo -e "${CYAN}${BOLD}Agent Home Docker Builder${NC}"
echo "─────────────────────────────"
echo

# --- Check git repo ---
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}Error: Not a git repository${NC}"
    exit 1
fi

# --- Get git state ---
COMMIT_HASH=$(git rev-parse --short HEAD)
COMMIT_FULL=$(git rev-parse HEAD)
DIRTY=false; DETACHED=false

if BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null); then
    : # normal branch
else
    BRANCH="(detached HEAD at $COMMIT_HASH)"
    DETACHED=true
fi

git diff --quiet && git diff --cached --quiet || DIRTY=true

# --- Determine defaults ---
if [[ "$BRANCH" == "main" && "$DIRTY" == false ]]; then
    BASE_TAG="production"
else
    BASE_TAG="experimental"
fi

# --- Display build info ---
echo -e "Repository:  ${BOLD}$REPO_ROOT${NC}"
echo -e "Branch:      ${BOLD}$BRANCH${NC}"
echo -e "Commit:      ${BOLD}$COMMIT_HASH${NC} ($COMMIT_FULL)"
[[ "$DIRTY" == true ]] && echo -e "${YELLOW}⚠ Dirty working tree — uncommitted changes will be included${NC}"
[[ "$DETACHED" == true ]] && echo -e "${YELLOW}ℹ Detached HEAD${NC}"
echo
echo -e "Will build:  ${GREEN}${BOLD}agent-home:$BASE_TAG${NC}"
echo

# --- Confirmation prompt ---
if [[ "$BASE_TAG" == "production" ]]; then
    # Production build (on main, clean) — require explicit 'y'
    read -p "Build production image? (y/N): " -r REPLY
    [[ "$REPLY" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
elif [[ "$DIRTY" == true ]]; then
    # Dirty tree — require 'y', but allow 'production' override
    echo "Dirty working tree requires confirmation."
    read -p "Proceed? (y/N, or 'production' to build production): " -r REPLY
    if [[ "$REPLY" == "production" ]]; then
        BASE_TAG="production"
        echo -e "${YELLOW}Overriding to production build.${NC}"
    elif [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "Aborted."; exit 1
    fi
else
    # Experimental (non-main or detached, clean) — Enter to proceed, 'production' to override
    read -p "Proceed? (Enter = yes, 'production' = build production, n = abort): " -r REPLY
    if [[ "$REPLY" == "production" ]]; then
        BASE_TAG="production"
        echo -e "${YELLOW}Overriding to production build.${NC}"
        read -p "Confirm production build from non-main branch? (y/N): " -r CONFIRM
        [[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
    elif [[ "$REPLY" =~ ^[Nn]$ ]]; then
        echo "Aborted."; exit 1
    fi
fi

HASH_TAG="${BASE_TAG}-${COMMIT_HASH}"
echo
echo -e "Building: ${GREEN}${BOLD}agent-home:$BASE_TAG${NC} + ${CYAN}agent-home:$HASH_TAG${NC}"
echo

# --- Build ---
docker build \
    -t "agent-home:$BASE_TAG" \
    -t "agent-home:$HASH_TAG" \
    --build-arg "GIT_COMMIT=$COMMIT_FULL" \
    --build-arg "GIT_BRANCH=$BRANCH" \
    "$REPO_ROOT"

echo
echo -e "${GREEN}${BOLD}Build complete!${NC}"
echo -e "  agent-home:$BASE_TAG"
echo -e "  agent-home:$HASH_TAG"
