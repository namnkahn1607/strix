#!/usr/bin/env bash
################################################################################
# This script will install and set up environment for local development in
# strix/engine/, which includes:
#   1. System packages: git, python3, curl, ninja-build, just, LLVM toolchain
#      and other utilities.
#   2. Install a vendored CMake into vendor/cmake/ if minimum CMake requirements 
#      are not met.
#   3. Synchronize and bootstrap vcpkg submodule on pinned release.
#   4. Generate local secrets.
#   5. Pre-configure specified CMake presets.
#
# Note: might require 'sudo' privileges.
# Idempotent: safe to re-run after a failure; each step checks before acting.
# Usage: bash setup.sh
################################################################################

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)" # strix/
PROJECT_ROOT="$REPO_ROOT/engine"
cd "$PROJECT_ROOT"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }

TOTAL_PHASE=3

# ==============================================================================
# 1. Install all devtools/dependencies
# ==============================================================================
log "[1/$TOTAL_PHASE] Installing devtools & dependencies."
bash "$REPO_ROOT/scripts/install/cpp_toolchain.sh" all

CMAKE_BIN_STATE="$PROJECT_ROOT/.state/cmake_bin"
[[ -f "$CMAKE_BIN_STATE" ]] || die "Missing $CMAKE_BIN_STATE - Install script did not run or failed."
CMAKE_BIN="$(cat "$CMAKE_BIN_STATE")"
[[ "$CMAKE_BIN" == "cmake" || -x "$CMAKE_BIN" ]] || die "Resolved CMAKE_BIN ('$CMAKE_BIN') is not executable."
 
# ==============================================================================
# 2. Secrets
# ==============================================================================
log "[2/$TOTAL_PHASE] Generate/update secrets"
 
ENV_FILE="$PROJECT_ROOT/.env"
touch "$ENV_FILE"
 
upsert_env_var() {
    local key="$1" value="$2"
    if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    else
        echo "${key}=${value}" >> "$ENV_FILE"
    fi
}
 
resolve_model() {
    local filename="$1"
    local results
 
    mapfile -t results < <(find "$REPO_ROOT" -name "$filename" -type f 2>/dev/null)
 
    if [[ ${#results[@]} -eq 0 ]]; then
        die "'$filename' not found anywhere under $REPO_ROOT"
    fi
    if [[ ${#results[@]} -gt 1 ]]; then
        warn "Multiple '$filename' found - ambiguous:"
        printf '  %s\n' "${results[@]}" >&2
        die "Remove duplicates or set the path manually in .env."
    fi
 
    realpath "${results[0]}"
}
 
TOKENIZER_PATH="$(resolve_model tokenizer.onnx)"
TRANSFORMER_PATH="$(resolve_model transformer.onnx)"
 
upsert_env_var "TOKENIZER_PATH" "$TOKENIZER_PATH"
upsert_env_var "TRANSFORMER_PATH" "$TRANSFORMER_PATH"
 
# ==============================================================================
# 3. Configure CMake presets
# ==============================================================================
log "[3/$TOTAL_PHASE] Configure CMake presets"
 
echo ""
echo "Which presets do you want configured?"
echo "  [Y] Release only          (try out Strix)"
echo "  [n] Debug + Asan + Tsan   (development, release stays unconfigured)"
echo "  [a] All                   (might take longer)"
read -rp "Choice [Y/n/a]: " answer
answer="${answer:-Y}"
 
compile_database() {
    local build_type="$1"
    local output_file=".clangd"
 
    if [[ -z "$build_type" ]]; then
        die "Missing build type argument to compile_database()."
    fi
 
    case "$build_type" in
        "debug" | "asan" | "tsan")
            cat << 'EOF' > "$output_file"
---
CompileFlags:
  CompilationDatabase: out/debug
EOF
            log "Generated compile database for build type: $build_type"
            ;;
 
        "release")
            cat << 'EOF' > "$output_file"
---
CompileFlags:
  CompilationDatabase: out/release
EOF
            log "Generated compile database for build type: $build_type"
            ;;
 
        *)
            die "Unsupported/wrong build type: $build_type"
            ;;
    esac
}
 
case "$answer" in
    [Aa]*)
        log "Configuring all four presets."
        "$CMAKE_BIN" --preset debug
        "$CMAKE_BIN" --preset asan
        "$CMAKE_BIN" --preset tsan
        "$CMAKE_BIN" --preset release
        compile_database "release"
        ;;
    [Nn]*)
        log "Configuring debug, asan, tsan (shared vcpkg triplet)."
        "$CMAKE_BIN" --preset debug
        "$CMAKE_BIN" --preset asan
        "$CMAKE_BIN" --preset tsan
        compile_database "debug"
        log "release not configured. Run '$CMAKE_BIN --preset release' when you need it."
        ;;
    *)
        log "Configuring release only."
        "$CMAKE_BIN" --preset release
        compile_database "release"
        ;;
esac
 
log "Finished!"
