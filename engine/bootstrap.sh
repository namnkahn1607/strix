#!/usr/bin/env bash
################################################################################
# This script will install and bootstrap environment for local development in
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
# Usage: bash bootstrap.sh
################################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # strix/engine/
REPO_ROOT="$(dirname "$PROJECT_ROOT")"                       # strix/
cd "$PROJECT_ROOT"

LLVM_VERSION=17
CMAKE_MIN_MAJOR=3
CMAKE_MIN_MINOR=22
CMAKE_VENDOR_VERSION="3.31.12" # Latest CMake 3

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }

TOTAL_PHASE=8

# ==============================================================================
# 0. Platform
# ==============================================================================
log "[0/$TOTAL_PHASE] Platform validation"

[[ "$(uname -s)" == "Linux" ]] || die "Linux only. Consider remote SSH development for MacOS or WSL2 for Windows."
[[ "$(uname -m)" == "x86_64" ]] || die "x86-64 only. ARM is not yet supported."

# ==============================================================================
# 1. System utilities
# ==============================================================================
log "[1/$TOTAL_PHASE] Check & install system utilities"

REQUIRED_BASE_TOOLS=(git python3 pkg-config curl tar zip unzip realpath)
missing_base=()

for tool in "${REQUIRED_BASE_TOOLS[@]}"; do
    command -v "$tool" >/dev/null 2>&1 || missing_base+=("$tool")
done
if [[ ${#missing_base[@]} -gt 0 ]]; then
    log "Missing ${missing_base[*]} - install via apt (sudo needed)."
    sudo apt-get update -qq
    sudo apt-get install -y "${missing_base[@]}"
else
    log "All base utilities present."
fi

# ==============================================================================
# 2. LLVM toolchain
# ==============================================================================
log "[2/$TOTAL_PHASE] Check & install LLVM $LLVM_VERSION toolchain"

LLVM_TOOLS=("clang-$LLVM_VERSION" "clang++-$LLVM_VERSION" "clang-tidy-$LLVM_VERSION" "lld-$LLVM_VERSION" "lldb-$LLVM_VERSION")
missing_llvm=()

for tool in "${LLVM_TOOLS[@]}"; do
    command -v "$tool" >/dev/null 2>&1 || missing_llvm+=("$tool")
done

if [[ ${#missing_llvm[@]} -gt 0 ]]; then
    log "Missing ${missing_llvm[*]} - installing LLVM $LLVM_VERSION via apt.llvm.org"
    curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh
    chmod +x /tmp/llvm.sh
    sudo /tmp/llvm.sh $LLVM_VERSION
    rm -f /tmp/llvm.sh
    
    # clang-tidy and lldb is not supported by llvm.sh without "all".
    sudo apt-get install -y "clang-tidy-$LLVM_VERSION"
    sudo apt-get install -y "lldb-$LLVM_VERSION"
else
    log "All necessary LLVM $LLVM_VERSION tools present."
fi

# ==============================================================================
# 3. Generator: Ninja
# ==============================================================================
log "[3/$TOTAL_PHASE] Check & install generator: Ninja"

command -v ninja >/dev/null 2>&1 || {
    log "Missing Ninja - install via apt"
    sudo apt-get install -y ninja-build
}

# ==============================================================================
# 4. Task runner: just
# ==============================================================================
log "[4/$TOTAL_PHASE] Check & install task runner: just"

command -v just >/dev/null 2>&1 || {
    log "Missing just - installing to /usr/local/bin"
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
        | sudo bash -s -- --to /usr/local/bin
}

# ==============================================================================
# 5. Build system: CMake (>= 3.22)
# ==============================================================================
log "[5/$TOTAL_PHASE] Check build system: CMake"

CMAKE_BIN="cmake"
need_vendor_cmake=true

if command -v cmake >/dev/null 2>&1; then
    CURR_VER="$(cmake --version | head -1 | grep -oP '\d+\.\d+\.\d+')"
    CURR_MAJOR="${CURR_VER%%.*}"
    CURR_MINOR="$(echo "$CURR_VER" | cut -d. -f2)"

    if (( CURR_MAJOR > CMAKE_MIN_MAJOR || (CURR_MAJOR == CMAKE_MIN_MAJOR && CURR_MINOR >= CMAKE_MIN_MINOR) )); then
        need_vendor_cmake=false
    else
        warn "Incompatible CMake version was found (need >= $CMAKE_MIN_MAJOR.$CMAKE_MIN_MINOR)."
    fi
else
    warn "Cannot detect system CMake. An alternate will be installed."
fi

if $need_vendor_cmake; then
    VENDOR_CMAKE_DIR="$PROJECT_ROOT/vendor/cmake"
    if [[ ! -x "$VENDOR_CMAKE_DIR/bin/cmake" ]]; then
        log "Installing vendored CMAKE $CMAKE_VENDOR_VERSION into vendor/cmake."
        mkdir -p "$VENDOR_CMAKE_DIR"
        TARFILE="cmake-${CMAKE_VENDOR_VERSION}-linux-x86_64.tar.gz"
        curl -fsSL -o "/tmp/$TARFILE" \
            "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VENDOR_VERSION}/${TARFILE}"
        tar -xzf "/tmp/$TARFILE" -C "$VENDOR_CMAKE_DIR" --strip-components=1
        rm -f "/tmp/$TARFILE"
    fi
    CMAKE_BIN="$VENDOR_CMAKE_DIR/bin/cmake"
    log "Vendored CMake is now at $CMAKE_BIN."
fi

# ==============================================================================
# 6. Package manager: vcpkg (submodule)
# ==============================================================================
log "[6/$TOTAL_PHASE] Synchronize and bootstrap vcpkg"

git -C "$REPO_ROOT" submodule update --init --recursive

mkdir -p "$HOME/.cache/vcpkg-archives" "$HOME/.cache/vcpkg-downloads"

if [[ ! -x "$PROJECT_ROOT/vendor/vcpkg/vcpkg" ]]; then
    "$PROJECT_ROOT/vendor/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
else
    log "vcpkg is already bootstrapped."
fi

BASELINE_COMMIT=$(grep -oP '"builtin-baseline"\s*:\s*"\K[a-f0-9]{40}' "$PROJECT_ROOT/vcpkg.json")
if ! git -C "$PROJECT_ROOT/vendor/vcpkg" cat-file -e "${BASELINE_COMMIT}" 2>/dev/null; then
    log "Baseline commit not present in vendored vcpkg. Fetching..."
    git -C "$PROJECT_ROOT/vendor/vcpkg" fetch origin "${BASELINE_COMMIT}"
fi

# ==============================================================================
# 7. Secrets
# ==============================================================================
log "[7/$TOTAL_PHASE] Generate/update secrets"

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
# 8. Configure CMake presets
# ==============================================================================
log "[8/$TOTAL_PHASE] Configure CMake presets" 

echo ""
echo "Which presets do you want configured?"
echo "  [Y] Release only          (try out Strix)"
echo "  [n] Debug + Asan + Tsan   (development, release stays unconfigured)"
echo "  [a] All                   (might take longer)"
read -rp "Choice [Y/n/a]: " answer
answer="${answer:-Y}"
 
case "$answer" in
    [Aa]*)
        log "Configuring all four presets."
        "$CMAKE_BIN" --preset debug
        "$CMAKE_BIN" --preset asan
        "$CMAKE_BIN" --preset tsan
        "$CMAKE_BIN" --preset release
        ;;
    [Nn]*)
        log "Configuring debug, asan, tsan (shared vcpkg triplet)."
        "$CMAKE_BIN" --preset debug
        "$CMAKE_BIN" --preset asan
        "$CMAKE_BIN" --preset tsan
        log "release not configured. Run '$CMAKE_BIN --preset release' when you need it."
        ;;
    *)
        log "Configuring release only."
        "$CMAKE_BIN" --preset release
        ;;
esac

log "Finished!"
