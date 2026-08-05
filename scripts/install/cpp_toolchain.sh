#!/usr/bin/env bash
################################################################################
# Idempotent installer/bootstrapper for engine/ devtools.
#
# Usage:
#   bash cpp_toolchain.sh <target> [<target> ...]
#   bash cpp_toolchain.sh all
#
# Targets:
#   ninja   Ninja generator
#   cmake   CMake build system, vendored fallback.
#           Resolved binary path is persisited to .state/cmake_bin.
#   utils   base utilities (python3, pkg-config, curl, tar, zip, unzip, realpath)
#   llvm    clang/clang++/clang-tidy/lld/lldb, pinned to LLVM_VERSION
#   just    `just`` task runner
#   vcpkg   vcpkg submodule sync + bootstrap + builtin-baseline pin.
#   ort     onnxruntime submodule sync + Clang patch series + build + harvest.

# Exit non-zero on failure or unknown target.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)" # strix/
PROJECT_ROOT="$REPO_ROOT/engine"
cd "$PROJECT_ROOT"

log() { echo "[INFO] $*" >&2; }
warn() { echo "[WARN] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }

assert_platform() {
    [[ "$(uname -s)" == "Linux" ]] || die "Linux only. Consider remote SSH for MacOS or WSL2 for Windows."
    [[ "$(uname -m)" == "x86_64" ]] || die "x86-64 only. ARM is not yet supported."
}

usage() {
    cat <<'EOF'
Usage: bash cpp_toolchain.sh <target> [<target> ...]
       bash cpp_toolchain.sh all

Targets: ninja  cmake  utils  llvm  just  vcpkg  ort  all
EOF
}

# ==============================================================================
# ninja
# ==============================================================================
install_ninja() {
    log "Check & install generator."
    command -v ninja >/dev/null 2>&1 || {
        log "Missing Ninja - install via apt."
        sudo apt-get install -y ninja-build
    }
}

# ==============================================================================
# cmake
# ==============================================================================
CMAKE_MIN_MAJOR=3
CMAKE_MIN_MINOR=22
CMAKE_VENDOR_VERSION="3.31.12" # Latest CMake 3

install_cmake() {
    log "Check & install build system."
    local cmake_bin="cmake"
    local need_vendor=true

    if command -v cmake >/dev/null 2>&1; then
        local curr_ver curr_major curr_minor
        curr_ver="$(cmake --version | head -1 | grep -oP '\d+\.\d+\.\d+')"
        curr_major="${curr_ver%%.*}"
        curr_minor="$(echo "$curr_ver" | cut -d. -f2)"

        if (( curr_major > CMAKE_MIN_MAJOR || (curr_major == CMAKE_MIN_MAJOR && curr_minor >= CMAKE_MIN_MINOR) )); then
            need_vendor=false
        else
            warn "Incompatible system CMake $curr_ver (need >= $CMAKE_MIN_MAJOR.$CMAKE_MIN_MINOR)."
        fi
    else
        warn "Cannot detect CMake. An alternate will be installed."
    fi

    if $need_vendor; then
        local vendor_dir="$PROJECT_ROOT/vendor/cmake"
        if [[ ! -x "$vendor_dir/bin/cmake" ]]; then
            log "Installing vendored CMake $CMAKE_VENDOR_VERSION into vendor/cmake."
            mkdir -p "$vendor_dir"
            local tarfile="cmake-${CMAKE_VENDOR_VERSION}-linux-x86_64.tar.gz"
            curl -fsSL -o "/tmp/$tarfile" \
                "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VENDOR_VERSION}/${tarfile}"
            tar -xzf "/tmp/$tarfile" -C "$vendor_dir" --strip-components=1
            rm -f "/tmp/$tarfile"
        else
            log "Vendored CMake already present and executable."
        fi
        cmake_bin="$vendor_dir/bin/cmake"
    fi

    STATE_DIR="$PROJECT_ROOT/.state"
    mkdir -p "$STATE_DIR"

    echo "$cmake_bin" > "$STATE_DIR/cmake_bin"
    log "Resolved binary: $cmake_bin."
}

# ==============================================================================
# utils
# ==============================================================================
install_utils() {
    log "Check & install system base utilities."
    local required=(python3 pkg-config curl tar zip unzip realpath)
    local missing=()

    for tool in "${required[@]}"; do
        command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log "Missing ${missing[*]} - install via apt (sudo needed)."
        sudo apt-get update -qq
        sudo apt-get install -y "${missing[@]}"
    else
        log "All base utilities present."
    fi
}

# ==============================================================================
# llvm
# ==============================================================================
LLVM_VERSION=17

install_llvm() {
    log "Check & install LLVM $LLVM_VERSION toolchain."
    local tools=("clang-$LLVM_VERSION" "clang++-$LLVM_VERSION" "clang-tidy-$LLVM_VERSION" "lld-$LLVM_VERSION" "lldb-$LLVM_VERSION")
    local missing=()

    for tool in "${tools[@]}"; do
        command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log "Missing ${missing[*]} - installing LLVM $LLVM_VERSION via apt.llvm.org."
        curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh
        chmod +x /tmp/llvm.sh
        sudo /tmp/llvm.sh "$LLVM_VERSION"
        rm -f /tmp/llvm.sh

        # clang-tidy and lldb are not pulled in by llvm.sh without "all".
        sudo apt-get install -y "clang-tidy-$LLVM_VERSION"
        sudo apt-get install -y "lldb-$LLVM_VERSION"
    else
        log "All necessary LLVM $LLVM_VERSION tools present."
    fi
}

# ==============================================================================
# just
# ==============================================================================
install_just() {
    log "Check & install task runner 'just'."
    command -v just >/dev/null 2>&1 || {
        log "Missing just - installing to /usr/local/bin."
        curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
            sudo bash -s -- --to /usr/local/bin
    }
}

# ==============================================================================
# vcpkg
# ==============================================================================
bootstrap_vcpkg() {
    log "Synchronize & bootstrap vcpkg"
    git -C "$REPO_ROOT" submodule update --init --recursive --filter=blob:none \
        -- engine/vendor/vcpkg

    mkdir -p "$HOME/.cache/vcpkg-archives" "$HOME/.cache/vcpkg-downloads"

    if [[ ! -x "$PROJECT_ROOT/vendor/vcpkg/vcpkg" ]]; then
        "$PROJECT_ROOT/vendor/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
    else
        log "vcpkg is already bootstrapped."
    fi

    # Ensure the current baseline commit hash is reachable in object database.
    local baseline_commit
    baseline_commit=$(grep -oP '"builtin-baseline"\s*:\s*"\K[a-f0-9]{40}' "$PROJECT_ROOT/vcpkg.json")

    if ! git -C "$PROJECT_ROOT/vendor/vcpkg" cat-file -e "${baseline_commit}" 2>/dev/null; then
        log "Baseline commit not present in vendored vcpkg. Fetching..."
        git -C "$PROJECT_ROOT/vendor/vcpkg" fetch origin "${baseline_commit}"
    fi

    # Ensure it's actually checked out in the vendor submodule.
    local current_head
    current_head=$(git -C "$PROJECT_ROOT/vendor/vcpkg" rev-parse HEAD)
    if [[ "$current_head" != "$baseline_commit" ]]; then
        warn "vendor/vcpkg HEAD ($current_head) != builtin-baseline ($baseline_commit). Checking out correct commit."
        git -C "$PROJECT_ROOT/vendor/vcpkg" checkout "$baseline_commit"
    else
        log "vendor/vcpkg HEAD already pinned to builtin-baseline."
    fi
}

# ==============================================================================
# ort
# ==============================================================================
# ORT build requires CMake >= 3.28
ORT_CMAKE_VERSION="3.28.6"

_resolve_ort_cmake() {
    local vendor_dir="$PROJECT_ROOT/vendor/cmake-ort"
    if [[ ! -x "$vendor_dir/bin/cmake" ]]; then
        log "Installing dedicated CMake $ORT_CMAKE_VERSION for ORT build."
        mkdir -p "$vendor_dir"

        local tarfile="cmake-${ORT_CMAKE_VERSION}-linux-x86_64.tar.gz"
        curl -fsSL -o "/tmp/$tarfile" \
            "https://github.com/Kitware/CMake/releases/download/v${ORT_CMAKE_VERSION}/${tarfile}"
        tar -xzf "/tmp/$tarfile" -C "$vendor_dir" --strip-components=1
        rm -f "/tmp/$tarfile"
    fi
    echo "$vendor_dir/bin/cmake"
}

ORT_TAG="v1.23.2"
EXPECTED_ORT_COMMIT="a83fc4d58cb48eb68890dd689f94f28288cf2278"

ORT_PATCH_DIR="$PROJECT_ROOT/vendor/patches/onnxruntime"
ORT_SRC_DIR="$PROJECT_ROOT/vendor/onnxruntime"
ORT_HARVEST_DIR="$PROJECT_ROOT/ort"
ORT_BUILT_MARKER="$ORT_HARVEST_DIR/.built_commit"

bootstrap_ort() {
    log "Synchronize & bootstrap onnxruntime $ORT_TAG"
    git -C "$REPO_ROOT" submodule update --init --recursive --filter=blob:none \
        -- engine/vendor/onnxruntime

    local actual_ort_commit
    actual_ort_commit=$(git -C "$ORT_SRC_DIR" rev-parse HEAD)
    if [[ "$actual_ort_commit" != "$EXPECTED_ORT_COMMIT" ]]; then
        die "onnxruntime local HEAD resolves to $actual_ort_commit, " \
            "expected $EXPECTED_ORT_COMMIT. "
    fi

    # Idempotent ensurance: Reset the submodule to pristine state BEFORE
    # applying any patches.
    git -C "$ORT_SRC_DIR" checkout --force "$actual_ort_commit"
    git -C "$ORT_SRC_DIR" clean -fdx
    git -C "$ORT_SRC_DIR" submodule foreach --recursive \
        'git checkout --force . && git clean -fdx'

    log "Applying clang-17 compatibility patch series."
    shopt -s nullglob
    local patches=("$ORT_PATCH_DIR"/*.diff)
    shopt -u nullglob

    if [[ ${#patches[@]} -eq 0 ]]; then
        warn "No patches found in $ORT_PATCH_DIR. Skipping patch step."
    else
        for p in "${patches[@]}"; do
            git -C "$ORT_SRC_DIR" apply --check "$p" || die \
                "Patch $p failed against $actual_ort_commit. " \
                "Patch series might be stale - regenerate it (see bumping procedure doc)." 
        done
        for p in "${patches[@]}"; do
            git -C "$ORT_SRC_DIR" apply "$p"
        done
    fi

    # Skip rebuild + reharvest if already done it.
    if [[ -f "$ORT_BUILT_MARKER" ]] && [[ "$(cat "$ORT_BUILT_MARKER")" == "$actual_ort_commit" ]]; then
        log "Already harvested for $actual_ort_commit. Skipping build."
        log "Revert to pristine state after applying patches."
        git -C "$ORT_SRC_DIR" reset --hard HEAD
        git -C "$ORT_SRC_DIR" clean -fdx
        return 0
    fi

    local cmake_ort_bin
    cmake_ort_bin="$(_resolve_ort_cmake)"

    log "Building onnxruntime. Might take a while."
    (
        cd "$ORT_SRC_DIR"
        ./build.sh \
            --cmake_path "$cmake_ort_bin" \
            --cmake_generator Ninja \
            --config Release \
            --build_shared_lib \
            --parallel 4 \
            --use_extensions \
            --skip_tests \
            --cmake_extra_defines \
                CMAKE_C_COMPILER=/usr/bin/clang-17 \
                CMAKE_CXX_COMPILER=/usr/bin/clang++-17 \
                CMAKE_CXX_FLAGS="-Wno-error -O3 -march=x86-64-v3" \
                CMAKE_C_FLAGS="-Wno-error -O3 -march=x86-64-v3"
    ) || die "Build failed."

    log "Harvesting artifacts & onnxruntime headers."
    mkdir -p "$ORT_HARVEST_DIR/lib" "$ORT_HARVEST_DIR/include"
    cp -a "$ORT_SRC_DIR"/build/Linux/Release/libonnxruntime.so* "$ORT_HARVEST_DIR/lib/"
    cp -a "$ORT_SRC_DIR"/include/onnxruntime/core/session/. "$ORT_HARVEST_DIR/include/"
    echo "$actual_ort_commit" > "$ORT_BUILT_MARKER"

    log "Revert to pristine state after applying patches."
    git -C "$ORT_SRC_DIR" reset --hard HEAD
    git -C "$ORT_SRC_DIR" clean -fdx
    log "onnxruntime ready ($actual_ort_commit)."
}

# ==============================================================================
# Entry point
# ==============================================================================
main() {
    [[ $# -ge 1 ]] || { usage; exit 1; }
    assert_platform

    local targets=("$@")
    [[ "${targets[0]}" == "all" ]] && targets=(ninja cmake utils llvm just vcpkg ort)

    for target in "${targets[@]}"; do
        case "$target" in
            ninja)  install_ninja ;;
            cmake)  install_cmake ;;
            utils)  install_utils ;;
            llvm)   install_llvm ;;
            just)   install_just ;;
            vcpkg)  bootstrap_vcpkg ;;
            ort)    bootstrap_ort ;;
            *) die "Unknown target: '$target'. Run without args to see usage." ;;
        esac
    done
}

main "$@"
