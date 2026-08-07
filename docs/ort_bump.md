# Bumping the vendored onnxruntime submodule

This document is the guideline for bumping `engine/vendor/onnxruntime` to a
newer upstream release.

## Prerequisites

- `clang-18` / `clang++-18` available.
- CMake 3.28.x available.
- A spare working directory *outside* the repository (do not do exploratory
  compiles inside `engine/vendor/onnxruntime`).

## Procedure

### 1. Fetch the candidate release

Do this in a scratch directory, **not** in `engine/vendor/onnxruntime`. Do
a disposable iteration surface while rediscovering which patches still apply
and which new breaks appear. Only touch the real submodule once having a
working patch series.

```bash
git clone --recursive -b <new-tag> https://github.com/microsoft/onnxruntime.git  ort-bump-scratch
cd ort-bump-scratch
git rev-parse HEAD   # record this - this is your candidate commit
```

### 2. Check whether the existing patch series still applies

```bash
cd ort-bump-scratch
for p in <strix>/engine/vendor/patches/onnxruntime/*.diff; do
  echo "=== $p ==="
  git apply --check "$p" || echo "MISMATCH - needs regeneration"
done
```

If all patches pass `--check`, proceed to Step 4. Note that a clean `--check`
only proves the *context lines* still match; it says nothing about whether
the new release introduced *additional* Clang incompatibilities elsewhere.

Otherwise if any patch fails `--check`, proceed to step 3.

### 3. How to regenerate a broken patch?

For each patch that no longer applies, apply what capable to fix the situation,
then let the compiler find the rest:

```bash
# Attempt a full compile:
./build.sh --cmake_path <path-to-cmake-3.28.x> \
  --ctest_path <path-to-ctest-3.28.x> \
  --cmake_generator Ninja --config Release --build_shared_lib \
  --parallel 4 --use_extensions --skip_tests \
  --cmake_extra_defines \
    CMAKE_C_COMPILER=/usr/bin/clang-18 \
    CMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
    CMAKE_CXX_FLAGS="-Wno-error -O3 -march=x86-64-v3" \
    CMAKE_C_FLAGS="-Wno-error -O3 -march=x86-64-v3"
```

Iterate: compiler aborts -> read the exact error -> patch the minimal offending
lines -> recompile -> repeat until compilation completes. Note every change as
proceeding - a consolidation into a new `.diff` file will be make at the end.

Pay close attention to patches that alters the runtime behavior of the library,
which might cause performance degradation, or worse: internal state corruption.

Known historical failure modes to watch for:

- Preprocessor-guarded static const declared *outside* the `#ifdef` that
  guards its only use site -> `-Wunused-variable` under `-Wextra`/`-march`
  combinations that strip the guarded block. Fix: `[[maybe_unused]]`.
- GCC-only CPU-feature strings in `__builtin_cpu_supports("...")` that
  LLVM's builtin dictionary doesn't recognize. Fix: `#if defined(__clang__)`
  branch with a safe fallback.
- GCC and Clang disagreeing on intrinsic arity for the same instruction
  (e.g. `__builtin_ia32_tpause`: GCC takes 2 args, while Clang's lower-level
  intrinsic takes 3 - the 64-bit deadline split across two registers must be
  done manually for Clang). Fix: compiler-gated branch.

Also watch for a class of bugs the compilation  **will not catch**: silent
miscompilation where code is *accepted* by both compilers but produces
different runtime behavior (type-punning relying on GCC-specific strict-
aliasing tolerance, layout/`offsetof` assumptions on non-standard-layout
types, packed-attribute alignment differences). The compilation loop only
proves "no errors" - it does not prove "identical semantics".

Once compilation is clean:

```bash
git diff > <new-patch-name>.diff
```

Split by touched file, matching the existing convention
`NNNN-<file>-<reason>.diff`.

### 4. Re-verification of the new patch series

Discard the scratch checkout, re-clone fresh, apply the neư `.diff` files, and
compile one more time from a pristine tree. This might catch a specific 
mistake: iterating Step 3 *in place* means the working tree accumulates the fix
incrementally, and it's easy to miss that a patch's `git diff` output captured
slightly more (or less) than intended.

```bash
rm -rf ort-bump-scratch && git clone --recursive -b <new-tag> ... ort-bump-scratch
cd ort-bump-scratch
git apply --check <new-patches>/*.diff && git apply <new-patches>/*.diff
./build.sh ...   # same invocation as step 3
```

### 5. Update the pin in the actual submodule

The submodule is configured with `shallow = true` to avoid pulling the whole
history of `microsoft/onnxruntime`.

```bash
cd <strix>/engine/vendor/onnxruntime
git fetch --depth 1 origin <new-commit-sha>   # explicit SHA-256 hash from Step 1
git checkout FETCH_HEAD
cd ../../..
git add engine/vendor/onnxruntime      # stages the gitlink only
```

Remember to pin using the commit SHA-256 hash retrieved from Step 1, not a tag.
Tags in third-party repositories are not guaranteed immutable.

### 6. Update the drift-detection constants in `cpp_toolchain.sh`

```bash
ORT_TAG="<new-tag>"
EXPECTED_ORT_COMMIT="<new-commit-sha>"
```

This pair exists purely to catch maintaining error (gitlink bumped without
updating this constant, or vice versa) - it does not itself constitute the
pin. The gitlink staged in Step 5 is the actual pin; `git ls-tree HEAD
engine/vendor/onnxruntime` is the ground truth, verifiable without entering
the submodule.

### 7. Replace the patch series

Remove the old `.diff` files from `engine/vendor/patches/onnxruntime/`,
add the new ones from Step 4.

### 8. Rebuild and run the accuracy test suites

```bash
rm -f engine/ort/.built_commit   # force rebuild; stale marker would skip it
bash cpp_toolchain.sh ort
```

Then run the inference-specific test suite to check whether the upstream
release brokes any of the source code (or even the tests themselves).

### 9. Open a Pull Request

Include in the description:
- Old commit -> new commit (both full SHAs, not just tags)
- Which patches were regenerated vs. unchanged, and why (link the exact
  compiler error each new hunk fixes).
- Confirmation that Step 8's accuracy test passed, with the actual
  similarity score.
