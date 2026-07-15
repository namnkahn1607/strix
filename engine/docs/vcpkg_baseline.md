# Bumping `builtin-baseline` in `vcpkg.json`

## Principles

`vcpkg.json`'s `builtin-baseline` field and the `vendor/vcpkg` submodule commit
**must always move together, in the same commit**. Baseline resolution reads git
history inside `vendor/vcpkg` - if the two drift apart, the build can often
still work (a sufficiently recent vcpkg tool will auto-fetch the missing commit),
but dev machines and CI stop being deterministic in the way pinning the
submodule was meant to guarantee.

## When NOT to bump

If the diff in step 2 shows no version changes for any port in Strix's
actual dependency graph (only unrelated ports changed), bumping brings no
benefit and only costs CI verification time. Skip it and wait for a bump
with substantive content.

## Procedure

The following procedure expected the CWD to be `strix/engine/`.

### 1. Pick the target commit

```bash
cd vendor/vcpkg
git fetch origin
git log origin/master --oneline -20   # or pick a specific tag, e.g. v2026.04.27
```

### 2. Diff the baseline to see exactly what changed

```bash
OLD_COMMIT=$(grep -oP '"builtin-baseline"\s*:\s*"\K[a-f0-9]{40}' ../vcpkg.json)
NEW_COMMIT=<commit-chosen-in-step-1>

git diff "$OLD_COMMIT" "$NEW_COMMIT" -- versions/baseline.json
```

Only care about ports Strix actually depends on: `grpc`, `gtest`, `benchmark`,
and their transitive deps (`abseil`, `protobuf`, `openssl`, `zlib`, `c-ares`,
`re2`, `utf8-range`). A version bump on an unrelated port can be ignored.

### 3. Move the submodule to the target commit

```bash
git checkout "$NEW_COMMIT"
cd ..
git add vendor/vcpkg
```

### 4. Update `vcpkg.json` to match the checked-out commit

```json
"builtin-baseline": "<NEW_COMMIT>"
```

### 5. Rebuild all four presets

```bash
cmake --preset debug   && cmake --build --preset debug
cmake --preset asan    && cmake --build --preset asan
cmake --preset tsan    && cmake --build --preset tsan
cmake --preset release && cmake --build --preset release
```

Since the binary cache key is `{port, version, triplet, compiler, flags}`, only
ports whose version actually changed in Step 2 will miss cache and rebuild -
everything else in the dependency graph still hits cache, so rebuild time scales
with the size of the diff, not a full rebuild.

### 6. Commit atomically

```bash
git add vcpkg.json vendor/vcpkg
git commit -m "ops: bump builtin-baseline to <NEW_COMMIT> (grpc X.Y.Z, ...)"
```

The commit message should list which ports actually changed version, so a future
reviewer knows the reasons for the bump without re-diffing `baseline.json`.

### 7. Open a Pull Request 

Let CI verify build+test on all four presets before finally merging.
