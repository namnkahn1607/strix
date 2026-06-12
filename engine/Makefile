# strix/engine/Makefile

export TOKENIZER_PATH   := $(shell echo ~/Alder/strix/model/tokenizer.onnx)
export TRANSFORMER_PATH := $(shell echo ~/Alder/strix/model/transformer.onnx)

.PHONY: all config build run bench clean

all: build

check-env:
	@echo "Tokenizer Absolute path: ${TOKENIZER_PATH}"
	@echo "Transformer Absolute path: ${TRANSFORMER_PATH}"

clean:
	rm -rf out/
	mkdir -p out/
	@echo "Nuked artifact directory"

# ===========================================================================
# RELEASE profile Recipes
# ===========================================================================
config-r:
	@echo "=== Loading CMake project (Release) ==="
	cmake --preset release

contract-r: config
	cmake --build --preset release --target engine_contract

build-r: config-r
	@echo "=== Building executables (Release) ==="
	cmake --build --preset release

run-r: build-r
	@echo "=== Running Vector Engine (Release) ==="
	./out/release/strix_engine

bench: build-r
	@echo "=== Benchmark AVX2 Dot Product ==="
	./out/release/avx2_bench

	@echo "=== Benchmark Inference model (Zero-shot) ==="
	./out/release/model_bench

	@echo "=== Benchmark Inference model (Custom Ops) ==="
	./out/release/customops_bench

# ===========================================================================
# DEBUG profile Recipes
# ===========================================================================
config-d:
	@echo "=== Loading CMake project (Debug) ==="
	cmake --preset debug

contract-d: config
	cmake --build --preset debug --target engine_contract

build-d: config-d
	@echo "=== Building executables (Debug) ==="
	cmake --build --preset debug

run-d: build-d
	@echo "=== Running Vector Engine (Debug) ==="
	./out/debug/strix_engine

test: build-d
	@echo "=== Unit-testing AVX2 Dot Product ==="
	./out/debug/avx2_test

	@echo "=== Unit-testing Inference model ==="
	./out/debug/model_test

	@echo "=== Unit-testing Control Block operations ==="
	./out/debug/ctrl_test
