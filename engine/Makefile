# strix/engine/Makefile

export TOKENIZER_PATH   := $(shell echo ~/Alder/strix/model/tokenizer.onnx)
export TRANSFORMER_PATH := $(shell echo ~/Alder/strix/model/transformer.onnx)

.PHONY: all checkenv clean bench test

all: rbuild

checkenv:
	@echo "Tokenizer Absolute path: ${TOKENIZER_PATH}"
	@echo "Transformer Absolute path: ${TRANSFORMER_PATH}"

clean:
	rm -rf out/
	mkdir -p out/
	@echo "Nuked artifact directory"

# ===========================================================================
# RELEASE profile Recipes
# ===========================================================================
rconfig:
	@echo "=== Loading CMake project (Release) ==="
	cmake --preset release

rcontract: rconfig
	cmake --build --preset release --target engine_contract

rbuild: rconfig
	@echo "=== Building executables (Release) ==="
	cmake --build --preset release

rrun: rbuild
	@echo "=== Running Vector Engine (Release) ==="
	./out/release/strix_engine

# ===========================================================================
# DEBUG profile Recipes
# ===========================================================================
dconfig:
	@echo "=== Loading CMake project (Debug) ==="
	cmake --preset debug

dcontract: dconfig
	cmake --build --preset debug --target engine_contract

dbuild: dconfig
	@echo "=== Building executables (Debug) ==="
	cmake --build --preset debug

drun: dbuild
	@echo "=== Running Vector Engine (Debug) ==="
	./out/debug/strix_engine

# ===========================================================================
# Bench/Test Recipes
# ===========================================================================
bench: rbuild
	@echo "=== Benchmark AVX2 Dot Product ==="
	./out/release/avx2_bench

	@echo "=== Benchmark Inference model (Zero-shot) ==="
	./out/release/model_bench

	@echo "=== Benchmark Inference model (Custom Ops) ==="
	./out/release/customops_bench

	@echo "=== Benchmark Vector searching on L0 Buffer ==="
	./out/release/searchl0_bench

	@echo "=== Benchmark Arena's Payload Read/Write ==="
	./out/release/payloadrw_bench

test: dbuild rbuild
	@echo "=== Unit-testing AVX2 Dot Product ==="
	./out/debug/avx2_test
	./out/release/avx2_test

	@echo "=== Unit-testing Inference model ==="
	./out/debug/model_test
	./out/release/model_test

	@echo "=== Unit-testing Control Block operations ==="
	./out/debug/ctrl_test
	./out/release/ctrl_test

	@echo "=== Unit-testing Vector searching on L0 Buffer ==="
	./out/debug/searchl0_test
	./out/release/searchl0_test

	@echo "=== Unit-testing Arena's Payload Read/Write ==="
	./out/debug/payloadrw_test
	./out/release/payloadrw_test
