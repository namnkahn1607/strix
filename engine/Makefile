# strix/engine/Makefile

export TOKENIZER_PATH   := $(shell echo ~/Alder/strix/model/tokenizer.onnx)
export TRANSFORMER_PATH := $(shell echo ~/Alder/strix/model/transformer.onnx)

.PHONY: all checkenv clean bench test

checkenv:
	@echo "Tokenizer Absolute path: ${TOKENIZER_PATH}"
	@echo "Transformer Absolute path: ${TRANSFORMER_PATH}"

clean:
	@rm -rf out/
	@mkdir -p out/
	@echo "Nuked artifact directory"

# ==============================================================================
# RELEASE profile recipes
# ==============================================================================
rconfig:
	@echo "=== Loading CMake project: Release ==="
	@cmake --preset release

rcontract: rconfig
	@cmake --build --preset release --target engine_contract

rbuild: rcontract rconfig
	@echo "=== Building executables: Release ==="
	@cmake --build --preset release --target strix_engine

rrun: rbuild
	@echo "=== Running Vector Engine: Release ==="
	./out/release/strix_engine

# ==============================================================================
# DEBUG profile recipes
# ==============================================================================
dconfig:
	@echo "=== Loading CMake project: Debug ==="
	cmake --preset debug

dcontract: dconfig
	@cmake --build --preset debug --target engine_contract

dbuild: dcontract dconfig
	@echo "=== Building executables: Debug ==="
	cmake --build --preset debug --target strix_engine

drun: dbuild
	@echo "=== Running Vector Engine: Debug ==="
	./out/debug/strix_engine

# ==============================================================================
# Bench/Test recipes
# ==============================================================================
bench: rconfig
	@echo "=== Benchmark AVX2 Dot Product ==="
	@cmake --build --preset release --target avx2_bench && ./out/release/avx2_bench

	@echo "=== Benchmark Inference model (Zero-shot) ==="
	@cmake --build --preset release --target model_bench && ./out/release/model_bench

	@echo "=== Benchmark Inference model (Custom Ops) ==="
	@cmake --build --preset release --target customops_bench && ./out/release/customops_bench

	@echo "=== Benchmark Vector searching on L0 Buffer ==="
	@cmake --build --preset release --target searchl0_bench && ./out/release/searchl0_bench

	@echo "=== Benchmark Arena's Payload Read/Write ==="
	@cmake --build --preset release --target payloadrw_bench && ./out/release/payloadrw_bench

test: rconfig dconfig
	@echo "=== Unit-testing AVX2 Dot Product ==="
	@cmake --build --preset debug --target avx2_test && ./out/debug/avx2_test
	@cmake --build --preset release --target avx2_test && ./out/release/avx2_test

	@echo "=== Unit-testing Inference model ==="
	@cmake --build --preset debug --target model_test && ./out/debug/model_test
	@cmake --build --preset release --target model_test && ./out/release/model_test

	@echo "=== Unit-testing Control Block operations ==="
	@cmake --build --preset debug --target ctrl_test && ./out/debug/ctrl_test
	@cmake --build --preset release --target ctrl_test && ./out/release/ctrl_test

	@echo "=== Unit-testing Vector searching on L0 Buffer ==="
	@cmake --build --preset debug --target searchl0_test && ./out/debug/searchl0_test
	@cmake --build --preset release --target searchl0_test && ./out/release/searchl0_test

	@echo "=== Unit-testing Arena's Payload Read/Write ==="
	@cmake --build --preset debug --target payloadrw_test && ./out/debug/payloadrw_test
	@cmake --build --preset release --target payloadrw_test && ./out/release/payloadrw_test

# ==============================================================================
# SANITIZER recipes
# ==============================================================================
aconfig:
	@echo "=== Loading CMake project: Address Sanitizer ==="
	@cmake --preset asan

tconfig:
	@echo "=== Loading CMake project: Thread Sanitizer ==="
	@cmake --preset tsan

sanitize: aconfig tconfig
	@echo "=== Address Sanitizing Arena's Payload Read/Write ==="
	@cmake --build --preset asan --target payloadrw_test && .out/asan/payloadrw_test
	