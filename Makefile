PP_RUN ?= /home/alp/Projects/profile-program/bin/pp-run
CACHE ?= /home/alp/Projects/profile-program/cache
HF_HOME ?= $(CACHE)/hf
REVISION := 7ae557604adf67be50417f59c2c2f167def9a775
LLAMA_COMMIT := 6c73b3e12dc501de35fe5f6979960d06921a2f6c
MODEL ?= $(HF_HOME)/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/$(REVISION)
QUANT ?= $(CACHE)/cpu-decode/int8
LLAMA_ROOT := $(CACHE)/llama.cpp/$(LLAMA_COMMIT)
LLAMA_BIN := $(LLAMA_ROOT)/build/bin/llama-bench
GGUF := $(LLAMA_ROOT)/qwen-$(REVISION)-q8_0.gguf
HEAVY := $(if $(wildcard $(PP_RUN)),$(PP_RUN) heavy,nice -n 19)
BENCH := nice -n 19 $(PP_RUN) bench
export HF_HOME

.PHONY: build prepare test correctness model-test measure

build:
	$(HEAVY) cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	$(HEAVY) cmake --build build -j4

prepare: build
	$(HEAVY) uv sync --locked
	$(HEAVY) uv run python -m tools.download_model --output results/model-manifest.json
	PP_MEM=2000M $(HEAVY) uv run python -m tools.quantize --source $(MODEL) --output $(QUANT) --manifest results/quantized-manifest.json
	PP_MEM=2000M $(HEAVY) uv run python -m tools.prepare_llama --model $(MODEL) --jobs 4 --output results/llama-preparation.json
	$(MAKE) correctness

correctness:
	PP_MEM=2000M $(HEAVY) uv run python -m tools.reference --model $(MODEL) --quant-model $(QUANT) --engine build/cpu-decode --kernel simd512 --output results/correctness.json

model-test:
	CPU_DECODE_MODEL=$(MODEL) CPU_DECODE_QUANT_MODEL=$(QUANT) CPU_DECODE_KERNEL=simd512 PP_MEM=2000M $(HEAVY) uv run pytest -q tests/test_reference.py

test:
	nice -n 19 ctest --test-dir build --output-on-failure
	nice -n 19 uv run pytest -q tests

# Separate windows leave the shared machine available between timing slices.
measure:
	$(BENCH) uv run python -m tools.measure bandwidth --kernels simd256,simd512
	@for context in 128 1024 4096; do \
	  $(BENCH) uv run python -m tools.measure engine --model $(QUANT) --contexts $$context || exit $$?; \
	  $(BENCH) uv run python -m tools.measure llama --model $(GGUF) --llama $(LLAMA_BIN) --contexts $$context || exit $$?; \
	  $(BENCH) uv run python -m tools.measure eager --model $(MODEL) --contexts $$context || exit $$?; \
	done
	$(BENCH) uv run python -m tools.measure engine --model $(QUANT) --threads 6 --contexts 128 --kernels scalar,simd256,simd512 --output results/ablations/int8-cached
	$(BENCH) uv run python -m tools.measure engine --model $(MODEL) --threads 6 --contexts 128 --kernels scalar --output results/ablations/bf16-cached
	$(BENCH) uv run python -m tools.measure engine --model $(QUANT) --threads 6 --contexts 128 --kernels simd512 --rope direct --output results/ablations/int8-direct
	nice -n 19 uv run python -m tools.summarize
