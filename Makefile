CACHE ?= external
REVISION := 7ae557604adf67be50417f59c2c2f167def9a775
LLAMA_COMMIT := 6c73b3e12dc501de35fe5f6979960d06921a2f6c
HF_HUB_CACHE = $(shell uv run python -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)')
MODEL ?= $(HF_HUB_CACHE)/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/$(REVISION)
QUANT ?= $(CACHE)/int8
LLAMA_ROOT := $(CACHE)/llama.cpp/$(LLAMA_COMMIT)
LLAMA_BIN := $(LLAMA_ROOT)/build/bin/llama-bench
GGUF := $(LLAMA_ROOT)/qwen-$(REVISION)-q8_0.gguf
RUN := nice -n 19

.PHONY: build prepare test correctness model-test measure

build:
	$(RUN) cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	$(RUN) cmake --build build -j4

prepare: build
	$(RUN) uv sync --locked
	$(RUN) uv run python -m tools.download_model --output results/model-manifest.json
	$(RUN) uv run python -m tools.quantize --source $(MODEL) --output $(QUANT) --manifest results/quantized-manifest.json
	$(RUN) uv run python -m tools.prepare_llama --model $(MODEL) --cache $(CACHE) --jobs 4 --output results/llama-preparation.json
	$(MAKE) correctness

correctness:
	$(RUN) uv run python -m tools.reference --model $(MODEL) --quant-model $(QUANT) --engine build/cpu-decode --kernel simd512x4 --output results/correctness.json
	$(RUN) uv run python -m tools.summarize_quality

.PHONY: llama-quality
llama-quality:
	$(RUN) cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPU_DECODE_LLAMA_ROOT="$(LLAMA_ROOT)"
	$(RUN) cmake --build build --target llama-logits -j4
	$(RUN) uv run python -m tools.llama_quality --model "$(GGUF)" --reader build/llama-logits --reference-dir external/reference --artifact-manifest results/llama-preparation.json --threads 1 --output results/llama-quality.json

model-test:
	CPU_DECODE_MODEL=$(MODEL) CPU_DECODE_QUANT_MODEL=$(QUANT) CPU_DECODE_KERNEL=simd512x4 $(RUN) uv run pytest -q tests/test_reference.py

test:
	$(RUN) ctest --test-dir build --output-on-failure
	$(RUN) uv run pytest -q tests

measure:
	$(RUN) uv run python -m tools.measure bandwidth --kernels simd256,simd512
	@for context in 128 1024 4096; do \
	  $(RUN) uv run python -m tools.measure engine --model $(QUANT) --contexts $$context || exit $$?; \
	  $(RUN) uv run python -m tools.measure llama --model $(GGUF) --llama $(LLAMA_BIN) --contexts $$context || exit $$?; \
	  $(RUN) uv run python -m tools.measure eager --model $(MODEL) --contexts $$context || exit $$?; \
	done
	$(RUN) uv run python -m tools.measure engine --model $(QUANT) --threads 6 --contexts 128 --kernels scalar,simd256,simd512,simd512x4 --output results/ablations/int8-cached
	$(RUN) uv run python -m tools.measure engine --model $(MODEL) --threads 6 --contexts 128 --kernels scalar --output results/ablations/bf16-cached
	$(RUN) uv run python -m tools.measure engine --model $(QUANT) --threads 6 --contexts 128 --kernels simd512x4 --rope direct --output results/ablations/int8-direct
	$(RUN) uv run python -m tools.summarize

.PHONY: traffic
traffic:
	$(RUN) uv run python -m tools.traffic --llama-root $(LLAMA_ROOT) --gguf $(GGUF)
