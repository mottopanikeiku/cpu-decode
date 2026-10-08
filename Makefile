RUN := nice -n 19
CACHE ?= external
QUANT_DIR ?= external
REVISION := 7ae557604adf67be50417f59c2c2f167def9a775
LLAMA_COMMIT := 6c73b3e12dc501de35fe5f6979960d06921a2f6c
HF_HUB_CACHE = $(shell uv run python -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)')
MODEL ?= $(HF_HUB_CACHE)/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/$(REVISION)
LLAMA_ROOT := $(CACHE)/llama.cpp/$(LLAMA_COMMIT)
LLAMA_BIN := $(LLAMA_ROOT)/build/bin/llama-bench
GGUF_Q8 := $(LLAMA_ROOT)/qwen-$(REVISION)-q8_0.gguf
GGUF_Q4 := $(LLAMA_ROOT)/qwen-$(REVISION)-q4_0.gguf
QUANT_MODELS := --quant-model q8=$(QUANT_DIR)/q8 --quant-model q4=$(QUANT_DIR)/q4 --quant-model q4h8=$(QUANT_DIR)/q4h8
ENGINE := $(RUN) uv run python -m tools.measure engine
LLAMA := $(RUN) uv run python -m tools.measure llama --llama $(LLAMA_BIN)
ABLATION := --threads 6 --contexts 128,4096 --output results/ablations

.PHONY: build prepare correctness llama-logits llama-quality long-context measure traffic test model-test

build:
	$(RUN) cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	$(RUN) cmake --build build -j4

prepare: build
	$(RUN) uv sync --locked
	$(RUN) uv run python -m tools.download_model --output results/model-manifest.json
	$(RUN) uv run python -m tools.quantize --source $(MODEL) --output $(QUANT_DIR)/q8 --format q8 --head-format q8 --manifest results/quantized-manifest-q8.json
	$(RUN) uv run python -m tools.quantize --source $(MODEL) --output $(QUANT_DIR)/q4 --format q4 --head-format q4 --manifest results/quantized-manifest-q4.json
	$(RUN) uv run python -m tools.quantize --source $(MODEL) --output $(QUANT_DIR)/q4h8 --format q4 --head-format q8 --manifest results/quantized-manifest-q4h8.json
	$(RUN) uv run python -m tools.prepare_llama --model $(MODEL) --cache $(CACHE) --jobs 4 --output results/llama-preparation.json
	$(MAKE) correctness

correctness:
	$(RUN) uv run python -m tools.reference --model $(MODEL) --engine build/cpu-decode $(QUANT_MODELS) --kernel auto --output results/correctness.json
	$(RUN) uv run python -m tools.summarize_quality

# Reconfigures build/ against the prepared llama.cpp library to add build/llama-logits.
llama-logits:
	$(RUN) cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPU_DECODE_LLAMA_ROOT="$(LLAMA_ROOT)"
	$(RUN) cmake --build build --target llama-logits -j4

llama-quality: llama-logits
	$(RUN) uv run python -m tools.llama_quality --model "$(GGUF_Q8)" --reader build/llama-logits --reference-dir external/reference --artifact-manifest results/llama-preparation.json --artifact Q8_0 --label q8_0 --threads 1 --output results/llama-quality.json
	$(RUN) uv run python -m tools.llama_quality --model "$(GGUF_Q4)" --reader build/llama-logits --reference-dir external/reference --artifact-manifest results/llama-preparation.json --artifact Q4_0 --label q4_0 --threads 1 --output results/llama-quality-q4_0.json

long-context: llama-logits
	$(RUN) uv run python -m tools.long_context --model $(MODEL) --engine build/cpu-decode --quant-model q8=$(QUANT_DIR)/q8 --quant-model q4h8=$(QUANT_DIR)/q4h8 --llama-reader build/llama-logits --gguf q8_0=$(GGUF_Q8) --gguf q4_0=$(GGUF_Q4) --artifact-manifest results/llama-preparation.json --output results/long-context.json

model-test:
	CPU_DECODE_MODEL=$(MODEL) CPU_DECODE_QUANT_MODEL=q8=$(QUANT_DIR)/q8,q4=$(QUANT_DIR)/q4,q4h8=$(QUANT_DIR)/q4h8 CPU_DECODE_KERNEL=auto $(RUN) uv run pytest -q tests/test_reference.py

test:
	$(RUN) ctest --test-dir build --output-on-failure
	$(RUN) uv run pytest -q tests

# Matched pairs at every thread count and context; then single-change ablations at 6 threads.
# The ablation directory carries its own q8 and q4h8 baselines so every effect is paired within one session.
measure:
	$(RUN) uv run python -m tools.measure bandwidth
	@for context in 128 1024 4096; do \
	  $(ENGINE) --model $(QUANT_DIR)/q8 --contexts $$context || exit $$?; \
	  $(LLAMA) --model $(GGUF_Q8) --label q8_0-f16 --contexts $$context || exit $$?; \
	  $(ENGINE) --model $(QUANT_DIR)/q4h8 --contexts $$context || exit $$?; \
	  $(LLAMA) --model $(GGUF_Q4) --label q4_0-f16 --contexts $$context || exit $$?; \
	done
	$(ENGINE) --model $(QUANT_DIR)/q8 --kv f32 --threads 6 --contexts 128,4096
	$(LLAMA) --model $(GGUF_Q8) --kv f32 --label q8_0-f32 --threads 6 --contexts 128,4096
	$(ENGINE) --model $(QUANT_DIR)/q8 $(ABLATION)
	$(ENGINE) --model $(QUANT_DIR)/q8 --kv f32 $(ABLATION)
	$(ENGINE) --model $(QUANT_DIR)/q8 --weights mmap --label q8-f16-mmap $(ABLATION)
	$(ENGINE) --model $(QUANT_DIR)/q8 --fuse off --label q8-f16-unfused $(ABLATION)
	$(ENGINE) --model $(QUANT_DIR)/q8 --kernels scalar --label q8-f16-scalar $(ABLATION)
	$(ENGINE) --model $(QUANT_DIR)/q4h8 $(ABLATION)
	$(ENGINE) --model $(QUANT_DIR)/q4 $(ABLATION)
	$(RUN) uv run python -m tools.summarize
	$(RUN) uv run python -m tools.plot

traffic:
	$(RUN) uv run python -m tools.traffic --llama-root $(LLAMA_ROOT)
