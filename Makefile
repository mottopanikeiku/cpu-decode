CACHE ?= external
RESULTS ?= results/v2/reproduction
RAW ?= $(CACHE)/quality-v2-$(shell uv run python -c 'import hashlib, sys; from pathlib import Path; print(hashlib.sha256(str(Path(sys.argv[1]).resolve()).encode()).hexdigest())' "$(RESULTS)")
REVISION := 7ae557604adf67be50417f59c2c2f167def9a775
LLAMA_COMMIT := 6c73b3e12dc501de35fe5f6979960d06921a2f6c
HF_HUB_CACHE = $(shell uv run python -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)')
MODEL ?= $(HF_HUB_CACHE)/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/$(REVISION)
ROW_QUANT := $(CACHE)/int8-row
LLAMA_ROOT := $(CACHE)/llama.cpp/$(LLAMA_COMMIT)
LLAMA_BIN := $(LLAMA_ROOT)/build/bin/llama-bench
GGUF := $(LLAMA_ROOT)/qwen-$(REVISION)-q8_0.gguf
RUN := nice -n 19
QUALITY_THREADS ?= 2
CPU_SET = $(shell $(RUN) build/cpu-decode cpus | uv run python -c 'import json,sys; print(",".join(map(str,json.load(sys.stdin)["preferred_cpu_ids"][:$(QUALITY_THREADS)])))')
SELECTED_LABEL = $(shell uv run python -c 'import json; print(json.load(open("$(RESULTS)/format-selection.json"))["chosen"]["label"])')
SELECTED_MODEL = $(CACHE)/$(SELECTED_LABEL)
KERNEL ?= simd512x4
QUALITY_OPTIONS = --corpus $(RESULTS)/corpus.json --raw-dir $(RAW) --selection $(RESULTS)/format-selection.json --threads $(QUALITY_THREADS) --cpu-set $(CPU_SET) --affinity strict --attention blocked --scheduler pool
FINAL_FORMAT_SELECTION ?= $(RESULTS)/format-selection.json
FINAL_LABEL = $(shell $(RUN) uv run python -c 'import json; print(json.load(open("$(FINAL_FORMAT_SELECTION)"))["chosen"]["label"])')
FINAL_MODEL ?= $(CACHE)/$(FINAL_LABEL)
FINAL_MANIFEST ?= $(RESULTS)/quantized-$(FINAL_LABEL).json
FINAL_QUALITY ?= $(RESULTS)/quality.json
FINAL_OUTPUT ?= $(RESULTS)/final
FINAL_LLAMA_BIN ?= $(LLAMA_BIN)
FINAL_GGUF ?= $(GGUF)
FINAL_PREPARATION ?= $(RESULTS)/llama-preparation.json
FINAL_KERNEL ?= $(KERNEL)
FINAL_AFFINITY ?= strict
FINAL_CPU_ORDER ?=
FINAL_WRAPPER ?=
export FINAL_WRAPPER
FINAL_OPTIONS ?=

.PHONY: build prepare oracle calibration quality final test model-test

build:
	$(RUN) uv sync --locked
	$(RUN) cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPU_DECODE_NATIVE=ON
	$(RUN) cmake --build build -j4

prepare: build
	$(RUN) uv run python -m tools.download_model --output $(RESULTS)/model-manifest.json
	$(RUN) uv run python -m tools.quantize --source $(MODEL) --output $(ROW_QUANT) --group-size 0 --scale-dtype f32 --manifest $(RESULTS)/row-manifest.json
	@for format in 32:f16 64:f32 64:f16 128:f16; do \
	  group=$${format%:*}; scale=$${format#*:}; label=g$$group$$scale; \
	  $(RUN) uv run python -m tools.quantize --source $(MODEL) --output $(CACHE)/$$label --group-size $$group --scale-dtype $$scale --manifest $(RESULTS)/quantized-$$label.json || exit $$?; \
	done
	$(RUN) uv run python -m tools.prepare_llama --model $(MODEL) --cache $(CACHE) --jobs 4 --output $(RESULTS)/llama-preparation.json
	$(RUN) cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPU_DECODE_NATIVE=ON -DCPU_DECODE_LLAMA_ROOT="$(LLAMA_ROOT)"
	$(RUN) cmake --build build --target llama-logits -j4
	$(RUN) uv run python -m tools.quality_v2 prepare --model $(MODEL) --output-dir $(RESULTS) --raw-dir $(RAW)/sources

oracle:
	$(RUN) uv run python -m tools.quality_v2 oracle --model $(MODEL) --corpus $(RESULTS)/corpus.json --raw-dir $(RAW) --output $(RESULTS)/oracle-summary.json

calibration: oracle
	@for label in g32f16 g64f32 g64f16 g128f16; do \
	  $(RUN) uv run python -m tools.quality_v2 evaluate --split calibration --model $(CACHE)/$$label --label $$label --kernel simd512x4 --kv f16 $(QUALITY_OPTIONS) --output $(RESULTS)/cal-$$label.json || exit $$?; \
	done
	$(RUN) uv run python -m tools.quality_v2 evaluate --split calibration --backend llama --model $(GGUF) --artifact-manifest $(RESULTS)/llama-preparation.json --label q8_0-calibration --kv f16 $(QUALITY_OPTIONS) --output $(RESULTS)/cal-q8_0.json
	$(RUN) uv run python -m tools.quality_v2 compare --split calibration --corpus $(RESULTS)/corpus.json --selection $(RESULTS)/format-selection.json --reports $(RESULTS)/cal-g32f16.json $(RESULTS)/cal-g64f32.json $(RESULTS)/cal-g64f16.json $(RESULTS)/cal-g128f16.json $(RESULTS)/cal-q8_0.json --output $(RESULTS)/format-calibration.json

quality: calibration
	$(RUN) uv run python -m tools.quality_v2 evaluate --split heldout --model $(ROW_QUANT) --label per-row --kernel simd512x4 --kv f16 $(QUALITY_OPTIONS) --output $(RESULTS)/heldout-row.json
	$(RUN) uv run python -m tools.quality_v2 evaluate --split heldout --model $(SELECTED_MODEL) --label grouped-f16 --kernel simd512x4 --kv f16 $(QUALITY_OPTIONS) --output $(RESULTS)/heldout-f16.json
	$(RUN) uv run python -m tools.quality_v2 evaluate --split heldout --model $(SELECTED_MODEL) --label grouped-f32 --kernel simd512x4 --kv f32 $(QUALITY_OPTIONS) --output $(RESULTS)/heldout-f32.json
	$(RUN) uv run python -m tools.quality_v2 evaluate --split heldout --model $(SELECTED_MODEL) --label grouped-vnni --kernel vnni --kv f16 $(QUALITY_OPTIONS) --output $(RESULTS)/heldout-vnni.json
	$(RUN) uv run python -m tools.quality_v2 evaluate --split heldout --backend llama --model $(GGUF) --artifact-manifest $(RESULTS)/llama-preparation.json --label q8_0 $(QUALITY_OPTIONS) --output $(RESULTS)/heldout-q8.json
	$(RUN) uv run python -m tools.quality_v2 compare --split heldout --corpus $(RESULTS)/corpus.json --selection $(RESULTS)/format-selection.json --reports $(RESULTS)/heldout-row.json $(RESULTS)/heldout-f16.json $(RESULTS)/heldout-f32.json $(RESULTS)/heldout-vnni.json $(RESULTS)/heldout-q8.json --output $(RESULTS)/quality.json


final:
	$(RUN) uv run python -m tools.run_final_v2 --model "$(FINAL_MODEL)" --model-manifest "$(FINAL_MANIFEST)" --format-selection "$(FINAL_FORMAT_SELECTION)" --engine build/cpu-decode --bandwidth build/read-bandwidth --llama "$(FINAL_LLAMA_BIN)" --gguf "$(FINAL_GGUF)" --preparation "$(FINAL_PREPARATION)" --quality "$(FINAL_QUALITY)" --kernel "$(FINAL_KERNEL)" --affinity "$(FINAL_AFFINITY)" $(if $(FINAL_CPU_ORDER),--cpu-order "$(FINAL_CPU_ORDER)") --wrapper "$$FINAL_WRAPPER" --output "$(FINAL_OUTPUT)" $(FINAL_OPTIONS)

model-test:
	CPU_DECODE_MODEL=$(MODEL) CPU_DECODE_QUANT_MODEL=$(ROW_QUANT) CPU_DECODE_KERNEL=simd512x4 $(RUN) uv run pytest -q tests/test_reference.py

test:
	$(RUN) ctest --test-dir build --output-on-failure
	$(RUN) uv run pytest -q tests
