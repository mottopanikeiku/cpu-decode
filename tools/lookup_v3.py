"""Fixed public prompts: tokenization, exact greedy comparison, acceptance counts."""
import argparse
import json
import subprocess
from pathlib import Path

from tools.download_model import MODEL_ID, REVISION, file_hash
from tools.kv_quality_v3 import save


def tokenize(model, config):
    from transformers import AutoTokenizer

    if (config["model_id"], config["revision"]) != (MODEL_ID, REVISION):
        raise ValueError("Prompt tokenizer pin differs")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    return [{**case, "tokens": tokenizer.apply_chat_template(
        [{"role": "user", "content": case["text"]}], tokenize=True, add_generation_prompt=True)}
        for case in config["prompts"]]


def measure(engine, model, case, config, threads, kernel, kv):
    common = [str(engine.resolve()), "generate", "--model", str(model.resolve()),
              "--tokens", ",".join(map(str, case["tokens"])), "--steps", str(config["steps"]),
              "--threads", str(threads), "--kernel", kernel, "--kv", kv,
              "--ngram", str(config["max_ngram"])]
    records = []
    for draft, prefill in ((0, 1), (config["lookup_draft_tokens"], config["prefill_batch"])):
        completed = subprocess.run(common + ["--lookup", str(draft), "--prefill-batch", str(prefill)],
                                   check=True, text=True, capture_output=True)
        record = json.loads(completed.stdout)
        record["model"] = "$INT8_MODEL"
        records.append(record)
    baseline, lookup = records
    return {"id": case["id"], "category": case["category"], "prompt_tokens": case["tokens"],
            "exact_match": baseline["generated_tokens"] == lookup["generated_tokens"],
            "baseline": baseline, "lookup": lookup}


def aggregate(cases):
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Expected distinct measured prompts")
    result = {}
    for category in ("all", "copy-heavy", "open-ended"):
        selected = [case for case in cases if category == "all" or case["category"] == category]
        if not selected:
            continue
        counts = {key: sum(case["lookup"]["lookup"][key] for case in selected)
                  for key in ("drafted_tokens", "accepted_tokens", "verification_batches", "forward_tokens", "ordinary_steps")}
        counts["acceptance_rate"] = counts["accepted_tokens"] / counts["drafted_tokens"] if counts["drafted_tokens"] else None
        counts["prompts"] = len(selected)
        counts["generated_tokens"] = sum(len(case["lookup"]["generated_tokens"]) for case in selected)
        counts["exact_match_prompts"] = sum(case["exact_match"] for case in selected)
        result[category] = counts
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "measure", "summarize"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    parser.add_argument("--config", type=Path, default=Path("configs/lookup-prompts.json"))
    parser.add_argument("--tokens", type=Path, default=Path("results/v3/lookup-inputs.json"))
    parser.add_argument("--output", type=Path, default=Path("results/v3/lookup-acceptance.json"))
    parser.add_argument("--cases", type=Path, default=Path("results/v3/lookup-cases"))
    parser.add_argument("--case", type=int, choices=range(12))
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--kernel", default="vnni16")
    parser.add_argument("--kv", choices=("f16", "f32", "i8", "i8-centered"), default="f16")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == "prepare":
        save(args.tokens, {"config_sha256": file_hash(args.config), "chat_template": "pinned tokenizer; one user message; add_generation_prompt=True", "prompts": tokenize(args.tokenizer, config)})
        return
    inputs = json.loads(args.tokens.read_text())
    if inputs["config_sha256"] != file_hash(args.config):
        raise ValueError("Prompts changed after tokenization")
    if args.command == "measure":
        if args.case is None:
            parser.error("measure requires --case index so each job is small")
        measured = measure(args.engine, args.model, inputs["prompts"][args.case], config, args.threads, args.kernel, args.kv)
        measured["config_sha256"] = file_hash(args.config)
        measured["inputs_sha256"] = file_hash(args.tokens)
        measured["binary_sha256"] = file_hash(args.engine)
        measured["weights_sha256"] = file_hash(args.model / "model.safetensors")
        save(args.cases / f"{measured['id']}.json", measured)
        if not measured["exact_match"]:
            raise RuntimeError(f"Greedy mismatch on {measured['id']}")
        return
    cases = [json.loads((args.cases / f"{case['id']}.json").read_text()) for case in inputs["prompts"]]
    for case, expected in zip(cases, inputs["prompts"], strict=True):
        if (case["id"] != expected["id"] or case["category"] != expected["category"] or
                case["prompt_tokens"] != expected["tokens"] or case["config_sha256"] != file_hash(args.config) or
                case["inputs_sha256"] != file_hash(args.tokens) or not case["exact_match"] or
                len(case["lookup"]["generated_tokens"]) != config["steps"]):
            raise ValueError("Measured case differs from fixed prompt suite")
    identity_keys = ("binary_sha256", "weights_sha256")
    if any(any(case[key] != cases[0][key] for key in identity_keys) for case in cases):
        raise ValueError("Generation binary or weights changed across prompts")
    save(args.output, {"config_sha256": file_hash(args.config), "inputs_sha256": file_hash(args.tokens),
                      "model_id": MODEL_ID, "revision": REVISION, "timing": "not measured",
                      "aggregate": aggregate(cases), "cases": [f"lookup-cases/{case['id']}.json" for case in cases],
                      "binary_sha256": cases[0]["binary_sha256"], "weights_sha256": cases[0]["weights_sha256"]})


if __name__ == "__main__":
    main()
