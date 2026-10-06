from pathlib import Path

from tools.portable import ROOT, portable


def test_nested_locations_keep_numeric_observations_and_input_unchanged():
    source = Path("/private/cache/model")
    quantized = source / "int8"
    record = {
        "command": [str(ROOT / "build/cpu-decode"), "--model", str(quantized)],
        "nested": [{"source": str(source / "weights"), "median": 12.5, "passed": True}],
        "sha256": "0123456789abcdef",
    }
    published = portable(record, {source: "$MODEL", quantized: "$INT8"})
    assert published == {
        "command": ["./build/cpu-decode", "--model", "$INT8"],
        "nested": [{"source": "$MODEL/weights", "median": 12.5, "passed": True}],
        "sha256": "0123456789abcdef",
    }
    assert record["command"][-1] == str(quantized)


def test_stderr_and_null_values_are_portable():
    assert portable(f"warning at {ROOT}/src/model.cpp") == "warning at ./src/model.cpp"
    assert portable([None, 0, False, 0.125]) == [None, 0, False, 0.125]


def test_relative_aliases_cannot_rewrite_hashes_or_command_words():
    record = {
        "sha256": "a" * 64,
        "command": ["quantize", "--model", "a", "--output", "a/int8"],
        "absolute_model": str((Path("a") / "weights").resolve()),
    }
    result = portable(record, {Path("a"): "$MODEL", Path("a/int8"): "$INT8"})
    assert result["sha256"] == record["sha256"]
    assert result["command"] == record["command"]
    assert result["absolute_model"] == "$MODEL/weights"
