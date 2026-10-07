"""Small, model-free tests of ISA selection and completed-result durability."""
import json

import pytest

from tools import cloud_host, cloud_run


def cpu_record(index=0, flags=cloud_host.REQUIRED_FLAGS, model="test CPU"):
    return f"processor : {index}\nmodel name : {model}\nflags : {' '.join(flags)}\n"


def test_complete_flags_support_the_actual_kernel():
    value = cloud_host.parse_cpuinfo(cpu_record() + "\n" + cpu_record(1))
    assert value["supports_vnni16"]
    assert value["missing_flags"] == []
    assert value["model_name"] == "test CPU"
    assert value["parsed_cpu_count"] == 2


@pytest.mark.parametrize("missing", cloud_host.REQUIRED_FLAGS)
def test_each_required_flag_is_necessary(missing):
    flags = [flag for flag in cloud_host.REQUIRED_FLAGS if flag != missing]
    value = cloud_host.parse_cpuinfo(cpu_record(flags=flags))
    assert not value["supports_vnni16"]
    assert value["missing_flags"] == [missing]


def test_flags_are_intersection_not_union():
    flags = [flag for flag in cloud_host.REQUIRED_FLAGS if flag != "avx512_vnni"]
    value = cloud_host.parse_cpuinfo(cpu_record() + "\n" + cpu_record(1, flags))
    assert value["missing_flags"] == ["avx512_vnni"]
    assert not value["supports_vnni16"]


def test_missing_processor_flags_cannot_be_hidden():
    value = cloud_host.parse_cpuinfo(cpu_record() + "\nprocessor : 1\nmodel name : test CPU\n")
    assert value["missing_flags"] == list(cloud_host.REQUIRED_FLAGS)


@pytest.mark.parametrize("text", ["", "processor : 0\n", " \n\t\n"])
def test_missing_information_is_not_support(text):
    value = cloud_host.parse_cpuinfo(text)
    assert value["model_name"] is None
    assert not value["supports_vnni16"]


def test_heterogeneous_models_do_not_invent_one_model():
    value = cloud_host.parse_cpuinfo(cpu_record(model="CPU A") + "\n" + cpu_record(1, model="CPU B"))
    assert value["model_name"] is None
    assert value["model_names"] == ["CPU A", "CPU B"]
    assert value["supports_vnni16"]


def test_missing_model_name_is_retained_as_unknown():
    value = cloud_host.parse_cpuinfo(cpu_record() + "\nprocessor : 1\nflags : " + " ".join(cloud_host.REQUIRED_FLAGS))
    assert value["model_name"] is None
    assert value["supports_vnni16"]


def test_unsupported_container_never_prepares_or_loads_models(tmp_path, monkeypatch):
    design = tmp_path / "design.json"
    design.write_text("{}")
    monkeypatch.setattr(cloud_run, "environment", lambda: {"flags": []})
    monkeypatch.setattr(cloud_host, "snapshot", lambda: {"supports_vnni16": False})

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("unsupported host attempted model preparation")

    monkeypatch.setattr(cloud_run, "prepare", unexpected_prepare)
    with pytest.raises(ValueError, match="required vnni16 CPU flag"):
        cloud_run.compare(tmp_path, tmp_path, tmp_path / "output", design, require_vnni=True)


def test_atomic_save_replaces_complete_json_and_leaves_no_temporary(tmp_path):
    result = tmp_path / "raw.json"
    cloud_run.save(result, {"cells": [1]})
    cloud_run.save(result, {"cells": [1, 2]})
    assert json.loads(result.read_text()) == {"cells": [1, 2]}
    assert not result.with_suffix(".json.tmp").exists()


def test_invalid_next_result_does_not_destroy_completed_cells(tmp_path):
    result = tmp_path / "raw.json"
    cloud_run.save(result, {"cells": [1]})
    with pytest.raises(ValueError):
        cloud_run.save(result, {"cells": [float("nan")]})
    assert json.loads(result.read_text()) == {"cells": [1]}
