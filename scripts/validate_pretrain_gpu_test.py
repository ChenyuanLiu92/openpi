from __future__ import annotations

import argparse
import json
import pathlib

import pytest

from . import validate_pretrain_gpu


def _resources(**overrides) -> validate_pretrain_gpu.ResourceSnapshot:
    values = {
        "load_one": 10.0,
        "cpu_psi_full_avg60": 0.1,
        "memory_current_gib": 50.0,
        "inactive_file_gib": 10.0,
        "working_set_gib": 40.0,
        "memory_limit_gib": 180.0,
        "working_set_headroom_gib": 140.0,
        "pfs_free_gib": 1000.0,
        "gpu_processes": (),
    }
    values.update(overrides)
    return validate_pretrain_gpu.ResourceSnapshot(**values)


def test_resource_gate_reports_every_failed_dimension():
    failures = validate_pretrain_gpu.resource_gate_failures(
        _resources(
            load_one=100.0,
            cpu_psi_full_avg60=2.0,
            working_set_headroom_gib=20.0,
            pfs_free_gib=10.0,
            gpu_processes=("123, GPU-test, 100",),
        ),
        validate_pretrain_gpu.GateConfig(140.0, 45.0, 1.0, 100.0),
    )

    assert len(failures) == 5
    assert any("working-set" in failure for failure in failures)
    assert any("GPUs are not exclusive" in failure for failure in failures)


def test_nvidia_snapshot_rejects_uncorrected_ecc_and_row_failure():
    xml = """<nvidia_smi_log><driver_version>580</driver_version><cuda_version>13.0</cuda_version>
    <gpu><product_name>test</product_name><uuid>GPU-1</uuid><gpu_recovery_action>None</gpu_recovery_action>
    <pci><pci_bus_id>0000:01:00.0</pci_bus_id><pci_gpu_link_info><pcie_gen><max_link_gen>5</max_link_gen>
    <current_link_gen>1</current_link_gen></pcie_gen><link_widths><max_link_width>16x</max_link_width>
    <current_link_width>16x</current_link_width></link_widths></pci_gpu_link_info></pci>
    <ecc_mode><current_ecc>Enabled</current_ecc></ecc_mode><ecc_errors><volatile><sram_correctable>0</sram_correctable>
    <dram_correctable>0</dram_correctable><sram_uncorrectable_parity>0</sram_uncorrectable_parity>
    <sram_uncorrectable_secded>0</sram_uncorrectable_secded><dram_uncorrectable>1</dram_uncorrectable>
    </volatile><unrepairable_memory>No</unrepairable_memory></ecc_errors><remapped_rows>
    <remapped_row_pending>No</remapped_row_pending><remapped_row_failure>Yes</remapped_row_failure></remapped_rows>
    <temperature><gpu_temp>40 C</gpu_temp></temperature><fb_memory_usage><used>0 MiB</used></fb_memory_usage>
    </gpu></nvidia_smi_log>"""

    snapshot = validate_pretrain_gpu.parse_nvidia_snapshot(xml, expected_gpus=1)

    assert snapshot["gpu_count"] == 1
    assert any("uncorrected ECC" in failure for failure in snapshot["failures"])
    assert any("row remapping" in failure for failure in snapshot["failures"])


def test_aggregate_baselines_uses_median_and_rejects_unstable_rounds(tmp_path: pathlib.Path):
    def write(path: pathlib.Path, bandwidth: float):
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "process_count": 1,
                    "global_device_count": 8,
                    "topology": {},
                    "results": [
                        {
                            "operation": "all_reduce",
                            "payload_mib": 16,
                            "median_seconds": 0.1,
                            "p95_seconds": 0.2,
                            "algorithm_gib_per_second": bandwidth,
                            "bus_gib_per_second": bandwidth * 1.75,
                            "rank_straggler_ratio": 1.1,
                        }
                    ],
                }
            )
        )

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write(first, 10.0)
    write(second, 11.0)
    output = tmp_path / "baseline.json"

    validate_pretrain_gpu.aggregate_baselines((first, second), output)
    assert json.loads(output.read_text())["results"][0]["algorithm_gib_per_second"] == 10.5

    write(second, 20.0)
    with pytest.raises(RuntimeError, match="round spread"):
        validate_pretrain_gpu.aggregate_baselines((first, second), output)


def test_collective_command_preserves_numa_aligned_2x4(tmp_path: pathlib.Path):
    command = validate_pretrain_gpu.build_collective_command(
        tmp_path / "config.yaml",
        topology="2x4",
        log_dir=tmp_path / "logs",
        baseline=tmp_path / "baseline.json",
        minimum_memory_headroom_gib=140,
    )

    groups = [command[index + 1] for index, value in enumerate(command) if value == "--device-group"]
    assert groups == ["0,1,2,3", "4,5,6,7"]
    assert command[command.index("--write-baseline") + 1] == str(tmp_path / "baseline.json")


def test_telemetry_requires_all_gpus_and_gen5_under_load(tmp_path: pathlib.Path):
    path = tmp_path / "telemetry.jsonl"
    rows = [
        {
            "index": str(index),
            "uuid": f"GPU-{index}",
            "temperature_c": "70",
            "power_w": "500",
            "utilization_percent": "95",
            "memory_used_mib": "75000",
            "sm_clock_mhz": "2200",
            "memory_clock_mhz": "12000",
            "pcie_gen": "4" if index == 7 else "5",
            "pcie_width": "16",
        }
        for index in range(8)
    ]
    path.write_text(json.dumps({"time": "now", "gpus": rows}) + "\n")

    summary, failures = validate_pretrain_gpu.validate_telemetry(
        (path,), expected_gpus=8, require_compute_saturation=True, require_model_allocation=True
    )

    assert len(summary["gpus"]) == 8
    assert failures == ("GPU GPU-7 PCIe generation stayed below Gen5 under load",)


def test_generated_probe_config_loads_strict_schema(tmp_path: pathlib.Path):
    from openpi.training import pretrain_config_loader

    output = tmp_path / "probe.yaml"
    validate_pretrain_gpu._write_probe_config(  # noqa: SLF001
        validate_pretrain_gpu._DEFAULT_CONFIG, output, tmp_path  # noqa: SLF001
    )

    config = pretrain_config_loader.load(output).config
    assert config.initialization.type == "random"
    assert config.distributed.diagnostics.tensor_sizes_mib == (1.0, 16.0, 64.0, 256.0, 1024.0)
    assert config.distributed.diagnostics.measure_iterations == 10


def test_dry_run_does_not_create_work_directory(tmp_path: pathlib.Path, capsys):
    work_dir = tmp_path / "new"

    result = validate_pretrain_gpu.main(["--work-dir", str(work_dir), "--dry-run"])

    assert result == 0
    assert not work_dir.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["phases"] == ["passive", "burn", "collective", "model"]
    assert len(payload["collective_round_one_commands"]) == 4


def test_burn_argument_validation():
    args = argparse.Namespace(
        expected_device_count=8,
        duration_seconds=600,
        matrix_size=8192,
        memory_gib_per_device=75,
        memory_chunk_mib=1000,
        max_throughput_spread=0.1,
    )
    # Importing the worker does not import JAX or allocate a GPU.
    from . import gpu_burn

    gpu_burn._validate_args(args)  # noqa: SLF001
