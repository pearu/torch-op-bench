import json
import sys

import pytest

from torch_op_bench import cli, records


def run_cli(tests_dir, tmp_path, *args):
    return cli.main(
        ["run", "toy_op:TOY", "--pythonpath", str(tests_dir), "--results", str(tmp_path), *args]
    )


def test_run_sweep_and_resume(tests_dir, tmp_path, capsys):
    args = ["--impl", "plain,missing", "--mode", "fwd", "--sweep", "K", "--dtype", "float32"]
    args += ["--warmup", "1", "--repeat", "2", "--seed", "3", "--timeout", "120", "--flush-l2"]
    assert run_cli(tests_dir, tmp_path, *args) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == f"4 new records (2 ok, 2 unavailable) under {tmp_path}/toy/cpu/"
    recs = records.read(tmp_path)
    assert recs[0]["settings"] == {
        "warmup": 1,
        "repeat": 2,
        "flush_l2": True,
        "seed": 3,
        "timeout": 120.0,
    }
    assert run_cli(tests_dir, tmp_path, *args) == 0
    assert "(2 unavailable)" in capsys.readouterr().out


def test_run_cases_values_env(tests_dir, tmp_path, capsys):
    args = ["--impl", "elsewhere", "--dtype", "float64", "--repeat", "1", "--warmup", "0"]
    args += ["--case", "M=5,layout=transposed", "--env", f"other={sys.executable}"]
    args += ["--device-slug", "box"]
    assert run_cli(tests_dir, tmp_path, *args) == 0
    (record,) = records.read(tmp_path / "toy" / "box")
    assert record["status"] == "ok" and record["case"]["params"]["M"] == 5
    assert record["case"]["params"]["layout"] == "transposed"

    args = ["--impl", "plain", "--mode", "fwd", "--dtype", "float32", "--repeat", "1"]
    args += ["--sweep", "M", "--values", "M=3,7"]
    assert run_cli(tests_dir, tmp_path, *args) == 0
    assert [r["case"]["params"]["M"] for r in records.read(tmp_path / "toy" / "cpu")] == [3, 7]


@pytest.mark.parametrize(
    "bad",
    [
        ["--case", "M"],
        ["--values", "M"],
        ["--impl", "nope"],
        ["--env", "x"],
        ["--dtype", "int8"],
        ["--mode", "bwd"],
        ["--sweep", "M", "--values", "K=99"],
        ["--device", "cuda", "--env", "torch=false"],
    ],
)
def test_run_bad_arguments(tests_dir, tmp_path, bad, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli(tests_dir, tmp_path, *bad)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_unknown_op(capsys):
    with pytest.raises(SystemExit):
        cli.main(["list", "toy"])
    assert "unknown op 'toy'" in capsys.readouterr().err


def test_env(capsys):
    assert cli.main(["env", "--package", "pytest,nonexistent-xyz"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["packages"]["nonexistent-xyz"] is None and record["packages"]["pytest"]


def test_list(tests_dir, capsys):
    assert cli.main(["list", "toy_op:TOY", "--pythonpath", str(tests_dir)]) == 0
    out = capsys.readouterr().out
    assert "  M (int): 8 16 32  [default 16]" in out
    assert "  layout (str): contiguous transposed  [default contiguous]" in out
    assert "dtypes: float32 float64" in out
    assert "  plain  modes=fwd,fwd_bwd  env=torch  available" in out
    assert "  missing  modes=fwd,fwd_bwd  env=torch  unavailable: toy package not installed" in out
    assert "  elsewhere  modes=fwd  env=other  not checked (environment 'other')" in out
    assert "  reference (reference)  modes=fwd,fwd_bwd" in out
    assert "tolerances: y_rel_err<=0.0001" in out


def test_module_entry_point(tests_dir):
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch_op_bench.cli",
            "list",
            "toy_op:TOY",
            "--pythonpath",
            str(tests_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0 and "op: toy" in proc.stdout


def test_parent_imports_no_torch_numpy_or_plot_dependencies():
    import subprocess

    code = (
        "import sys, torch_op_bench.cli, torch_op_bench.runner, torch_op_bench.records;"
        "bad = {'torch', 'numpy', 'pandas', 'matplotlib'} & set(sys.modules);"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
