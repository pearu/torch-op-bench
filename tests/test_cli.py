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
        "matmul_reduction": "default",
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
        ["--device", "gpu0"],
        ["--case", "layout=a|b"],
    ],
)
def test_run_bad_arguments(tests_dir, tmp_path, bad, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli(tests_dir, tmp_path, *bad)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_run_strict(tests_dir, tmp_path, capsys):
    # The worker, a subprocess, sets the flags and reads them back into the record.
    args = ["--impl", "plain", "--mode", "fwd", "--case", "M=8", "--dtype", "float32"]
    args += ["--repeat", "1", "--warmup", "0", "--matmul-reduction", "strict"]
    strict = tmp_path / "results-strict"
    assert run_cli(tests_dir, strict, *args) == 0
    (record,) = records.read(strict)
    assert record["status"] == "ok", record["traceback"]
    assert record["settings"]["matmul_reduction"] == "strict"
    matmul = record["env"]["torch"]["matmul"]
    assert matmul["allow_bf16_reduced_precision_reduction"] is False
    assert matmul["allow_fp16_reduced_precision_reduction"] is False


@pytest.mark.parametrize("results", [None, "results", "./results"])
def test_run_strict_refuses_the_default_results(tests_dir, tmp_path, monkeypatch, results, capsys):
    monkeypatch.chdir(tmp_path)
    args = ["run", "toy_op:TOY", "--pythonpath", str(tests_dir), "--matmul-reduction", "strict"]
    if results is not None:
        args += ["--results", results]
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--matmul-reduction strict needs a results directory of its own" in err
    assert "results-strict/" in err
    assert not (tmp_path / "results").exists()


def test_unknown_op(capsys):
    with pytest.raises(SystemExit):
        cli.main(["list", "toy"])
    assert "unknown op 'toy'" in capsys.readouterr().err


def test_env(capsys):
    assert cli.main(["env", "--package", "pytest,nonexistent-xyz"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["packages"]["nonexistent-xyz"] is None and record["packages"]["pytest"]


def test_env_vcs_package(tmp_path, monkeypatch, capsys):
    from test_env import COMMIT, _fake_dist

    vcs = {"url": "https://github.com/org/pkg-git", "vcs_info": {"commit_id": COMMIT}}
    _fake_dist(tmp_path, "pkg-git", vcs)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert cli.main(["env", "--package", "pkg-git"]) == 0
    assert json.loads(capsys.readouterr().out)["packages"]["pkg-git"] == {
        "version": "1.2.3",
        "url": "https://github.com/org/pkg-git",
        "commit": COMMIT,
        "requested": None,
    }


def test_list(tests_dir, capsys):
    assert cli.main(["list", "toy_op:TOY", "--pythonpath", str(tests_dir)]) == 0
    out = capsys.readouterr().out
    assert "  M (int): 8 16 32  [default 16]" in out
    assert "  layout (str): contiguous transposed  [default contiguous]" in out
    assert "dtypes: float32 float64" in out
    assert "default dtypes: float32 float64" in out
    assert "  plain  modes=fwd,fwd_bwd  env=torch  available" in out
    assert "  missing  modes=fwd,fwd_bwd  env=torch  unavailable: toy package not installed" in out
    assert (
        "  flagged  modes=fwd,fwd_bwd  env=torch  available (checked without TOB_TOY_FLAG)" in out
    )
    assert "  elsewhere  modes=fwd  env=other  not checked (environment 'other')" in out
    assert "  plain_other (control of plain)  modes=fwd,fwd_bwd  env=other  not checked" in out
    assert "  reference (reference)  modes=fwd,fwd_bwd" in out
    assert "  chunked[split=2] (variant of chunked)  modes=fwd,fwd_bwd  env=torch  available" in out
    assert "tolerances: y_rel_err<=0.0001" in out


def test_split_keeps_variant_names():
    # A comma inside a variant's brackets is part of its name.
    assert cli._split(["a,b[x=1,y=2]", "c[k=3],d"]) == ["a", "b[x=1,y=2]", "c[k=3]", "d"]
    assert cli._split(None) == [] and cli._split(["a,,b"]) == ["a", "b"]


def test_list_default_dtypes(tests_dir, capsys, monkeypatch):
    import toy_op

    monkeypatch.setattr(toy_op.TOY, "default_dtypes", ["float64"])
    assert cli.main(["list", "toy_op:TOY", "--pythonpath", str(tests_dir)]) == 0
    out = capsys.readouterr().out
    assert "dtypes: float32 float64\ndefault dtypes: float64\n" in out


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
