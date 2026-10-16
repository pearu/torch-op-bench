import json
import subprocess

import torch

from torch_op_bench import env


def test_collect_is_json_and_complete():
    record = env.collect(["pytest", "no-such-package-xyz"])
    assert json.loads(json.dumps(record)) == record
    assert record["torch"]["version"] == torch.__version__
    assert record["torch"]["git_version"] == torch.version.git_version
    assert record["packages"] == {
        "pytest": record["packages"]["pytest"],
        "no-such-package-xyz": None,
    }
    assert record["packages"]["pytest"]
    assert record["timestamp"].endswith("+00:00")
    assert set(record["torch_op_bench"]) == {"version", "hash", "dirty"}


def test_torch_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = env.torch_info()
    assert info["devices"] == [] and info["cudnn"] is None
    json.dumps(env.collect())


def test_without_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert env.torch_info()["version"] is None


def test_cudnn_failure_tolerated(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    def broken():
        raise RuntimeError("no cudnn")

    monkeypatch.setattr(torch.backends.cudnn, "version", broken)
    assert env.torch_info()["cudnn"] is None


def test_git_info(tmp_path):
    assert env.git_info(tmp_path) == {"hash": None, "dirty": None}
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "f").write_text("1")
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "add", "f"], check=True)
    subprocess.run([*git, "commit", "-qm", "c"], check=True)
    info = env.git_info(tmp_path)
    assert len(info["hash"]) == 40 and info["dirty"] is False
    (tmp_path / "f").write_text("2")
    assert env.git_info(tmp_path)["dirty"] is True


def test_driver_version(monkeypatch):
    monkeypatch.setattr(env.shutil, "which", lambda name: None)
    assert env.driver_version() is None
    monkeypatch.setattr(env.shutil, "which", lambda name: "/x/nvidia-smi")
    monkeypatch.setattr(env, "_run", lambda cmd, cwd=None: "580.1\n580.1")
    assert env.driver_version() == "580.1"
    monkeypatch.setattr(env, "_run", lambda cmd, cwd=None: None)
    assert env.driver_version() is None


def test_run_failures():
    assert env._run(["no-such-command-xyz"]) is None
    assert env._run(["false"]) is None
    assert env._run(["echo", " hi "]) == "hi"
