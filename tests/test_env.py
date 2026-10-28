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
    assert record["package_root"] == str(env.package_root())
    assert env.is_checkout(env.package_root())  # the tests run from a checkout


def test_matmul_precision(monkeypatch):
    matmul = env.collect()["torch"]["matmul"]
    assert matmul == {
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "allow_bf16_reduced_precision_reduction_split_k": getattr(
            torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction_split_k", None
        ),
        "allow_fp16_reduced_precision_reduction_split_k": getattr(
            torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction_split_k", None
        ),
        "fp32_precision": getattr(torch.backends.cuda.matmul, "fp32_precision", None),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    # As the process has them when the record is made, not the defaults.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False)
    torch.set_float32_matmul_precision("high")
    try:
        matmul = env.torch_info()["matmul"]
    finally:
        torch.set_float32_matmul_precision("highest")
    assert matmul["allow_bf16_reduced_precision_reduction"] is False
    assert matmul["float32_matmul_precision"] == "high"
    # A setting this torch lacks is None, not an error.
    monkeypatch.setattr(env, "MATMUL_FLAGS", ("allow_tf32", "no_such_flag"))
    assert env.matmul_precision()["no_such_flag"] is None


def test_matmul_precision_pairs_and_no_pairs(monkeypatch):
    import fake_matmul

    flags = fake_matmul.PairFlags()
    flags.allow_bf16_reduced_precision_reduction = (False, True)
    flags.allow_fp16_reduced_precision_reduction = (True, False)
    monkeypatch.setattr(torch.backends.cuda, "matmul", flags)
    matmul = env.matmul_precision()
    assert matmul["allow_bf16_reduced_precision_reduction"] is False
    assert matmul["allow_bf16_reduced_precision_reduction_split_k"] is True
    assert matmul["allow_fp16_reduced_precision_reduction"] is True
    assert matmul["allow_fp16_reduced_precision_reduction_split_k"] is False
    assert matmul["fp32_precision"] == "ieee"
    # A build without pairs: the split-K fields are None, the flags as they are.
    monkeypatch.setattr(torch.backends.cuda, "matmul", fake_matmul.old_flags())
    matmul = env.matmul_precision()
    assert matmul["allow_bf16_reduced_precision_reduction"] is True
    assert matmul["allow_bf16_reduced_precision_reduction_split_k"] is None
    assert matmul["allow_fp16_reduced_precision_reduction_split_k"] is None
    assert matmul["fp32_precision"] is None
    assert json.loads(json.dumps(matmul)) == matmul


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
    info = env.torch_info()
    assert info["version"] is None and info["matmul"] is None


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


def test_collect_without_torch_info():
    assert env.collect(include_torch=False)["torch"] is None


def _fake_dist(root, name, direct_url=None):
    """A ``NAME-1.2.3.dist-info`` directory under ``root``, as pip installs it."""
    info = root / f"{name.replace('-', '_')}-1.2.3.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.2.3\n")
    if direct_url is not None:
        text = direct_url if isinstance(direct_url, str) else json.dumps(direct_url)
        (info / "direct_url.json").write_text(text)


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_package_versions_vcs_origin(tmp_path, monkeypatch):
    vcs = {
        "url": "https://user:token@github.com/org/pkg-git",
        "vcs_info": {"vcs": "git", "commit_id": COMMIT, "requested_revision": "main"},
    }
    _fake_dist(tmp_path, "pkg-git", vcs)
    no_request = {"url": "https://github.com/org/pkg-pin", "vcs_info": {"commit_id": COMMIT}}
    _fake_dist(tmp_path, "pkg-pin", no_request)
    _fake_dist(tmp_path, "pkg-plain")
    _fake_dist(tmp_path, "pkg-dir", {"url": "file:///src/pkg", "dir_info": {"editable": True}})
    archive = {"url": "https://x.org/pkg.tar.gz", "archive_info": {"hashes": {}}}
    _fake_dist(tmp_path, "pkg-archive", archive)
    _fake_dist(tmp_path, "pkg-broken", "{not json")
    _fake_dist(tmp_path, "pkg-odd", {"url": 1, "vcs_info": {"commit_id": COMMIT}})
    monkeypatch.syspath_prepend(str(tmp_path))
    versions = env.package_versions(
        ["pkg-git", "pkg-pin", "pkg-plain", "pkg-dir", "pkg-archive", "pkg-broken", "pkg-odd"]
    )
    assert versions == {
        "pkg-git": {
            "version": "1.2.3",
            "url": "https://github.com/org/pkg-git",
            "commit": COMMIT,
            "requested": "main",
        },
        "pkg-pin": {
            "version": "1.2.3",
            "url": "https://github.com/org/pkg-pin",
            "commit": COMMIT,
            "requested": None,
        },
        "pkg-plain": "1.2.3",
        "pkg-dir": "1.2.3",
        "pkg-archive": "1.2.3",
        "pkg-broken": "1.2.3",
        "pkg-odd": "1.2.3",
    }
    assert json.loads(json.dumps(versions)) == versions


def test_without_credentials():
    assert env._without_credentials("https://github.com/a/b") == "https://github.com/a/b"
    assert env._without_credentials("https://u:p@h.org:8443/a") == "https://h.org:8443/a"
    assert env._without_credentials("ssh://git@h.org/a.git") == "ssh://h.org/a.git"
