"""Per-profile HERMES_WRITE_SAFE_ROOT and cross-profile control-file write denies.

Regression: under the multiplexed gateway ``get_safe_write_roots`` read only ``os.environ``
(the launch profile's value), so a profile's own ``.env`` line was ignored in gateway turns
while ``hermes -p X`` honoured it; a profile with no line and no process value was
unrestricted; and the file tools could rewrite another profile's ``.env`` / ``config.yaml``.
"""

from __future__ import annotations

import contextvars
import os
from pathlib import Path

import pytest

import agent.file_safety as fs
from agent.secret_scope import reset_secret_scope, set_secret_scope

KEY = "HERMES_WRITE_SAFE_ROOT"


@pytest.fixture()
def layout(tmp_path, monkeypatch):
    """A root with a shared ``.env`` value and two profiles; ``a`` is active."""
    root = tmp_path / "root"
    shared, a_only = tmp_path / "shared", tmp_path / "a_only"
    for d in (shared, a_only, root / "profiles" / "a", root / "profiles" / "b"):
        d.mkdir(parents=True)
    (root / ".env").write_text(f"{KEY}={shared}\n", encoding="utf-8")
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: root / "profiles" / "a")
    monkeypatch.delenv(KEY, raising=False)
    return {"root": root, "shared": shared, "a_only": a_only}


@pytest.fixture()
def scope():
    tokens = []

    def bind(mapping):
        tokens.append(set_secret_scope(mapping))

    yield bind
    for t in reversed(tokens):
        reset_secret_scope(t)


def _denied(path: Path) -> bool:
    return fs.is_write_denied(str(path))


def _credential(path: Path) -> bool:
    return fs._classify_write_denial(str(path)) == "credential"


def test_bound_scope_value_replaces_process_value(layout, scope, monkeypatch):
    monkeypatch.setenv(KEY, str(layout["shared"]))
    scope({KEY: str(layout["a_only"])})
    assert not _denied(layout["a_only"] / "x.md")
    assert _denied(layout["shared"] / "x.md")
    assert fs.get_safe_write_roots() == {os.path.realpath(layout["a_only"])}


def test_scope_without_key_or_blank_value_gets_shared_value(layout, scope, monkeypatch):
    monkeypatch.setenv(KEY, str(layout["shared"]))
    for mapping in ({}, {KEY: ""}, {KEY: " : "}, {KEY: "relative/dir"}):
        scope(mapping)
        assert _denied(layout["a_only"] / "x.md"), mapping
        assert not _denied(layout["shared"] / "x.md"), mapping


def test_root_env_floor_when_no_scope_and_no_process_value(layout):
    """Single-profile process (kanban worker / ``hermes -p``) whose own .env has no line."""
    assert _denied(layout["a_only"] / "x.md")
    assert not _denied(layout["shared"] / "x.md")


def test_unrestricted_only_when_no_value_anywhere(layout):
    (layout["root"] / ".env").write_text("OTHER=1\n", encoding="utf-8")
    assert not _denied(layout["a_only"] / "x.md")


def test_entries_are_stripped_and_relative_dropped(layout, monkeypatch):
    monkeypatch.setenv(KEY, f" {layout['a_only']} {os.pathsep} {os.pathsep}rel")
    assert fs.get_safe_write_roots() == {os.path.realpath(layout["a_only"])}


def test_unresolvable_entry_counts_as_absent(layout, scope):
    # realpath rejects an embedded NUL; the layer must fall through, never go unrestricted.
    scope({KEY: "/x\x00y"})
    assert fs.get_safe_write_roots() == {os.path.realpath(layout["shared"])}
    assert _denied(layout["a_only"] / "x.md")


def test_copied_context_inherits_scope(layout, scope):
    scope({KEY: str(layout["a_only"])})
    ctx = contextvars.copy_context()
    assert ctx.run(lambda: _denied(layout["a_only"] / "x.md")) is False


def test_symlink_from_shared_root_into_profile_root(layout, scope):
    link = layout["shared"] / "link"
    link.symlink_to(layout["a_only"], target_is_directory=True)
    assert _denied(link / "x.md")  # unbound: shared roots only
    scope({KEY: str(layout["a_only"])})
    assert not _denied(link / "x.md")


def test_wide_root_never_unlocks_credentials(layout, scope, tmp_path):
    scope({KEY: str(tmp_path)})
    assert _credential(Path.home() / ".ssh" / "id_ed25519")
    assert _credential(layout["root"] / ".env")


@pytest.mark.parametrize("name", [".env", ".op.env", "config.yaml"])
def test_other_profile_control_files_denied(layout, name):
    target = layout["root"] / "profiles" / "b" / name
    target.write_text("x\n", encoding="utf-8")
    assert _credential(target)


def test_active_profile_config_and_root_config_not_denied_here(layout):
    # The active profile's own config.yaml and the root's are governed by other guards (#45947).
    for target in (layout["root"] / "profiles" / "a" / "config.yaml", layout["root"] / "config.yaml"):
        target.write_text("x\n", encoding="utf-8")
        assert not _credential(target)


def test_other_profile_env_denied_through_symlinked_dir(layout, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(layout["root"] / "profiles" / "b", target_is_directory=True)
    assert _credential(alias / ".env")
    # A symlinked profile dir (home lives elsewhere) is covered by its realpath too.
    real_home = tmp_path / "elsewhere"
    real_home.mkdir()
    (layout["root"] / "profiles" / "c").symlink_to(real_home, target_is_directory=True)
    assert _credential(real_home / "config.yaml")


class TestFileOperationPaths:
    """write_file, patch and move destinations all hit the deny (real ShellFileOperations)."""

    @pytest.fixture()
    def ops(self, tmp_path):
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations

        return ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))

    @pytest.fixture()
    def victim(self, layout):
        path = layout["root"] / "profiles" / "b" / ".env"
        path.write_text("SAFE=1\n", encoding="utf-8")
        return path

    def test_write_file(self, ops, victim):
        res = ops.write_file(str(victim), f"{KEY}=/\n")
        assert res.error and "protected" in res.error
        assert victim.read_text(encoding="utf-8") == "SAFE=1\n"

    def test_patch(self, ops, victim):
        res = ops.patch_replace(str(victim), "SAFE=1", f"{KEY}=/")
        assert res.error and "protected" in res.error
        assert victim.read_text(encoding="utf-8") == "SAFE=1\n"

    def test_move_destination(self, ops, victim, layout):
        src = layout["shared"] / "planted.env"
        src.write_text(f"{KEY}=/\n", encoding="utf-8")
        res = ops.move_file(str(src), str(victim))
        assert res.error and "protected" in res.error
        assert victim.read_text(encoding="utf-8") == "SAFE=1\n" and src.exists()

    def test_profile_dir_rename_denied(self, ops, victim, layout):
        """Rename out, edit .env, rename back must fail at the first step (also for profiles/)."""
        b = victim.parent
        for src, dst in ((b, layout["root"] / "b-tmp"), (b.parent, layout["root"] / "p-tmp")):
            res = ops.move_file(str(src), str(dst))
            assert res.error and "protected" in res.error, src
            assert not dst.exists()
        assert victim.read_text(encoding="utf-8") == "SAFE=1\n"
