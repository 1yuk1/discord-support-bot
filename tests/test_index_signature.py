"""Проверяет, что heredoc-блок подписи индекса из scripts/start.sh работает.

Извлекает Python-код между маркерами heredoc и запускает его дважды, чтобы
убедиться в детерминированности подписи.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
START_SCRIPT = REPO_ROOT / "scripts" / "start.sh"

_HEREDOC_START = "<<'PY'"
_HEREDOC_END = "\nPY\n"


def extract_signature_script() -> str:
    source = START_SCRIPT.read_text(encoding="utf-8")
    start = source.index(_HEREDOC_START) + len(_HEREDOC_START)
    end = source.index(_HEREDOC_END, start)
    return source[start:end]


def run_signature_script(script: str, cwd: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, f"скрипт подписи упал: {result.stderr}"
    return result.stdout.strip()


def test_signature_script_runs_and_is_stable():
    script = extract_signature_script()
    first = run_signature_script(script, REPO_ROOT)
    second = run_signature_script(script, REPO_ROOT)

    assert len(first) == 64, f"ожидался sha256, получено: {first!r}"
    assert first == second, "подпись должна быть детерминированной"


def test_signature_changes_when_knowledge_changes(tmp_path):
    """Правка базы знаний обязана менять подпись, иначе индекс не обновится."""
    script = extract_signature_script()

    workdir = tmp_path / "repo"
    (workdir / "knowledge").mkdir(parents=True)
    (workdir / "bot").mkdir()
    (workdir / "indexer.py").write_text("# stub", encoding="utf-8")
    (workdir / "settings.toml").write_text('[ai]\nembedding_model = "m"\n', encoding="utf-8")

    knowledge_file = workdir / "knowledge" / "a.json"
    knowledge_file.write_text("[]", encoding="utf-8")
    before = run_signature_script(script, workdir)

    knowledge_file.write_text('[{"id": "x"}]', encoding="utf-8")
    after = run_signature_script(script, workdir)

    assert before != after


def test_signature_changes_when_embedding_model_changes(tmp_path):
    """Смена модели должна вызывать переиндексацию: векторы несовместимы."""
    script = extract_signature_script()

    workdir = tmp_path / "repo"
    (workdir / "knowledge").mkdir(parents=True)
    settings_file = workdir / "settings.toml"

    settings_file.write_text('[ai]\nembedding_model = "model-a"\n', encoding="utf-8")
    before = run_signature_script(script, workdir)

    settings_file.write_text('[ai]\nembedding_model = "model-b"\n', encoding="utf-8")
    after = run_signature_script(script, workdir)

    assert before != after


def test_signature_ignores_unrelated_bot_code(tmp_path):
    """Правка handlers.py не должна тянуть переиндексацию с загрузкой модели."""
    script = extract_signature_script()

    workdir = tmp_path / "repo"
    bot_dir = workdir / "bot"
    bot_dir.mkdir(parents=True)
    (workdir / "knowledge").mkdir()
    (workdir / "settings.toml").write_text('[ai]\nembedding_model = "m"\n', encoding="utf-8")

    handlers = bot_dir / "handlers.py"
    handlers.write_text("# version 1", encoding="utf-8")
    before = run_signature_script(script, workdir)

    handlers.write_text("# version 2, много изменений", encoding="utf-8")
    after = run_signature_script(script, workdir)

    assert before == after
