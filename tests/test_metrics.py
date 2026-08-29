"""Тесты модуля сбора метрик и учёта токенов AI."""

import json
from pathlib import Path
import pytest
from bot.metrics import TokenMetricsTracker


def test_token_metrics_recording(tmp_path: Path):
    metrics_file = tmp_path / "metrics.json"
    tracker = TokenMetricsTracker(filepath=metrics_file)

    # Записываем вызов для ответа
    tracker.record_usage(
        provider="openrouter",
        model="qwen/qwen3.7-plus",
        task="reply",
        prompt_tokens=150,
        completion_tokens=50,
        total_tokens=200,
    )

    # Записываем вызов для напоминания
    tracker.record_usage(
        provider="fallback",
        model="gemini-3.7-flash",
        task="reminder",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )

    summary = tracker.get_summary()

    # Тоталы
    assert summary["total_requests"] == 2
    assert summary["total_prompt_tokens"] == 250
    assert summary["total_completion_tokens"] == 70
    assert summary["total_tokens"] == 320

    # По провайдерам
    assert "openrouter" in summary["by_provider"]
    assert summary["by_provider"]["openrouter"]["requests"] == 1
    assert summary["by_provider"]["openrouter"]["total_tokens"] == 200

    assert "fallback" in summary["by_provider"]
    assert summary["by_provider"]["fallback"]["requests"] == 1
    assert summary["by_provider"]["fallback"]["total_tokens"] == 120

    # По моделям
    assert "qwen/qwen3.7-plus" in summary["by_model"]
    assert summary["by_model"]["qwen/qwen3.7-plus"]["total_tokens"] == 200

    assert "gemini-3.7-flash" in summary["by_model"]
    assert summary["by_model"]["gemini-3.7-flash"]["total_tokens"] == 120

    # По задачам
    assert summary["by_task"]["reply"]["total_tokens"] == 200
    assert summary["by_task"]["reminder"]["total_tokens"] == 120

    # Сохранение на диск
    tracker.save(force=True)
    assert metrics_file.exists()

    # Загрузка в новый трекер
    tracker2 = TokenMetricsTracker(filepath=metrics_file)
    summary2 = tracker2.get_summary()
    assert summary2["total_requests"] == 2
    assert summary2["total_tokens"] == 320


def test_token_metrics_format_report(tmp_path: Path):
    metrics_file = tmp_path / "metrics.json"
    tracker = TokenMetricsTracker(filepath=metrics_file)

    tracker.record_usage(
        provider="openrouter",
        model="qwen/qwen3.7-plus",
        task="reply",
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
    )

    report = tracker.format_report()
    assert "Статистика расхода токенов AI" in report
    assert "Сегодня" in report
    assert "За всё время" in report
    assert "1,500" in report
    assert "qwen/qwen3.7-plus" in report
    assert "Ответы в тикетах" in report
