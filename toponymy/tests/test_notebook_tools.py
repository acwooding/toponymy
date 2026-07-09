import os
from pathlib import Path

import nbformat
import pytest

from nbformat.v4 import new_notebook, new_code_cell

from toponymy.tools.notebook_test_helpers import (
    get_test_ollama_model,
    notebook_test_replacement,
)
from toponymy.tools.notebook_data_load import (
    load_small_newsgroups,
    load_newsgroups,
    load_small_bundled_arxiv,
    load_bundled_arxiv,
    notebook_output_dir,
)
from toponymy.tools.notebook_runner import (
    _inject_logging_capture_cell,
    run_notebook,
    collect_log_lines,
)
from toponymy.llm_wrappers import OpenAINamer, OllamaNamer, LiteLLMNamer


def test_notebook_test_replacement_decorator(notebook_testing_env, monkeypatch):
    """Decorator should replace the wrapped function when NOTEBOOK_TESTING=true and pass through otherwise."""

    def replacement(a, b=2):
        return f"replaced:{a}:{b}"

    @notebook_test_replacement(replacement)
    def original(a, b=2):
        return f"original:{a}:{b}"

    assert original(1, b=3) == "replaced:1:3"

    monkeypatch.setenv("NOTEBOOK_TESTING", "false")
    assert original(1, b=3) == "original:1:3"


# Test local data loading functions to ensure they return expected sizes and shapes, and that the decorator replacement works as intended.
# Avoid testing the datasets that require network access
def test_load_small_newsgroups_size():
    """Ensure the bundled small newsgroups dataset loads and has expected size."""
    df = load_small_newsgroups()
    assert hasattr(df, "shape")
    assert df.shape[0] == 150


def test_load_newsgroups_decorator_behaviour(notebook_testing_env):
    """When NOTEBOOK_TESTING is set, load_newsgroups should be replaced with the small loader unless override is specified."""
    df = load_newsgroups()
    assert len(df) == 150


def test_load_bundled_arxiv_small_size_and_shapes():
    """Verify small bundled arXiv loader returns expected lengths and array shapes."""
    docs, doc_vectors, cluster_vectors = load_small_bundled_arxiv()
    assert len(docs) == 350
    assert getattr(doc_vectors, "shape", (0,))[0] == 350
    assert getattr(cluster_vectors, "shape", (0,))[0] == 350


def test_load_bundled_arxiv_decorator_behaviour(notebook_testing_env):
    """When NOTEBOOK_TESTING is set, load_bundled_arxiv should be replaced with the small loader unless override is specified."""
    docs, _, _ = load_bundled_arxiv()
    assert len(docs) == 350
    docs, _, _ = load_bundled_arxiv(use_small=False)
    assert len(docs) == 10000


def test_inject_logging_cell_and_run(tmp_path):
    """Inject logging capture cell, write a simple notebook that logs, and run it to collect logs."""
    nb = new_notebook(
        cells=[new_code_cell("import logging\nlogging.warning('from test')")]
    )
    _inject_logging_capture_cell(nb)
    assert nb.cells[0].source.strip().startswith("import sys")

    path = tmp_path / "logging_capture.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)

    lines = run_notebook(str(path), timeout=30, return_log_lines=True)
    assert any("from test" in text for level, text in lines)


def test_collect_log_lines_from_stream_output():
    nb = new_notebook(cells=[new_code_cell("print('WARNING: hello from notebook')")])
    nb.cells[0].outputs = [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": "WARNING: hello from notebook\n",
        }
    ]

    assert collect_log_lines(nb) == [("warning", "WARNING: hello from notebook")]


def test_run_notebook_captures_logger_output_on_success(tmp_path):
    path = tmp_path / "logging_capture.ipynb"
    nb = new_notebook(
        cells=[
            new_code_cell(
                "import logging\nlogging.warning('hello from logger')\nlogging.info('info from logger')"
            )
        ]
    )
    with open(path, "w") as f:
        import nbformat

        nbformat.write(nb, f)

    lines = run_notebook(str(path), timeout=30, return_log_lines=True)
    assert any(
        level == "warning" and "hello from logger" in line for level, line in lines
    )
    assert any(level == "info" and "info from logger" in line for level, line in lines)


def test_run_notebook_ignores_litellm_output(tmp_path):
    path = tmp_path / "litellm_ignore.ipynb"
    nb = new_notebook(
        cells=[
            new_code_cell(
                "import logging\nlogging.warning('LiteLLM:WARNING: silly warning from provider')"
            )
        ]
    )
    with open(path, "w") as f:
        import nbformat

        nbformat.write(nb, f)

    lines = run_notebook(
        str(path),
        timeout=30,
        return_log_lines=True,
        ignore_litellm=True,
    )
    assert not lines


def test_run_notebook_captures_logs_on_failure(tmp_path):
    """Regression test: logs should be captured even when notebook fails to complete."""
    path = tmp_path / "failing_notebook.ipynb"
    nb = new_notebook(
        cells=[
            new_code_cell(
                "import logging\nlogging.warning('warning before failure')\nlogging.info('info before failure')"
            ),
            new_code_cell("raise ValueError('notebook execution failed')"),
        ]
    )
    with open(path, "w") as f:
        import nbformat

        nbformat.write(nb, f)

    lines = run_notebook(str(path), timeout=30, return_log_lines=True)
    # Verify that logs from before the failure are still captured
    assert any(
        level == "warning" and "warning before failure" in line for level, line in lines
    )
    assert any(
        level == "info" and "info before failure" in line for level, line in lines
    )


def test_openainamer_fallback_to_notebook_mock(notebook_testing_env, monkeypatch):
    """When NOTEBOOK_TESTING=true, OpenAINamer should fallback to NotebookOpenAINamerMock which returns OllamaNamer."""
    # test fallback case
    namer = OpenAINamer()
    assert isinstance(namer, LiteLLMNamer)
    # Verify it uses the Ollama model (from the mock)
    assert get_test_ollama_model() in namer.model.lower()

    # Test the non-fallback case
    monkeypatch.setenv("NOTEBOOK_TESTING", "false")
    namer = OpenAINamer()
    # Without the env var, should use openai model
    assert isinstance(namer, LiteLLMNamer)
    assert "openai" in namer.model.lower()


def test_notebook_testing_env_fixture_sets_vars(notebook_testing_env):
    """Verify notebook_testing_env fixture sets the required environment variables."""
    assert os.environ.get("NOTEBOOK_TESTING") == "true"
    assert os.environ.get("OPENAI_API_KEY") == "notarealkey"
    assert "NB_TEST_OUTPUT_DIR" in os.environ


def test_notebook_output_dir_uses_pytest_fixture_env_var(notebook_testing_env):
    """When NB_TEST_OUTPUT_DIR is set by pytest fixture, notebook_output_dir should return that directory."""
    nb_output_dir = notebook_output_dir()

    assert isinstance(nb_output_dir, Path)
    assert nb_output_dir.exists()
    assert str(nb_output_dir) == os.environ.get("NB_TEST_OUTPUT_DIR")


def test_notebook_output_dir_manual_fallback(monkeypatch):
    """When NOTEBOOK_TESTING=true but NB_TEST_OUTPUT_DIR is unset, fallback creates a temp directory."""
    monkeypatch.setenv("NOTEBOOK_TESTING", "true")
    monkeypatch.delenv("NB_TEST_OUTPUT_DIR", raising=False)

    nb_output_dir = notebook_output_dir()

    assert isinstance(nb_output_dir, Path)
    assert nb_output_dir.exists()
    assert nb_output_dir != Path().resolve()


def test_notebook_output_dir_normal_mode(monkeypatch):
    """When NOTEBOOK_TESTING is false, notebook_output_dir returns the current working directory."""
    monkeypatch.setenv("NOTEBOOK_TESTING", "false")

    nb_output_dir = notebook_output_dir()

    assert isinstance(nb_output_dir, Path)
    assert nb_output_dir == Path().resolve()
