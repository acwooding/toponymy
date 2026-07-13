import pytest
from pathlib import Path
import logging
import os

from nbformat.v4 import new_notebook, new_code_cell

from conftest import ollama_has_model, ollama_running
from toponymy.tools.notebook_runner import collect_log_lines, run_notebook
from toponymy.tools.notebook_test_helpers import (
    doc_dir,
    get_notebooks,
    get_test_ollama_model,
)

logger = logging.getLogger(__name__)

NOTEBOOK_CONFIG = {
    "basic_usage.ipynb": {
        "has_openainamer": True,
        "run_in_pr": True,
        "timeout": 1000,
    },
    "clusterers.ipynb": {
        "has_openainamer": False,
        "run_in_pr": False,
        "timeout": 3600,
    },
    "clustering_options.ipynb": {
        "has_openainamer": False,
        "run_in_pr": False,
        "timeout": 300,
    },
    "exemplar_texts.ipynb": {
        "has_openainamer": False,
        "run_in_pr": False,
        "timeout": 300,
    },
    "how_toponymy_works.ipynb": {
        "has_openainamer": True,
        "run_in_pr": False,
        "timeout": 3600,
    },
    "keyphrases.ipynb": {
        "has_openainamer": False,
        "run_in_pr": False,
        "timeout": 1000,
    },
    "saving_loading.ipynb": {
        "has_openainamer": True,
        "run_in_pr": True,
        "timeout": 1000,
    },
    "test_audit_functionality.ipynb": {
        "has_openainamer": True,
        "run_in_pr": False,
        "timeout": 600,
    },
    "test_max_layers_newsgroups.ipynb": {
        "has_openainamer": True,
        "run_in_pr": False,
        "timeout": 300,
    },
    "topic_summaries.ipynb": {
        "has_openainamer": True,
        "run_in_pr": False,
        "timeout": 600,
    },
}


def get_notebook_cfg(path: str):
    name = Path(path).name
    return NOTEBOOK_CONFIG.get(name, {"has_openainamer": True, "timeout": 6000})


TEST_NOTEBOOKS = get_notebooks(doc_dir())

CI = os.getenv("CI", "").lower() == "true"

NOTEBOOKS_WITHOUT_OPENAINAMER = [
    nb
    for nb in TEST_NOTEBOOKS
    if not get_notebook_cfg(nb).get("has_openainamer", False)
]
NOTEBOOKS_WITH_OPENAINAMER = [
    nb for nb in TEST_NOTEBOOKS if get_notebook_cfg(nb).get("has_openainamer", False)
]


# @pytest.mark.skipif(CI, reason="Skipping in CI environment")
@pytest.mark.parametrize("notebook", NOTEBOOKS_WITHOUT_OPENAINAMER)
def test_doc_notebook_no_openainamer(notebook, notebook_testing_env):
    cfg = get_notebook_cfg(notebook)

    # if not cfg.get("run_in_pr") and (os.getenv("BUILD_REASON") == "PullRequest"):
    #    pytest.skip(f"Skipped in PR CI")

    run_notebook(
        notebook,
        timeout=cfg["timeout"],
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS_WITH_OPENAINAMER)
def test_doc_notebook_has_openainamer(notebook, notebook_testing_env, ollama_running):
    cfg = get_notebook_cfg(notebook)

    # if not cfg.get("run_in_pr") and (os.getenv("BUILD_REASON") == "PullRequest"):
    #    pytest.skip(f"Skipped in PR CI")
    model = get_test_ollama_model()
    logger.info(f"ollama running:{ollama_running}")
    logger.info(f"ollama_has_model {model}:{ollama_has_model(model)}")
    if not ollama_has_model(model):
        pytest.skip(f"{model} not available in local Ollama for OpenAI mocking")
    run_notebook(
        notebook,
        timeout=cfg["timeout"],
    )
