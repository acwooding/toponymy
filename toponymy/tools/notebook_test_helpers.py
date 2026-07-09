import functools
from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

# Ollama test models
OLLAMA_CI_MODEL = "qwen2.5:0.5b"
OLLAMA_LOCAL_FAMILY = "llama3.2"


def doc_dir() -> Path:
    return PACKAGE_ROOT / "doc"


def examples_dir() -> Path:
    return PACKAGE_ROOT / "examples"


def get_notebooks(doc_dir: Path) -> list[Path]:
    """
    Retrieve all non-test notebook files from a directory.

    Recursively searches for .ipynb files and excludes those with "xxx" in the filename.

    Parameters
    ----------
    doc_dir : Path
        Path to the directory containing notebook files.

    Returns
    -------
    list[Path]
        Sorted list of Path objects for .ipynb notebooks excluding test files (those with "xxx" in the name).
    """
    return sorted(p for p in Path(doc_dir).glob("*.ipynb") if "xxx" not in p.name)


def get_test_ollama_model() -> str:
    """
    Get the Ollama model name appropriate for the current testing environment.

    Returns the CI-specific model when running in CI (faster, smaller model);
    otherwise returns the local model family for development/testing.

    Returns
    -------
    str
        Ollama model identifier: OLLAMA_CI_MODEL if in CI, else OLLAMA_LOCAL_FAMILY.
    """
    if os.getenv("CI", "").lower() in {"true", "1"}:
        return OLLAMA_CI_MODEL
    return OLLAMA_LOCAL_FAMILY


def notebook_test_replacement(replacement):
    """
    Decorator for replacing function implementations during example notebook testing.

    When the NOTEBOOK_TESTING environment variable is set to "true", the decorated function
    is replaced with the provided replacement function. This allows lightweight mocking
    during automated notebook runs.

    Parameters
    ----------
    replacement : callable
        The replacement function to call when NOTEBOOK_TESTING is enabled.

    Returns
    -------
    callable
        A decorator that wraps the target function and applies the replacement conditionally.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.getenv("NOTEBOOK_TESTING", "").lower() == "true":
                return replacement(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorator
