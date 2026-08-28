
# Contributing

Contributions of all kinds are welcome. In particular pull requests are appreciated. 
The authors will endeavour to help walk you through any issues in the pull request
discussion, so please feel free to open a pull request even if you are new to such things.

## Issues

The easiest contribution to make is to [file an issue](https://github.com/TutteInstitute/topicnaming/issues/new).
It is beneficial if you check the [FAQ](https://datamapplot.readthedocs.io/en/latest/faq.html), 
and do a cursory search of [existing issues](https://github.com/TutteInstitute/topicnaming/issues?utf8=%E2%9C%93&q=is%3Aissue).
It is also helpful, but not necessary, if you can provide clear instruction for 
how to reproduce a problem. If you have resolved an issue yourself please consider
contributing to the FAQ to add your problem, and its resolution, so others can
benefit from your work.

## Documentation

Contributing to documentation is the easiest way to get started. Providing simple
clear or helpful documentation for new users is critical. Anything that *you* as 
a new user found hard to understand, or difficult to work out, are excellent places
to begin. Contributions to more detailed and descriptive error messages is
especially appreciated. To contribute to the documentation please 
[fork the project](https://github.com/TutteInstitute/topicnaming/issues#fork-destination-box)
into your own repository, make changes there, and then submit a pull request.

### Building the Documentation Locally

To build the docs locally, install the documentation tools requirements:

```bash
pip install -r doc/requirements.txt
```

Then run:

```bash
sphinx-build -b html doc doc/_build
```

This will build the documentation in HTML format. You will be able to find the output
in the `doc/_build` folder.

## Code

Code contributions are always welcome, from simple bug fixes, to new features. To
contribute code please 
[fork the project](https://github.com/TutteInstitute/topicnameing/issues#fork-destination-box)
into your own repository, make changes there, and then submit a pull request. If
you are fixing a known issue please add the issue number to the PR message. If you
are fixing a new issue feel free to file an issue and then reference it in the PR.
You can [browse open issues](https://github.com/TutteInstitute/topicnameing/issues).

### Code formatting

This project uses [black](https://github.com/python/black) version **26.5.1** for code formatting. 
All code contributions must be formatted with black before submitting a pull request.

**Option 1: Using uvx (no installation required):**

```bash
uvx black==26.5.1 toponymy/ doc/
```

**Option 2: Using pip:**

```bash
pip install black==26.5.1
black toponymy/ doc/
```

**Option 3: Using the development environment:**

```bash
uv sync --extra dev
uv run black toponymy/ doc/
```

**Pre-commit hooks (optional):** If you'd like automatic formatting on commit, you can use pre-commit:

```bash
pip install pre-commit
pre-commit install
```

The CI system will automatically check that code is properly formatted with black 26.5.1. 
If the check fails, you'll see which files need formatting and can use any of the methods above to fix them.

### Running the Tests

Toponymy uses `pytest`. The tests live under `toponymy/tests/`.

Install the project dependencies from the repo root:
```shell
pip install --upgrade uv
uv sync --extra dev
```
Run all tests:
```shell
uv run pytest toponymy/tests -v
```
Run a specific test file:
```shell
uv run pytest toponymy/tests/test_toponymy.py -v
```
Run tests with coverage:
```shell
uv run pytest toponymy/tests --show-capture=no -v --disable-warnings \
  --junitxml=junit/test-results.xml \
  --cov=toponymy/ --cov-report=xml --cov-report=html
```

