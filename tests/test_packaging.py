"""The core installs and runs without any optional dependency.

Two separate things are asserted, and they are easy to confuse.

The first is that the package imports and its default pipeline runs with none of
*its own* optional extras present. Note what is not in that list: transformers and
torch arrive with docling whatever this package declares (docling-core and
docling-ibm-models both require them), so blocking those would test docling's
packaging, not ours.

The second is that the two features which import transformers directly say which
extra to install when it is missing — the case a slimmer docling would create.
"""

import builtins
import pathlib
import subprocess
import sys

import pytest

# genuinely optional to this package: nothing in its dependency tree pulls them
OPTIONAL = ("sentence_transformers", "spacy", "neo4j")
# imported directly by two features, but supplied transitively by docling today
MODEL_BACKENDS = ("transformers", "torch")


@pytest.fixture
def without_optional_deps(monkeypatch):
    """Make every optional dependency unimportable for the duration of a test."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        root = name.split(".")[0]
        if root in OPTIONAL:
            raise ImportError(f"{root} is blocked by this test")
        return real_import(name, *args, **kwargs)

    for module in list(sys.modules):
        if module.split(".")[0] in OPTIONAL:
            monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded)
    yield


def test_the_package_imports_with_no_optional_dependency():
    """In a fresh interpreter, which is what a clean install actually gets.

    Deliberately a subprocess: re-importing the package in this one would leave
    every other test module holding classes from the previous import, and pydantic
    rejects an instance of a class that is no longer the class it expects.
    """
    blocker = (
        "import builtins, sys\n"
        f"BLOCKED = {OPTIONAL!r}\n"
        "real = builtins.__import__\n"
        "def guarded(name, *a, **k):\n"
        "    if name.split('.')[0] in BLOCKED:\n"
        "        raise ImportError(name)\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guarded\n"
        "import document2graph\n"
        "assert document2graph.merge_units and document2graph.graph_from_docling\n"
        "print('ok')\n"
    )
    done = subprocess.run([sys.executable, "-c", blocker], capture_output=True, text=True,
                          cwd=str(pathlib.Path(__file__).parent.parent))
    assert done.returncode == 0, done.stderr[-2000:]
    assert "ok" in done.stdout


def test_the_default_pipeline_runs_with_no_optional_dependency(without_optional_deps):
    """Structural weights, the rule-based need and supply measures, and merging."""
    from document2graph.edge_weights import EdgeContext, compute_edge_weights
    from document2graph.models import EdgeWeightConfig, NeedConfig, SupplyConfig

    ctx = EdgeContext.from_texts(
        [("h", "p"), ("p", "b")],
        {"h": "Therapie des Typ-1-Diabetes",
         "p": "Die Dosis wird angepasst:",
         "b": "zweimal taeglich"},
    )
    for config in (
        EdgeWeightConfig(metric="structural"),
        EdgeWeightConfig(metric="uniform"),
        EdgeWeightConfig(metric="similarity",
                         similarity={"backend": "bm25"}),
        EdgeWeightConfig(metric="need_supply",
                         need=NeedConfig(estimator="reference_density"),
                         supply=SupplyConfig(similarity={"backend": "bm25"})),
        EdgeWeightConfig(metric="need_supply", need=NeedConfig(estimator="syntactic"),
                         supply=SupplyConfig(similarity={"backend": "bm25"})),
    ):
        weights = compute_edge_weights(ctx, config)
        assert len(weights) == 2
        assert all(0.0 <= w <= 1.0 for w in weights.values())


@pytest.fixture
def without_model_backends(monkeypatch):
    """transformers and torch unavailable, as a slimmer docling would leave them."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in MODEL_BACKENDS:
            raise ImportError(f"{name} is blocked by this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    yield


def test_surprisal_says_which_extra_to_install(without_model_backends):
    from document2graph.edge_weights.surprisal import SurprisalScorer

    with pytest.raises(ImportError, match=r"document2graph\[surprisal\]"):
        SurprisalScorer("Qwen/Qwen2.5-0.5B")


def test_the_token_counter_says_which_extra_to_install(without_model_backends):
    from document2graph.retrieval import huggingface_token_counter

    with pytest.raises(ImportError, match=r"document2graph\[tokenizers\]"):
        huggingface_token_counter("intfloat/multilingual-e5-large")


def test_declared_runtime_dependencies_are_only_what_the_core_needs():
    """transformers and torch belong to extras, and pytest is not a runtime dep."""
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    declared = {d.split(">")[0].split("=")[0].strip().lower()
                for d in pyproject["project"]["dependencies"]}
    assert not declared & {"transformers", "torch", "pytest", "sentence-transformers", "spacy"}
    extras = pyproject["project"]["optional-dependencies"]
    # declared rather than inherited from docling, so a slimmer docling is a
    # dependency-resolution failure instead of an ImportError at run time
    assert "torch" in " ".join(extras["surprisal"])
