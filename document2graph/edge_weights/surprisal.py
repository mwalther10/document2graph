"""Conditional surprisal: how much less surprising a parent makes its child.

Hand the child text to a language model and ask how surprised it is. Then show
it the parent first and ask again. If the parent made the child much less
surprising, the parent was carrying information the child needed::

    dNLL = NLL(child) - NLL(child | parent)

NLL is negative log-likelihood -- "how unexpected was this text" -- averaged per
token so length cancels out. A large positive dNLL means the parent helped a
lot; near zero means the child stood fine on its own.

This is the most general of the need estimators. It does not care what kind of
relation it is measuring: bullet to paragraph, cell to column header, paragraph
to heading, continuation to predecessor are all the same computation on one
continuous scale, which is what makes a matrix over many child types
comparable at all. The other estimators are built from relation-specific
heuristics.

**The trap.** Surprisal drops for two different reasons that look identical in
the number. Either the parent genuinely supplies missing context, or the parent
merely repeats the child's vocabulary, so its tokens become locally predictable
without anything being interpreted. A heading that just restates the
paragraph's topic words shows a large dNLL while supplying nothing.

The control is the counterfactual parent: condition on a different parent from
the same document that is not the child's ancestor. Genuine dependency
collapses under the swap; topical redundancy largely survives it, because
same-domain text shares vocabulary regardless. What this module reports is
dNLL *minus* the counterfactual dNLL, never the raw value -- set
``counterfactual="none"`` to opt out and see why that matters.
"""

from __future__ import annotations

import math
from typing import Callable

from ..models.EdgeWeightConfig import SurprisalConfig
from .context import Edge, EdgeContext

# (text, context | None) -> mean NLL per token; the seam tests inject through
NLLScorer = Callable[[str, str | None], float]

_scorer_cache: dict[tuple[str, str | None], "SurprisalScorer"] = {}


def _resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch  # only reached from SurprisalScorer, which has already checked for it

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SurprisalScorer:
    """Mean per-token NLL from a causal language model, with and without context.

    Absolute model quality matters less here than consistency: every number is
    compared against another number from the same model, so a small multilingual
    model is enough.
    """

    def __init__(self, model_name: str, device: str | None = None,
                 max_context_tokens: int = 384, max_child_tokens: int = 256):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - environment-dependent
            raise ImportError(
                "the surprisal need estimator needs a language model backend, which the "
                "core package does not install: pip install 'document2graph[surprisal]' "
                "(transformers and torch). Every other need and supply measure is "
                "rule-based or lexical and needs nothing extra."
            ) from e

        self._torch = torch
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_context_tokens = max_context_tokens
        self.max_child_tokens = max_child_tokens
        self._cache: dict[tuple[str, str | None], float] = {}

    def _encode(self, text: str, limit: int, keep: str = "head") -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= limit:
            return ids
        # a context is truncated from the front: the end of a parent is what sits
        # closest to the child and carries most of what the child picks up
        return ids[-limit:] if keep == "tail" else ids[:limit]

    def __call__(self, text: str, context: str | None = None) -> float:
        key = (text, context)
        if key not in self._cache:
            self._cache[key] = self._score(text, context)
        return self._cache[key]

    def _score(self, text: str, context: str | None) -> float:
        torch = self._torch
        child_ids = self._encode(text, self.max_child_tokens)
        if not child_ids:
            return 0.0
        context_ids: list[int] = []
        if context and context.strip():
            context_ids = self._encode(context, self.max_context_tokens, keep="tail")
            if context_ids:
                context_ids = context_ids + self._encode("\n", 4)
        bos = self.tokenizer.bos_token_id
        prefix = ([bos] if bos is not None and not context_ids else []) + context_ids
        # without a prefix the first child token would have nothing to condition
        # on and its loss would be undefined
        if not prefix:
            prefix = [child_ids[0]]
            child_ids = child_ids[1:]
            if not child_ids:
                return 0.0

        input_ids = torch.tensor([prefix + child_ids], device=self.device)
        labels = torch.full_like(input_ids, -100)
        labels[0, len(prefix):] = input_ids[0, len(prefix):]
        with torch.no_grad():
            loss = self.model(input_ids=input_ids, labels=labels).loss
        return float(loss)


def load_scorer(config: SurprisalConfig) -> SurprisalScorer:
    """A process-wide cached scorer, so a sweep loads each model once."""
    key = (config.model, config.device)
    if key not in _scorer_cache:
        _scorer_cache[key] = SurprisalScorer(
            config.model,
            device=config.device,
            max_context_tokens=config.max_context_tokens,
            max_child_tokens=config.max_child_tokens,
        )
    return _scorer_cache[key]


def delta_nll(ctx: EdgeContext, config: SurprisalConfig,
              scorer: NLLScorer | None = None) -> dict[Edge, float]:
    """Raw, counterfactual-corrected dNLL per edge, in nats. Not normalized."""
    score = scorer or load_scorer(config)
    baseline: dict[str, float] = {}
    deltas: dict[Edge, float] = {}

    for edge in ctx.edges:
        parent_id, child_id = edge
        child_text = ctx.text(child_id)
        if not child_text.strip():
            deltas[edge] = 0.0
            continue
        if child_id not in baseline:
            baseline[child_id] = score(child_text, None)
        delta = baseline[child_id] - score(child_text, ctx.text(parent_id))

        if config.counterfactual != "none":
            control_id = ctx.counterfactual_parent(edge)
            if control_id is not None:
                control_delta = baseline[child_id] - score(child_text, ctx.text(control_id))
                # subtract only the part topical redundancy can explain
                delta -= max(control_delta, 0.0)
        deltas[edge] = delta
    return deltas


def normalize(deltas: dict[Edge, float], config: SurprisalConfig) -> dict[Edge, float]:
    """Map dNLL in nats onto [0, 1].

    ``"max"`` clips at zero and divides by the largest delta in the document,
    matching the BM25 convention: the result is relative within one document.
    ``"logistic"`` squashes on an absolute scale instead, so two documents stay
    comparable even when neither contains a strongly dependent edge.
    """
    if not deltas:
        return {}
    if config.normalize == "logistic":
        scale = config.logistic_scale or 1.0
        return {edge: max(math.tanh(delta / (2 * scale)), 0.0) for edge, delta in deltas.items()}
    largest = max(deltas.values())
    if largest <= 0:
        return {edge: 0.0 for edge in deltas}
    return {edge: max(delta, 0.0) / largest for edge, delta in deltas.items()}


def surprisal_need(ctx: EdgeContext, config: SurprisalConfig,
                   scorer: NLLScorer | None = None) -> dict[Edge, float]:
    """Normalized conditional surprisal per edge, in [0, 1]."""
    return normalize(delta_nll(ctx, config, scorer=scorer), config)
