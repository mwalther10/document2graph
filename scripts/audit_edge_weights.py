"""What the edge weight metrics actually do to a corpus.

Reads the saved graphs -- no PDFs, no docling -- and answers the questions that
have to be settled before any retrieval run is worth doing:

1. **corpus**    what relations does this corpus even contain, and in what
                 language? The shipped supply priors were measured on German
                 guidelines; a corpus that is mostly English reports with a
                 different relation mix is not the corpus they describe.
2. **priors**    per supply relation: what the measured components say, what the
                 prior says, and how often the prior -- not the evidence -- is
                 what produced the score. A relation whose edges are nearly all
                 floor-bound has no measurement behind it, and reporting its mean
                 as one would be a mistake.
3. **evidence**  per supply relation, under the default ``"evidence"``
                 composition: how many edges evidence pushes toward merging, how
                 many it leaves at the neutral 0.5, and how many it pushes toward
                 independence -- and which term did the pushing.
4. **agreement** rank correlation between metrics. If `need_supply` orders edges
                 the way `structural` already did, every downstream condition is
                 measuring the same thing twice and the retrieval run is moot.
5. **merge**     what each metric costs in index tokens once it is actually used
                 to merge, against the merge-everything baseline it has to beat.

**This cannot produce new priors.** The shipped ones come from hand labels --
"does this parent supply the child's missing context: fully, partly, not at all"
-- and the components measured here are proxies for that judgement, not the
judgement. Flooring a proxy at a prior derived from the same proxy would be
circular. What this shows is whether the priors are *doing work* and whether they
*disagree with the evidence*, which is what tells you a relabelling is needed.

Usage:
    uv run python scripts/audit_edge_weights.py
    uv run python scripts/audit_edge_weights.py --section priors
    uv run python scripts/audit_edge_weights.py --section evidence
    uv run python scripts/audit_edge_weights.py --language de
    uv run python scripts/audit_edge_weights.py --graphs 'tests/.test-data/graphs/*.json'
"""

import argparse
import glob
import os
import statistics as st
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document2graph import load_graph, merge_units
from document2graph.edge_weights import (
    SUPPLY_RELATIONS,
    recompute_weights,
    components,
    compute_edge_weights,
    edge_context_from_graph,
    evidence_components,
    supply_relation,
)
from document2graph.edge_weights import linguistic as ling
from document2graph.models import (
    MEASURED_SUPPLY_PRIORS,
    EdgeWeightConfig,
    MergePolicy,
    NeedConfig,
    NeedSupplyConfig,
    SimilarityConfig,
    SupplyConfig,
)
from document2graph.retrieval import calibrate_min_weight, mean_unit_tokens

GRAPHS = "tests/.test-data/graphs/*_graph.json"


def build_configs(backend: str) -> dict[str, EdgeWeightConfig]:
    """One configuration per metric, sharing content settings so the sweep varies one thing."""
    similarity = SimilarityConfig(backend=backend)
    supply = SupplyConfig(similarity=similarity)
    return {
        "uniform": EdgeWeightConfig(metric="uniform"),
        "structural": EdgeWeightConfig(metric="structural"),
        "similarity": EdgeWeightConfig(metric="similarity", similarity=similarity),
        "inverted_similarity": EdgeWeightConfig(metric="inverted_similarity",
                                                similarity=similarity),
        "ns/reference_density": EdgeWeightConfig(
            metric="need_supply", supply=supply,
            need=NeedConfig(estimator="reference_density")),
        "ns/syntactic": EdgeWeightConfig(
            metric="need_supply", supply=supply, need=NeedConfig(estimator="syntactic")),
        "ns/blend": EdgeWeightConfig(
            metric="need_supply", supply=supply, need=NeedConfig(estimator="blend")),
        # the older need * prior-floored supply, for comparison with runs made under it
        "ns/blend/product": EdgeWeightConfig(
            metric="need_supply", supply=supply, need=NeedConfig(estimator="blend"),
            need_supply=NeedSupplyConfig(composition="product")),
    }


def document_language(graph) -> str:
    sample = " ".join(s.text for s in graph.snippets[:200])
    return ling.resolve_language("auto", sample)


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, ties averaged. nan when either side is constant."""
    if len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # average the ranks of ties, or a metric with few distinct values (structural)
    # correlates against an artefact of the sort order
    sorted_values = np.asarray(values, dtype=float)[order]
    start = 0
    for end in range(1, len(sorted_values) + 1):
        if end == len(sorted_values) or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                ranks[order[start:end]] = np.mean(np.arange(start, end, dtype=float))
            start = end
    return ranks


# --- sections ---------------------------------------------------------------

def section_corpus(graphs, languages) -> None:
    print("\n=== corpus ===\n")
    print(f"{'document':<46} {'lang':>4} {'nodes':>6} {'edges':>6} {'refs':>5} {'pages':>6}")
    for graph in graphs:
        print(f"{graph.filename[:46]:<46} {languages[graph.document_id]:>4} "
              f"{len(graph.snippets):>6} {len(graph.edges):>6} "
              f"{len(graph.reference_edges):>5} {len(graph.pages):>6}")

    by_language = Counter(languages.values())
    print(f"\n{len(graphs)} documents: " +
          ", ".join(f"{n} {lang}" for lang, n in sorted(by_language.items())))

    mix = Counter()
    mix_by_language = defaultdict(Counter)
    for graph in graphs:
        ctx = edge_context_from_graph(graph)
        for edge in ctx.edges:
            relation = supply_relation(ctx, edge)
            mix[relation] += 1
            mix_by_language[languages[graph.document_id]][relation] += 1

    total = sum(mix.values())
    langs = sorted(mix_by_language)
    print(f"\nrelation mix ({total} edges)")
    header = f"{'relation':<22} {'n':>7} {'share':>7}"
    for lang in langs:
        header += f" {lang + ' share':>10}"
    print(header)
    for relation in SUPPLY_RELATIONS:
        n = mix.get(relation, 0)
        row = f"{relation:<22} {n:>7} {100 * n / total:>6.1f}%"
        for lang in langs:
            sub = mix_by_language[lang]
            sub_total = sum(sub.values()) or 1
            row += f" {100 * sub.get(relation, 0) / sub_total:>9.1f}%"
        print(row)


def section_priors(graphs, languages, backend: str) -> None:
    print("\n=== priors: is the prior doing the work? ===\n")
    supply = SupplyConfig(similarity=SimilarityConfig(backend=backend))

    rows = defaultdict(lambda: {"coverage": [], "complementarity": [], "measured": [],
                                "supply": [], "floor_bound": 0, "n": 0})
    for graph in graphs:
        ctx = edge_context_from_graph(graph)
        for edge, part in components(ctx, supply).items():
            row = rows[supply_relation(ctx, edge)]
            row["n"] += 1
            row["coverage"].append(part["antecedent_coverage"])
            row["complementarity"].append(part["complementarity"])
            row["measured"].append(part["measured"])
            row["supply"].append(part["supply"])
            row["floor_bound"] += int(part["floor_bound"])

    print(f"{'relation':<22} {'n':>6} {'cover':>6} {'compl':>6} {'measured':>9} "
          f"{'prior':>6} {'supply':>7} {'floor-bound':>12} {'gap':>7}")
    for relation in SUPPLY_RELATIONS:
        row = rows.get(relation)
        if not row or not row["n"]:
            continue
        prior = MEASURED_SUPPLY_PRIORS.get(relation, supply.structural_prior.default)
        measured = st.mean(row["measured"])
        print(f"{relation:<22} {row['n']:>6} {st.mean(row['coverage']):>6.2f} "
              f"{st.mean(row['complementarity']):>6.2f} {measured:>9.2f} "
              f"{prior:>6.2f} {st.mean(row['supply']):>7.2f} "
              f"{100 * row['floor_bound'] / row['n']:>11.1f}% {measured - prior:>+7.2f}")

    print("\n  cover/compl   the two measured components, before the floor")
    print("  measured      max(cover, compl) -- what the evidence alone says")
    print("  prior         the shipped MEASURED_SUPPLY_PRIORS value for the relation")
    print("  floor-bound   share of edges where the prior, not the evidence, set the score")
    print("  gap           measured - prior. Large negative: the prior is carrying the")
    print("                relation. Large positive: the prior is holding it down.")
    print("\n  A relation that is mostly floor-bound has no measurement behind it, so its")
    print("  'supply' column is the prior restated. New priors need new hand labels --")
    print("  flooring a proxy at a prior derived from that same proxy would be circular.")


def section_evidence(graphs, backend: str, estimator: str, half_length: float) -> None:
    print(f"\n=== evidence: which way does each relation move? (need={estimator}) ===\n")
    config = EdgeWeightConfig(
        metric="need_supply",
        need=NeedConfig(estimator=estimator),
        supply=SupplyConfig(similarity=SimilarityConfig(backend=backend)),
        need_supply=NeedSupplyConfig(independence_half_length=half_length),
    )
    # a weight within this of 0.5 is reported as neutral: evidence too weak to act on
    band = 0.05

    rows = defaultdict(lambda: defaultdict(list))
    for graph in graphs:
        ctx = edge_context_from_graph(graph)
        for edge, part in evidence_components(ctx, config).items():
            row = rows[supply_relation(ctx, edge)]
            for key, value in part.items():
                row[key].append(value)

    print(f"{'relation':<22} {'n':>6} {'need':>6} {'supply':>7} {'merge':>6} {'indep':>6} "
          f"{'weight':>7} {'toward 1':>9} {'neutral':>8} {'toward 0':>9}")
    for relation in (*SUPPLY_RELATIONS, "all"):
        if relation == "all":
            row = defaultdict(list)
            for other in rows.values():
                for key, values in other.items():
                    row[key].extend(values)
        else:
            row = rows.get(relation)
        if not row or not row["weight"]:
            continue
        weights = row["weight"]
        n = len(weights)
        up = sum(1 for w in weights if w > 0.5 + band)
        down = sum(1 for w in weights if w < 0.5 - band)
        print(f"{relation:<22} {n:>6} {st.mean(row['need']):>6.2f} {st.mean(row['supply']):>7.2f} "
              f"{st.mean(row['merge_evidence']):>6.2f} "
              f"{st.mean(row['independence_evidence']):>6.2f} {st.mean(weights):>7.2f} "
              f"{100 * up / n:>8.1f}% {100 * (n - up - down) / n:>7.1f}% {100 * down / n:>8.1f}%")

    print(f"\n  merge / indep   mean merge_evidence (need x measured supply) and")
    print(f"                  independence_evidence ((1 - need) x confidence)")
    print(f"  toward 1 / 0    share of edges evidence moved more than {band} from 0.5;")
    print(f"                  neutral is the edge existing, with nothing measured either way")
    print("\n  A relation that is almost all 'toward 0' is one whose children read on their")
    print("  own; almost all 'neutral' means the measures cannot see into it, which is a")
    print("  gap in the estimators rather than a finding about the documents.")


def section_agreement(graphs, configs) -> None:
    print("\n=== agreement: do the metrics order edges differently? ===\n")
    per_metric = defaultdict(list)
    for graph in graphs:
        ctx = edge_context_from_graph(graph)
        ordered = list(ctx.edges)
        for name, config in configs.items():
            weights = compute_edge_weights(ctx, config)
            per_metric[name].extend(weights.get(edge, 0.0) for edge in ordered)

    names = list(configs)
    print(f"{'':<22} " + " ".join(f"{n[:11]:>11}" for n in names))
    for a in names:
        row = f"{a:<22} "
        for b in names:
            value = 1.0 if a == b else spearman(per_metric[a], per_metric[b])
            row += f"{value:>11.2f}" if value == value else f"{'n/a':>11}"
        print(row)

    print(f"\n{'metric':<22} {'mean':>7} {'sd':>7} {'min':>7} {'max':>7} {'distinct':>9} {'zeros':>7}")
    for name in names:
        values = per_metric[name]
        distinct = len({round(v, 4) for v in values})
        print(f"{name:<22} {st.mean(values):>7.3f} {st.pstdev(values):>7.3f} "
              f"{min(values):>7.3f} {max(values):>7.3f} {distinct:>9} "
              f"{sum(1 for v in values if v == 0.0):>7}")

    print("\n  A pair near 1.00 orders edges the same way, so a retrieval condition that")
    print("  swaps one for the other is not testing anything. `distinct` near 1 means the")
    print("  metric cannot separate edges at all, whatever its correlation says.")


def section_merge(graphs, configs, backend: str) -> None:
    print("\n=== merge: what each metric costs once it is used ===\n")
    bare = mean_unit_tokens(graphs, MergePolicy(strategy="none"))
    everything = mean_unit_tokens(graphs, MergePolicy(strategy="all_ancestors"))
    print(f"baselines: none={bare:.1f} tok/unit   all_ancestors={everything:.1f} tok/unit "
          f"({everything / bare:.2f}x)")
    print(f"\ncalibrating every metric to the midpoint, {(bare + everything) / 2:.1f} tok/unit\n")
    target = (bare + everything) / 2

    print(f"{'metric':<22} {'min_weight':>11} {'achieved':>9} {'err':>7} {'merged%':>8} {'vs none':>8}")
    for name, config in configs.items():
        rescored = [recompute_weights(graph, config) for graph in graphs]
        calibration = calibrate_min_weight(rescored, MergePolicy(strategy="weighted"), target)
        units = [unit for graph in rescored
                 for unit in merge_units(graph, calibration.policy)]
        merged = sum(1 for unit in units if unit.is_merged)
        total = sum(unit.token_count for unit in units)
        threshold = calibration.policy.min_weight
        shown = f"{threshold:.3f}" if threshold >= 1e-3 or threshold == 0 else f"{threshold:.1e}"
        print(f"{name:<22} {shown:>11} "
              f"{calibration.achieved_mean_tokens:>9.1f} {calibration.relative_error:>+6.1%} "
              f"{100 * merged / len(units):>7.1f}% {total / (bare * len(units)):>7.2f}x")

    print("\n  A metric that cannot reach the target (large err) cannot be calibrated to")
    print("  it, which is itself a finding about that metric. Two causes, both real:")
    print("  too few distinct values (structural has seven), or a large point mass at")
    print("  zero, which opens a hole in the reachable range -- bm25 similarity scores")
    print("  71% of edges exactly 0.0, so its corpus jumps from 109.0 tok/unit at a")
    print("  threshold of 0.0 to 64.9 at anything above it, with nothing in between.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", default=GRAPHS, help=f"glob of saved graphs (default: {GRAPHS})")
    parser.add_argument("--section", nargs="*",
                        choices=["corpus", "priors", "evidence", "agreement", "merge"],
                        default=["corpus", "priors", "evidence", "agreement", "merge"])
    parser.add_argument("--language", default=None, choices=["de", "en"],
                        help="restrict to documents detected as this language")
    parser.add_argument("--backend", default="bm25", choices=["bm25", "embedding", "blend"],
                        help="similarity backend (default: bm25, needs no model download)")
    parser.add_argument("--metrics", nargs="*", default=None, help="subset of metrics")
    parser.add_argument("--need", default="blend",
                        choices=["reference_density", "syntactic", "blend"],
                        help="need estimator for the evidence section (default: blend)")
    parser.add_argument("--half-length", type=float, default=20.0,
                        help="NeedSupplyConfig.independence_half_length for the evidence section")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.graphs))
    if not paths:
        parser.error(f"no graphs matched {args.graphs!r}; run the extractor first")
    graphs = [load_graph(path) for path in paths]

    stale = [g.filename for g in graphs if g.edges and not g.edge_relations]
    if stale:
        parser.error(
            f"{len(stale)} graph(s) predate the edge relation table (schema 3) and cannot be "
            f"rescored: {stale[:3]}{'...' if len(stale) > 3 else ''}. Re-extract the corpus."
        )

    languages = {graph.document_id: document_language(graph) for graph in graphs}
    if args.language:
        graphs = [g for g in graphs if languages[g.document_id] == args.language]
        if not graphs:
            parser.error(f"no documents detected as {args.language!r}")

    configs = build_configs(args.backend)
    if args.metrics:
        unknown = set(args.metrics) - set(configs)
        if unknown:
            parser.error(f"unknown metrics {sorted(unknown)}; have {sorted(configs)}")
        configs = {name: configs[name] for name in args.metrics}

    if "corpus" in args.section:
        section_corpus(graphs, languages)
    if "priors" in args.section:
        section_priors(graphs, languages, args.backend)
    if "evidence" in args.section:
        section_evidence(graphs, args.backend, args.need, args.half_length)
    if "agreement" in args.section:
        section_agreement(graphs, configs)
    if "merge" in args.section:
        section_merge(graphs, configs, args.backend)


if __name__ == "__main__":
    main()
