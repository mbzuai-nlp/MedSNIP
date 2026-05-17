"""Inter-annotator agreement on batch_0.

batch_0 is the IAA test batch: the same four entries (9, 17, 37, 75) were
each annotated by six annotators (`annotator_1` … `annotator_6`).

Annotators chunked each entry into snippets independently, so two
annotators may not even agree on *how* to group the atomic claims. To
get a common unit of comparison we project each annotator's labels
*down to the atomic-claim level*: for each (entry_id, atom_index), the
annotator's label is whatever they assigned to the snippet that
contains that atom.

We compute, for each labeled field (`label_in_general`,
`label_with_patient_context`):

  - overall agreement % (fraction of items where all 6 agree)
  - Fleiss' κ over the 6 annotators
  - pairwise Cohen's κ matrix (6×6) using sklearn
  - per-annotator label distribution (catches constant-rater outliers
    that collapse κ to zero)

Plus, on the *grouping* side (how annotators chunked atoms into
snippets, independent of labels):

  - Adjusted Rand Index (ARI) per entry + overall mean across the
    15 annotator pairs
  - per-annotator chunking summary (snippet count, mean atoms /
    snippet)

And a "consensus agenda": the top N atoms ranked by total label
disagreement across the three fields.

Output:
  data/3-annotated/iaa_stats.json
"""
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
B0_DIR = ROOT / "data" / "3-annotated" / "batch_0"
OUT_PATH = ROOT / "data" / "3-annotated" / "iaa_stats.json"

ANNOTATOR_RE = re.compile(r"batch_0_annotations_annotator_(\d+)\.json$")

FIELDS = ("label_in_general", "label_with_patient_context")


def fleiss_kappa(matrix: list[list[int]]) -> float:
    """Fleiss' κ for N items × K categories (matrix[i][k] = #raters who put item i in category k).

    Assumes all items have the same total #raters; ignores items with zero ratings.
    """
    matrix = [row for row in matrix if sum(row) > 0]
    if not matrix:
        return 0.0
    n = sum(matrix[0])             # raters per item
    if n < 2:
        return 0.0
    N = len(matrix)                # items
    K = len(matrix[0])             # categories

    # P_i = (Σ_j n_ij^2 - n) / (n * (n-1))
    P_per_item = [
        (sum(c * c for c in row) - n) / (n * (n - 1))
        for row in matrix
    ]
    P_bar = sum(P_per_item) / N

    # P_j: proportion of assignments to category j across all items
    P_cat = [sum(row[j] for row in matrix) / (N * n) for j in range(K)]
    P_e = sum(p * p for p in P_cat)

    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def main():
    # Load each annotator's file, keyed by annotator id (1..6)
    annotators: dict[int, list[dict]] = {}
    for f in sorted(B0_DIR.glob("batch_0_annotations_annotator_*.json")):
        m = ANNOTATOR_RE.search(f.name)
        if not m:
            continue
        annotators[int(m.group(1))] = json.loads(f.read_text())
    ids = sorted(annotators.keys())
    if not ids:
        raise SystemExit("no annotator files found")

    # Per-annotator chunking stats: snippet count + mean atoms per snippet
    chunking = {}
    for aid, payload in annotators.items():
        n_snips = sum(len(e["snippets"]) for e in payload)
        n_atoms = sum(len(s["source_claims"]) for e in payload for s in e["snippets"])
        chunking[aid] = {
            "snippets": n_snips,
            "atom_refs": n_atoms,
            "mean_atoms_per_snippet": round(n_atoms / n_snips, 2) if n_snips else 0.0,
        }

    # Atom -> cluster-id assignment per annotator, per entry, for ARI
    # (For atoms that an annotator placed in multiple snippets, take the
    # first snippet's index — overlap is rare and ARI assumes a partition.)
    entry_ids = sorted({e["entry_id"] for p in annotators.values() for e in p})
    atom_cluster: dict[int, dict[int, dict[int, int]]] = {
        eid: {aid: {} for aid in ids} for eid in entry_ids
    }
    for aid, payload in annotators.items():
        for entry in payload:
            eid = entry["entry_id"]
            for s_idx, sn in enumerate(entry["snippets"]):
                for atom_idx in sn["source_claims"]:
                    atom_cluster[eid][aid].setdefault(atom_idx, s_idx)

    ari_per_entry = {}
    ari_pair_means = []
    for eid in entry_ids:
        # Atoms covered by *all* annotators for this entry
        atoms_all = sorted(set.intersection(
            *(set(atom_cluster[eid][a].keys()) for a in ids)
        ))
        if len(atoms_all) < 2:
            ari_per_entry[eid] = {"atoms": len(atoms_all), "mean_ari": None}
            continue
        pair_aris = []
        for a, b in itertools.combinations(ids, 2):
            la = [atom_cluster[eid][a][x] for x in atoms_all]
            lb = [atom_cluster[eid][b][x] for x in atoms_all]
            pair_aris.append(adjusted_rand_score(la, lb))
        mean_ari = sum(pair_aris) / len(pair_aris)
        ari_per_entry[eid] = {
            "atoms":    len(atoms_all),
            "mean_ari": round(mean_ari, 4),
            "min_ari":  round(min(pair_aris), 4),
            "max_ari":  round(max(pair_aris), 4),
        }
        ari_pair_means.append(mean_ari)
    ari_overall = round(sum(ari_pair_means) / len(ari_pair_means), 4) if ari_pair_means else None

    # Build labels per (entry_id, atom_idx, annotator) by projecting snippet
    # labels onto atoms via source_claims.
    # labels[field][(eid, aidx)][annotator_id] -> "true"|"false"|"yes"|"no"
    labels: dict[str, dict[tuple[int, int], dict[int, str]]] = {
        f: defaultdict(dict) for f in FIELDS
    }
    for aid, payload in annotators.items():
        for entry in payload:
            eid = entry["entry_id"]
            for sn in entry["snippets"]:
                for atom_idx in sn["source_claims"]:
                    key = (eid, atom_idx)
                    for f in FIELDS:
                        labels[f][key][aid] = (sn.get(f) or "").strip().lower()

    # Compute metrics per field
    metrics: dict[str, dict] = {}
    for f in FIELDS:
        # Items where all 6 annotators provided a label
        complete_items = [(k, vals) for k, vals in labels[f].items()
                          if all(a in vals for a in ids)]
        n_items = len(complete_items)

        # Value space (binary in practice)
        categories = sorted({v for _, vals in complete_items for v in vals.values()})

        # Fleiss' matrix: n_items × len(categories)
        cat_idx = {c: i for i, c in enumerate(categories)}
        matrix = [
            [sum(1 for aid in ids if vals[aid] == c) for c in categories]
            for _, vals in complete_items
        ]
        fk = fleiss_kappa(matrix)

        # Overall agreement: fraction of items where all 6 agree
        unanimous = sum(1 for _, vals in complete_items
                        if len({vals[a] for a in ids}) == 1)
        agreement = unanimous / n_items if n_items else 0.0

        # Pairwise Cohen's κ matrix
        pairwise = {}
        for a, b in itertools.combinations(ids, 2):
            ya = [vals[a] for _, vals in complete_items]
            yb = [vals[b] for _, vals in complete_items]
            try:
                k = cohen_kappa_score(ya, yb)
            except Exception:
                k = float("nan")
            pairwise[f"{a}_{b}"] = round(float(k), 4) if k == k else None

        # Mean pairwise κ (skip None)
        vals_only = [v for v in pairwise.values() if v is not None]
        mean_pairwise = round(sum(vals_only) / len(vals_only), 4) if vals_only else None

        metrics[f] = {
            "n_items":           n_items,
            "categories":        categories,
            "agreement":         round(agreement, 4),
            "fleiss_kappa":      round(fk, 4),
            "mean_pairwise_cohen_kappa": mean_pairwise,
            "pairwise_cohen_kappa":      pairwise,
        }

    # Per-annotator label distribution (catches "constant rater" outliers)
    label_dist: dict[int, dict[str, dict[str, int]]] = {aid: {f: defaultdict(int) for f in FIELDS} for aid in ids}
    for aid, payload in annotators.items():
        for entry in payload:
            for sn in entry["snippets"]:
                for f in FIELDS:
                    label_dist[aid][f][(sn.get(f) or "").strip().lower()] += 1
    label_dist_out = {
        aid: {f: dict(counts) for f, counts in fields.items()}
        for aid, fields in label_dist.items()
    }

    # Top-disagreement atoms: rank by total disagreements summed over
    # the three fields. Per item, disagreement count = #raters - #majority-vote.
    def disagreements(vals: dict[int, str]) -> int:
        c: defaultdict[str, int] = defaultdict(int)
        for v in vals.values():
            c[v] += 1
        return sum(vals.values().__len__() and 0 for _ in [0]) or (len(vals) - max(c.values()))

    item_scores: dict[tuple[int, int], dict] = {}
    all_atoms = sorted({k for f in FIELDS for k in labels[f].keys()})
    for key in all_atoms:
        per_field = {}
        total = 0
        for f in FIELDS:
            vals = labels[f].get(key, {})
            if all(a in vals for a in ids):
                d = disagreements(vals)
            else:
                d = 0
            per_field[f] = d
            total += d
        item_scores[key] = {"per_field": per_field, "total": total}
    top_disagreements = [
        {"entry_id": eid, "atom_index": aidx,
         "total_disagreements": info["total"],
         "per_field": info["per_field"]}
        for (eid, aidx), info in sorted(
            item_scores.items(),
            key=lambda kv: (-kv[1]["total"], kv[0][0], kv[0][1])
        )
        if info["total"] > 0
    ][:10]

    out = {
        "n_annotators":  len(ids),
        "annotator_ids": ids,
        "entry_ids":     entry_ids,
        "chunking":      chunking,
        "label_distribution": label_dist_out,
        "grouping_ari": {
            "overall_mean": ari_overall,
            "per_entry":    ari_per_entry,
        },
        "metrics":       metrics,
        "top_disagreement_atoms": top_disagreements,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))

    # Pretty print summary
    print(f"annotators: {ids}")
    print(f"entries:    {entry_ids}")
    print()
    print(f"{'annotator':>11s}  snippets  atom_refs  mean_atoms/sn")
    for aid in ids:
        c = chunking[aid]
        print(f"  annot_{aid}     {c['snippets']:>7d}  {c['atom_refs']:>9d}  {c['mean_atoms_per_snippet']:>13.2f}")

    print(f"\nGrouping agreement (Adjusted Rand Index)")
    print(f"  overall mean ARI: {ari_overall}")
    for eid, info in ari_per_entry.items():
        print(f"    entry {eid}: {info['atoms']:>2d} atoms  mean_ari={info['mean_ari']}  "
              f"min={info.get('min_ari')}  max={info.get('max_ari')}")

    print(f"\n{'field':>30s}  {'n_items':>7s}  {'agreement':>9s}  {'fleiss_κ':>9s}  {'mean_cohen_κ':>13s}")
    for f, m in metrics.items():
        print(f"  {f:>28s}  {m['n_items']:>7d}     {m['agreement']:6.1%}     "
              f"{m['fleiss_kappa']:>6.3f}        {m['mean_pairwise_cohen_kappa']:.3f}")

    print(f"\nLabel distribution per annotator:")
    for aid in ids:
        parts = []
        for f in FIELDS:
            d = label_dist_out[aid][f]
            kv = ", ".join(f"{k}={v}" for k, v in sorted(d.items()))
            parts.append(f"{f}=({kv})")
        flag = "  ← constant on a field" if any(
            len(label_dist_out[aid][f]) == 1 for f in FIELDS
        ) else ""
        print(f"  annot_{aid}: " + "; ".join(parts) + flag)

    print(f"\nTop {len(top_disagreements)} disagreement atoms "
          f"(sum across {len(FIELDS)} fields: {'/'.join(FIELDS)}):")
    print(f"  {'entry':>5s} {'atom':>4s} {'total':>5s}  per-field")
    for d in top_disagreements:
        pf = "/".join(str(d["per_field"][f]) for f in FIELDS)
        print(f"  {d['entry_id']:>5d} {d['atom_index']:>4d} {d['total_disagreements']:>5d}  ({pf})")

    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
