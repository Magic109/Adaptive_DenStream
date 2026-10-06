"""
Adaptive DenStream -- demonstration
===================================

Streams a 2D dataset through DenStream twice, under identical parameters, and
compares the two arms:

    STATIC    the original algorithm -- dbscan_eps is fixed for the whole run
    ADAPTIVE  the same algorithm plus the Cluster Auditor, which periodically
              inspects the micro-cluster graph and re-tunes dbscan_eps online

Everything a user needs to change is found in the CONFIGURATION block below.

The demo prints for each arm, the number of macro-clusters recovered, the final
dbscan_eps and a per-cluster purity report against the ground-truth labels --
then draws both clusterings.

Run:  python Demo.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist

from adaptive_DenStream import DenStream

# ===========================================================================
# CONFIGURATION -- 
# ===========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "datasets", "moon.csv")

# ---- DenStream core parameters (shared by both arms for fair comparison) ----
LAMBDA    = 0.0001   # forgetting factor
EPS       = 0.04     # micro-cluster max radius
BETA      = 0.55     # outlier tolerance
MU        = 4        # weight a micro-cluster must carry to be a core
INIT_SIZE = 1        # warmup size


EPS_MULTIPLIER  = 2 
INITIAL_DBSCAN_EPS = EPS_MULTIPLIER * EPS

# ---- Cluster Auditor (ADAPTIVE arm only) ----
AUDIT_PERIOD = 250   
AUDIT_WARMUP = 250   # auditor warmup size
AUDIT_G      = 0.3   # gamma parameter

# ---- reporting ----
PROGRESS_EVERY = 400   # print the current dbscan_eps every N ticks (adaptive arm)
SHOW_PLOTS     = True


# ===========================================================================
# 1. DATA
# ===========================================================================
def load_dataset(path):
    """Read a CSV whose last column is the ground-truth label.

    No scaling is applied: the parameters above are expressed in the units of
    the raw data, so rescaling here would silently invalidate them.
    """
    print("=" * 62)
    print("1. DATASET")
    print("=" * 62)
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print(f"   file        : {os.path.relpath(path, HERE)}")
    print(f"   points      : {len(X)}   features: {X.shape[1]}")
    print(f"   true classes: {len(np.unique(y))}")
    print(f"   bounds      : X [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]   "
          f"Y [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    return X, y


# ===========================================================================
# 2. ONE STREAMING RUN
# ===========================================================================
def run_stream(X, use_auditor):
    """Stream X through DenStream one point at a time.

    use_auditor=False reproduces the original algorithm: dbscan_eps never moves.
    use_auditor=True  additionally calls audit_clusters(), which may raise
                      dbscan_eps when it finds macro-clusters that the current
                      radius is fragmenting.
    """
    clusterer = DenStream(lambd=LAMBDA, eps=EPS, beta=BETA, mu=MU,
                          dbscan_eps=INITIAL_DBSCAN_EPS, init_size=INIT_SIZE)

    for point in X:
        clusterer.partial_fit(point.reshape(1, -1), 1)

        if clusterer.t % AUDIT_PERIOD == 0:
            labels = clusterer.cluster_p_mcs()

            if use_auditor and clusterer.t > AUDIT_WARMUP:
                if clusterer.audit_clusters(labels, g=AUDIT_G) == "changed":
                    labels = clusterer.cluster_p_mcs()

            if use_auditor and clusterer.t % PROGRESS_EVERY == 0:
                print(f"   [t={clusterer.t:5d}]  dbscan_eps = {clusterer.dbscan_eps:.4f}")

    # the final macro-clustering -- this is what gets evaluated and plotted
    return clusterer, np.asarray(clusterer.cluster_p_mcs())


def run_arm(X, use_auditor, arm):
    """One complete run, labelled for the report."""
    print(f"\n   --- {arm} ---")
    clusterer, labels = run_stream(X, use_auditor)
    return dict(arm=arm, clusterer=clusterer, labels=labels)


# ===========================================================================
# 3. EVALUATION
# ===========================================================================
def macro_labels(labels):
    """The distinct macro-cluster ids, excluding the noise bin (-1)."""
    found = set(int(v) for v in np.unique(labels))
    found.discard(-1)
    return found


def evaluate(result, X, y):
    """Per-macro-cluster purity against the ground truth.

    Each micro-cluster votes for the majority true class among the points within
    eps of its centre; a macro-cluster is PURE when all of its micro-clusters
    agree.

    eps is used rather than each micro-cluster's own radius() because eps is the
    bound a micro-cluster is allowed to reach (_try_merge rejects any insertion
    that would push radius() past it), so radius() <= eps always holds. Voting at
    eps therefore samples the full neighbourhood the micro-cluster represents
    rather than the tighter shell it happens to occupy, which puts more points
    behind every vote.
    """
    clusterer, labels = result["clusterer"], result["labels"]

    print("\n" + "=" * 62)
    print(f"{result['arm']}")
    print("=" * 62)
    print(f"   macro-clusters : {len(macro_labels(labels))}")
    print(f"   micro-clusters : {len(clusterer.p_micro_clusters)}")
    print(f"   final dbscan_eps: {clusterer.dbscan_eps:.4f}"
          f"   (started at {INITIAL_DBSCAN_EPS:.4f})")

    pure = mixed = 0
    for m_label in sorted(macro_labels(labels)):
        votes = []
        for i in np.where(labels == m_label)[0]:
            pmc = clusterer.p_micro_clusters[i]
            dists = cdist([pmc.center()], X)[0]
            inside = np.where(dists <= clusterer.eps)[0]
            if len(inside) > 0:
                vals, counts = np.unique(y[inside], return_counts=True)
                votes.append(vals[np.argmax(counts)])
            else:
                votes.append(y[np.argmin(dists)])

        classes, counts = np.unique(votes, return_counts=True)
        if len(classes) == 1:
            pure += 1
            status = "PURE"
        else:
            mixed += 1
            status = "MIXED"

        breakdown = ", ".join(f"class {c}: {n} ({100 * n / len(votes):.1f}%)"
                              for c, n in zip(classes, counts))
        print(f"   [cluster {m_label}] {status:<5} -- {breakdown}")

    print(f"   summary: {pure} pure | {mixed} mixed")

    noise = int((labels == -1).sum())
    if noise:
        print(f"   noise bin: {noise} micro-cluster(s) unassigned")


# ===========================================================================
# 4. VISUALISATION
# ===========================================================================
def plot_clusters(result, X):
    """Raw points in grey, micro-cluster centres coloured by macro-cluster, and
    each micro-cluster's physical radius drawn as a bubble."""
    clusterer, labels = result["clusterer"], result["labels"]
    _, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X[:, 0], X[:, 1], c="lightgray", s=10, alpha=0.3,
               label="raw data", zorder=1)

    unique = sorted(set(int(v) for v in np.unique(labels)))
    cmap = plt.get_cmap("tab10", max(len(unique), 1))

    for i, m_label in enumerate(unique):
        first = True
        for j in np.where(labels == m_label)[0]:
            pmc = clusterer.p_micro_clusters[j]
            center, radius = pmc.center(), pmc.radius()
            if np.isnan(radius) or radius == 0.0:
                radius = clusterer.eps * 0.1

            if m_label == -1:
                ax.scatter(*center[:2], c="black", marker="x", s=40, zorder=3,
                           label="noise (-1)" if first else "")
            else:
                color = cmap(i)
                ax.scatter(*center[:2], c=[color], edgecolors="black", s=25, zorder=4,
                           label=f"macro-cluster {m_label}" if first else "")
                ax.add_patch(Circle(center[:2], radius, color=color, alpha=0.2, zorder=2))
            first = False

    ax.set_title(f"{result['arm']}  -- final dbscan_eps = {clusterer.dbscan_eps:.4f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    handles, names = ax.get_legend_handles_labels()
    ax.legend(dict(zip(names, handles)).values(), dict(zip(names, handles)).keys(),
              bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_aspect("equal", "datalim")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    X, y = load_dataset(DATASET)

    print("\n" + "=" * 62)
    print("2. PARAMETERS")
    print("=" * 62)
    print(f"   lambda={LAMBDA}  eps={EPS}  beta={BETA}  mu={MU}  init_size={INIT_SIZE}")
    print(f"   dbscan_eps starts at {INITIAL_DBSCAN_EPS:.4f} "
          f"(= {EPS_MULTIPLIER} x eps) for BOTH arms")
    print(f"   auditor: every {AUDIT_PERIOD} ticks after tick {AUDIT_WARMUP}, g={AUDIT_G}")

    print("\n" + "=" * 62)
    print("3. STREAMING")
    print("=" * 62)
    static = run_arm(X, use_auditor=False, arm="STATIC DenStream")
    adaptive = run_arm(X, use_auditor=True, arm="ADAPTIVE DenStream")

    evaluate(static, X, y)
    evaluate(adaptive, X, y)

    print("\n" + "=" * 62)
    print("RESULT")
    print("=" * 62)
    print(f"   STATIC   recovered {len(macro_labels(static['labels']))} macro-cluster(s) "
          f"at a fixed dbscan_eps of {INITIAL_DBSCAN_EPS:.4f}")
    print(f"   ADAPTIVE recovered {len(macro_labels(adaptive['labels']))} macro-cluster(s) "
          f"after raising dbscan_eps to {adaptive['clusterer'].dbscan_eps:.4f}")
    print(f"   ground truth: {len(np.unique(y))} classes")

    if SHOW_PLOTS:
        print("\nDrawing plots (close each window to continue)...")
        plot_clusters(static, X)
        plot_clusters(adaptive, X)
