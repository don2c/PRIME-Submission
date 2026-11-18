#!/usr/bin/env python3
"""Synthetic OSN graphs with SNAP-like structure: dense communities (ego circles)
plus degree-preferential links between communities.

OSN-Synth-A: 4,039 nodes, communities of 45-75 nodes, about 42 in-community
             neighbours, 1 cross link per node (sized like SNAP ego-Facebook:
             4,039 nodes, about 88k edges, clustering about 0.6).
OSN-Synth-B: 8,000 nodes, communities of 28-50 nodes, about 20 in-community
             neighbours, 1 cross link per node (a scaled-down, sparser graph in
             the style of SNAP ego-Twitter).

Attributes: 10 buckets with Zipf-skewed sizes (exponent 1.1); after the initial
draw, 30% of nodes copy the bucket of a random neighbour (homophily).
Each user also has a region (20 values). A moderation log is generated per graph:
OSN-Facebook 5,000 events and OSN-Twitter 50,000 events over 12 epochs, each with a
policy handle, plan identifier and epoch status digest. Graphs use the seeds below;
attributes and logs use separate seeded streams, so the graphs do not depend on them.
Everything is written to data/."""
import hashlib
import os
import numpy as np
import pandas as pd


def community_graph(n, size_lo, size_hi, deg_in, inter, rng):
    """Dense overlapping-free communities (ego circles) plus degree-preferential inter-community edges.
    Clustering is close to the within-community edge density."""
    sizes = []
    while sum(sizes) < n:
        sizes.append(int(np.clip(rng.lognormal(np.log((size_lo + size_hi) / 2), 0.4), size_lo, size_hi)))
    sizes[-1] -= sum(sizes) - n
    adj = [set() for _ in range(n)]
    start = 0
    for s_ in sizes:
        if s_ < 2:
            start += s_; continue
        p_in = min(0.95, deg_in / (s_ - 1))
        idx = np.arange(start, start + s_)
        iu, ju = np.triu_indices(s_, 1)
        keep = rng.random(len(iu)) < p_in
        for a, b in zip(idx[iu[keep]], idx[ju[keep]]):
            adj[a].add(int(b)); adj[b].add(int(a))
        start += s_
    deg = np.array([len(a) for a in adj], dtype=float) + 1
    for v in range(n):
        for w in rng.choice(n, size=inter, p=deg / deg.sum()):
            if w != v:
                adj[v].add(int(w)); adj[int(w)].add(v)
    return adj


def clustering(adj, sample, rng):
    nodes = rng.choice(len(adj), size=min(sample, len(adj)), replace=False)
    cs = []
    for v in nodes:
        nb = list(adj[v]); k = len(nb)
        if k < 2:
            cs.append(0.0); continue
        links = sum(1 for i in range(k) for j in range(i + 1, k) if nb[j] in adj[nb[i]])
        cs.append(2 * links / (k * (k - 1)))
    return float(np.mean(cs))


POLICIES = ["M_LABEL", "M_REMOVE_SPAM", "M_RESTRICT", "M_SUSPEND", "M_ESCALATE"]
BASE = np.array([0.45, 0.22, 0.15, 0.11, 0.07])


def make_log(name, adj, lab, region, n_events, seed, epochs=12, n_items=2000):
    """Moderation events. Signers act in proportion to sqrt(degree); the policy mix depends
    mildly on the signer's bucket (Dirichlet around BASE)."""
    rng = np.random.default_rng(seed)
    n = len(adj)
    deg = np.array([len(a) for a in adj], dtype=float)
    act = np.sqrt(deg + 1); act /= act.sum()
    B = lab.max() + 1
    mix = rng.dirichlet(40 * BASE, size=B)
    users = rng.choice(n, size=n_events, p=act)
    t = rng.integers(1, epochs + 1, size=n_events)
    pol = np.array([rng.choice(len(POLICIES), p=mix[lab[u]]) for u in users])
    status = {e: hashlib.sha256(f"{name}|{e}".encode()).hexdigest()[:16] for e in range(1, epochs + 1)}
    df = pd.DataFrame({
        "event_id": [f"{name.upper()}_{i:07d}" for i in range(n_events)],
        "epoch_t": t, "status_id": [status[e] for e in t],
        "plan_id": rng.choice(["PLAN_A", "PLAN_B", "PLAN_C"], size=n_events, p=[0.5, 0.35, 0.15]),
        "user_id": users, "bucket": [f"A{lab[u]}" for u in users],
        "attr_value": [f"A{lab[u]}|R{region[u]:02d}" for u in users],
        "policy_handle": [POLICIES[p] for p in pol],
        "item_id": rng.integers(0, n_items, size=n_events)})
    df.to_csv(f"data/{name}_log.csv", index=False)


def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.makedirs("data", exist_ok=True)
    rows = []
    for name, n, lo, hi, din, inter, seed, n_ev in [("osn_facebook", 4039, 45, 75, 42, 1, 11, 5000),
                                                    ("osn_twitter", 8000, 28, 50, 20, 1, 12, 50000)]:
        rng = np.random.default_rng(seed)
        adj = community_graph(n, lo, hi, din, inter, rng)
        edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
        pd.DataFrame(edges, columns=["src", "dst"]).to_csv(f"data/{name}_edges.csv", index=False)
        B = 10
        probs = 1.0 / np.arange(1, B + 1) ** 1.1
        probs /= probs.sum()
        lab = rng.choice(B, size=n, p=probs)
        for v in rng.choice(n, size=int(0.3 * n), replace=False):
            if adj[v]:
                lab[v] = lab[int(rng.choice(list(adj[v])))]
        region = np.random.default_rng(seed + 500).integers(0, 20, size=n)
        pd.DataFrame({"node": np.arange(n), "bucket": [f"A{x}" for x in lab], "region": region}).to_csv(
            f"data/{name}_attrs.csv", index=False)
        make_log(name, adj, lab, region, n_ev, seed + 1000)
        deg = np.array([len(a) for a in adj])
        rows.append(dict(graph=name, nodes=n, edges=len(edges), avg_degree=round(deg.mean(), 1),
                         clustering=round(clustering(adj, 400, rng), 3),
                         bucket_sizes=sorted(np.bincount(lab, minlength=B).tolist(), reverse=True)))
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv("data/osn_synth_stats.csv", index=False)


if __name__ == "__main__":
    main()
