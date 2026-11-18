"""Core PRIME-Ring algorithms: data loading, Alg. 1 (target), Alg. 2 (synthesis), Alg. 4 (exact posterior
certificate), heuristic ring builders, and the lattice OR-proof prototype used for cost measurements."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


@dataclass
class Domain:
    name: str
    G: object                      # scipy csr, symmetric, unweighted
    bucket: np.ndarray             # int bucket id per node
    bucket_names: list
    provenance: dict = field(default_factory=dict)
    log: pd.DataFrame = None       # e-health access log (optional)
    epochs: dict = None            # epoch t -> snapshot graph G_t on the same node index
    _near: dict = field(default_factory=dict)
    _ball: dict = field(default_factory=dict)
    ids: object = None             # external node identifiers, same order as the graph index

    def snapshot(self, rng):
        """Returns (key, G_t). One epoch snapshot is drawn per run when epochs exist."""
        if not self.epochs:
            return None, self.G
        key = list(self.epochs)[int(rng.integers(len(self.epochs)))]
        return key, self.epochs[key]

    def near(self, key, G, d0):
        k = (key, d0)
        if k not in self._near:
            self._near[k] = Near(G, d0)
        return self._near[k]

    def ball(self, key, G, elig, need):
        k = (key, need, int(elig.sum()))
        if k not in self._ball:
            self._ball[k] = Ball(G, elig, need)
        return self._ball[k]

    @property
    def N(self):
        return self.G.shape[0]


def _read_edges(path, src, dst, sep=None):
    df = pd.read_csv(path, sep=sep, engine="python", comment="#", header=0 if src else None)
    if src is None:
        df.columns = ["src", "dst"] + list(df.columns[2:])
        src, dst = "src", "dst"
    return df[src].astype(str).values, df[dst].astype(str).values


def build_graph(src_ids, dst_ids, keep_lcc=True):
    ids = pd.Index(pd.unique(np.concatenate([src_ids, dst_ids])))
    s = ids.get_indexer(src_ids)
    d = ids.get_indexer(dst_ids)
    mask = s != d
    s, d = s[mask], d[mask]
    n = len(ids)
    A = coo_matrix((np.ones(2 * len(s), dtype=np.int8),
                    (np.concatenate([s, d]), np.concatenate([d, s]))), shape=(n, n)).tocsr()
    A.data[:] = 1
    if keep_lcc:
        _, lab = connected_components(A, directed=False)
        big = np.argmax(np.bincount(lab))
        keep = np.where(lab == big)[0]
        A = A[keep][:, keep]
        ids = ids[keep]
    return A, ids


def load_domain(cfg):
    """cfg: dict with keys name, edges, src_col, dst_col, and one of
    attrs (node,bucket csv) | log (+ log_id_col, log_attr_cols) | none (degree fallback)."""
    epochs = None
    if cfg.get("epoch_col"):
        E = pd.read_csv(cfg["edges"])
        sc, dc, ec = cfg["src_col"], cfg["dst_col"], cfg["epoch_col"]
        extra = []
        if cfg.get("log"):
            extra = pd.read_csv(cfg["log"], usecols=[cfg["log_id_col"]])[cfg["log_id_col"]].astype(str).unique().tolist()
        ids = pd.Index(sorted(set(E[sc].astype(str)) | set(E[dc].astype(str)) | set(extra)))
        n = len(ids)

        def mk(df):
            s_, d_ = ids.get_indexer(df[sc].astype(str)), ids.get_indexer(df[dc].astype(str))
            m_ = s_ != d_
            s_, d_ = s_[m_], d_[m_]
            A = coo_matrix((np.ones(2 * len(s_), dtype=np.int8), (np.concatenate([s_, d_]), np.concatenate([d_, s_]))),
                           shape=(n, n)).tocsr()
            A.data[:] = 1
            return A
        G = mk(E)
        epochs = {int(t): mk(g) for t, g in E.groupby(ec)}
        prov = {"edges": cfg["edges"], "nodes": n, "edges_union": int(G.nnz // 2),
                "graph": f"per-epoch snapshot G_t from column {ec}, {len(epochs)} epochs, one drawn per run"}
    else:
        src, dst = _read_edges(cfg["edges"], cfg.get("src_col"), cfg.get("dst_col"), cfg.get("sep"))
        G, ids = build_graph(src, dst, keep_lcc=cfg.get("keep_lcc", True))
        prov = {"edges": cfg["edges"], "nodes_after_lcc": int(G.shape[0]), "edges_after_lcc": int(G.nnz // 2)}
    log = None
    if cfg.get("attrs"):
        a = pd.read_csv(cfg["attrs"])
        idc, bc = cfg.get("attrs_id_col", a.columns[0]), cfg.get("attrs_bucket_col", a.columns[1])
        m = dict(zip(a[idc].astype(str), a[bc].astype(str)))
        lab = np.array([m.get(x, "__missing__") for x in ids])
        prov["buckets_from"] = f"attrs file {cfg['attrs']} column {bc}"
    elif cfg.get("log"):
        log = pd.read_csv(cfg["log"])
        idc = cfg["log_id_col"]
        cols = cfg["log_attr_cols"]
        first = log.groupby(log[idc].astype(str))[cols].first()
        key = first.astype(str).agg("|".join, axis=1)
        m = dict(zip(key.index, key.values))
        lab = np.array([m.get(x, "__missing__") for x in ids])
        prov["buckets_from"] = f"access log {cfg['log']} columns {cols}"
    else:
        # Fallback: degree-quantile x hashed second attribute. Flagged in provenance.
        deg = np.diff(G.indptr)
        q = np.quantile(deg, np.linspace(0, 1, cfg.get("deg_bins", 4) + 1)[1:-1])
        dbin = np.searchsorted(q, deg, side="right")
        h = np.array([int(hashlib.sha256(x.encode()).hexdigest(), 16) % cfg.get("hash_bins", 4) for x in ids])
        lab = np.array([f"deg{a}|h{b}" for a, b in zip(dbin, h)])
        prov["buckets_from"] = "FALLBACK degree-quantile x hash buckets (replace with the paper's profiling file)"
    names, bucket = np.unique(lab, return_inverse=True)
    prov["n_buckets"] = int(len(names))
    dom = Domain(cfg["name"], G, bucket.astype(np.int64), list(names), prov, log, epochs)
    dom.ids = ids
    return dom


# --------------------------------------------------------------------------------------
# Algorithm 1: profiling and target resolution
# --------------------------------------------------------------------------------------


def h2_of_slots(slots):
    n = slots.sum()
    p = slots[slots > 0] / n
    return float(-math.log2(np.sum(p ** 2)))


def resolve_target(bucket, active, tau, Hstar, k, support="uniform", n_override=None, m_override=None):
    """Returns per-bucket slot counts n*q*(a) (int array), or None if infeasible.
    support='uniform': q* uniform on the m = ceil(2^H*) largest buckets (Alg. 1).
    support='raw'    : q* proportional to raw bucket frequencies (ablation)."""
    B = bucket.max() + 1
    counts = np.bincount(bucket[active], minlength=B)
    order = np.argsort(-counts, kind="stable")
    m = m_override or int(math.ceil(2 ** Hstar - 1e-9))
    if support == "uniform":
        if m > np.count_nonzero(counts):
            return None
        S0 = order[:m]
        if n_override:
            n = n_override
            if n % m:
                raise ValueError("n must be a multiple of the support size")
        else:
            n = m * math.ceil(math.ceil(k / tau) / m)
        slots = np.zeros(B, dtype=np.int64)
        slots[S0] = n // m
        if counts[S0].sum() < n:
            return None
        return slots
    # raw-frequency target, same n as the uniform rule
    n = n_override or m * math.ceil(math.ceil(k / tau) / m)
    p = counts / counts.sum()
    raw = p * n
    slots = np.floor(raw).astype(np.int64)
    rem = n - slots.sum()
    slots[np.argsort(-(raw - slots))[:rem]] += 1
    return slots


# --------------------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------------------


class Near:
    """Cached 'within distance d0-1' sets, computed with a limited BFS (dijkstra, unweighted)."""

    def __init__(self, G, d0):
        self.G, self.d0, self.cache = G, d0, {}

    def __call__(self, u):
        if self.d0 <= 1:
            return np.array([u], dtype=np.int64)
        if u not in self.cache:
            dist = dijkstra(self.G, directed=False, indices=u, unweighted=True, limit=self.d0 - 1 + 1e-9)
            self.cache[u] = np.where(np.isfinite(dist))[0]
        return self.cache[u]


def log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


# --------------------------------------------------------------------------------------
# Algorithm 2 (synthesis) and Algorithm 4 (exact posterior certificate)
# --------------------------------------------------------------------------------------


@dataclass
class Ring:
    real: np.ndarray        # real members including the signer
    covers: int
    empty_pool: bool


def synthesize(dom, u0, slots, near, active, rng, cover_budget, signer_counted=True):
    B = len(slots)
    need = slots.copy()
    if signer_counted:
        need[dom.bucket[u0]] -= 1
    if need.min() < 0:
        return None, "signer bucket outside support"
    excl = np.zeros(dom.N, dtype=bool)
    excl[near(u0)] = True
    excl[u0] = True
    chosen, covers, empty = [u0], 0, False
    for a in np.nonzero(need)[0]:
        cand = np.nonzero((dom.bucket == a) & active & ~excl)[0]
        take = min(int(need[a]), len(cand))
        if take < need[a]:
            empty = True
        if take:
            chosen.extend(rng.choice(cand, size=take, replace=False).tolist())
        covers += int(need[a]) - take
    if covers > cover_budget:
        return None, "cover budget exceeded"
    return Ring(np.array(chosen, dtype=np.int64), covers, empty), "ok"


def certify(dom, R, slots, near, active, prior, signer_counted=True):
    """Alg. 4. Returns posterior over R (same order) using exact likelihoods L(u)."""
    B = len(slots)
    bucket_active_counts = np.bincount(dom.bucket[active], minlength=B)
    logw = np.full(len(R), -np.inf)
    Rset = set(R.tolist())
    for i, u in enumerate(R):
        if prior[u] <= 0:
            continue
        n_a = slots.copy()
        if signer_counted:
            n_a[dom.bucket[u]] -= 1
        if n_a.min() < 0:
            continue
        ex = near(u)
        ex_mask_active = active[ex]
        excl_counts = np.bincount(dom.bucket[ex[ex_mask_active]], minlength=B)
        # u itself is in ex only if d0>1; make sure it is excluded exactly once
        if not np.isin(u, ex):
            excl_counts[dom.bucket[u]] += 1 if active[u] else 0
        Ca = bucket_active_counts - excl_counts
        D = np.array([v for v in R if v != u], dtype=np.int64)
        if len(D) and np.isin(D, ex).any():
            continue
        k_a = np.bincount(dom.bucket[D], minlength=B) if len(D) else np.zeros(B, dtype=np.int64)
        if not np.array_equal(k_a, np.minimum(n_a, Ca)):
            continue
        logw[i] = math.log(prior[u]) - sum(log_comb(int(Ca[a]), int(k_a[a])) for a in np.nonzero(k_a)[0])
    if not np.isfinite(logw).any():
        return None
    w = np.exp(logw - logw[np.isfinite(logw)].max())
    return w / w.sum()


# --------------------------------------------------------------------------------------
# Heuristic builders (Table 8). Each returns (ring, posterior) under the attacker who knows the sampler.
# --------------------------------------------------------------------------------------


def builder_uniform(dom, u0, n, elig, rng):
    pool = np.nonzero(elig)[0]
    pool = pool[pool != u0]
    R = np.concatenate([[u0], rng.choice(pool, size=n - 1, replace=False)])
    post = np.where(elig[R], 1.0, 0.0)
    return R, post / post.sum()


def builder_same_bucket(dom, u0, n, elig, rng):
    pool = np.nonzero(elig & (dom.bucket == dom.bucket[u0]))[0]
    pool = pool[pool != u0]
    take = min(n - 1, len(pool))
    R = np.concatenate([[u0], rng.choice(pool, size=take, replace=False)])
    post = np.ones(len(R))
    return R, post / post.sum()


class Ball:
    def __init__(self, G, elig, need, rmax=6):
        self.G, self.elig, self.need, self.rmax, self.cache = G, elig, need, rmax, {}

    def __call__(self, u):
        if u not in self.cache:
            dist = dijkstra(self.G, directed=False, indices=u, unweighted=True, limit=self.rmax + 1e-9)
            dist[u] = np.inf
            ok = np.isfinite(dist) & self.elig
            r_used = self.rmax
            for r in range(1, self.rmax + 1):
                if np.count_nonzero(ok & (dist <= r)) >= self.need:
                    r_used = r
                    break
            members = np.nonzero(ok & (dist <= r_used))[0]
            self.cache[u] = members
        return self.cache[u]


def builder_neighbour(dom, u0, n, elig, rng, ball):
    pool = ball(u0)
    take = min(n - 1, len(pool))
    R = np.concatenate([[u0], rng.choice(pool, size=take, replace=False)])
    logw = np.full(len(R), -np.inf)
    for i, u in enumerate(R):
        if not elig[u]:
            continue
        Bu = ball(u)
        D = np.array([v for v in R if v != u])
        if np.isin(D, Bu).all() and len(D) == min(n - 1, len(Bu)):
            logw[i] = -log_comb(len(Bu), len(D))
    w = np.exp(logw - logw[np.isfinite(logw)].max())
    return R, w / w.sum()


def score(dom, R, post, u0, rng, k, tau):
    mx = post.max()
    best = np.flatnonzero(post == mx)
    guess = R[rng.choice(best)]
    bmass = {}
    for u, p in zip(R, post):
        bmass[dom.bucket[u]] = bmass.get(dom.bucket[u], 0.0) + p
    bguess = max(bmass, key=bmass.get)
    return dict(maxpost=float(mx), success=float(guess == u0), bucket=float(bguess == dom.bucket[u0]),
                passed=float(k * mx <= tau + 1e-12), p_u0=float(post[np.where(R == u0)[0][0]]))


# --------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------


def wilson(succ, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = succ / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def prime_runs(dom, P, rng, runs, *, d0=None, support="uniform", gate=True, signer_counted=True,
               churn=0.0, collect=False):
    """Runs Alg. 1/2/4 'runs' times. Returns summary dict (and per-run rows if collect)."""
    d0 = P["d0"] if d0 is None else d0
    rows = []
    for r in range(runs):
        key, G = dom.snapshot(rng)
        near = dom.near(key, G, d0)
        active = np.ones(dom.N, dtype=bool)
        if churn > 0:
            active[rng.random(dom.N) < churn] = False
        slots = resolve_target(dom.bucket, active, P["tau"], P["Hstar"], P["k"], support=support)
        if slots is None:
            rows.append(dict(status="infeasible"))
            continue
        elig = active & (slots[dom.bucket] > 0)
        prior = elig / elig.sum()
        u0 = rng.choice(np.nonzero(elig)[0])
        n = int(slots.sum())
        budget = int(math.floor(P["cover_frac_max"] * n))
        ring, why = synthesize(dom, u0, slots, near, active, rng, budget, signer_counted)
        if ring is None:
            rows.append(dict(status="refused_synth", why=why))
            continue
        post = certify(dom, ring.real, slots, near, active, prior, signer_counted)
        if post is None:
            rows.append(dict(status="refused_cert", why="no candidate consistent"))
            continue
        sc = score(dom, ring.real, post, u0, rng, P["k"], P["tau"])
        released = (sc["passed"] == 1.0) or not gate
        rows.append(dict(status="released" if released else "refused_cert", n=n, real=len(ring.real),
                         covers=ring.covers, emptypool=float(ring.empty_pool), h2=h2_of_slots(slots),
                         tau_hat=sc["maxpost"], success=sc["success"], bucket=sc["bucket"],
                         above=float(P["k"] * sc["maxpost"] > P["tau"] + 1e-12),
                         gap=sc["p_u0"] - sc["maxpost"]))
    df = pd.DataFrame(rows)
    rel = df[df.status == "released"] if len(df) else df
    out = dict(runs=runs, released_pct=100.0 * len(rel) / runs,
               refused_pct=100.0 * (runs - len(rel)) / runs)
    if len(rel):
        lo, hi = wilson(rel.success.sum(), len(rel))
        out.update(mean_R=rel.real.mean(), mean_tau=rel.tau_hat.mean(), success=rel.success.mean(),
                   ci_lo=lo, ci_hi=hi, max_gap=rel.gap.max(), bucket=rel.bucket.mean(),
                   above_pct=100.0 * rel.above.mean(), mean_S=rel.n.mean(),
                   cover_frac=(rel.covers / rel.n).mean(), empty=rel.emptypool.mean(), h2=rel.h2.mean())
    synth_ok = df[df.status.isin(["released", "refused_cert"])] if len(df) else df
    if len(synth_ok) and "tau_hat" in synth_ok:
        synth_ok = synth_ok[synth_ok.tau_hat.notna()]
    out["empty_all"] = float(synth_ok.emptypool.mean()) if len(synth_ok) and "emptypool" in synth_ok else 0.0
    # pre-gate leakage of the sampler (all synthesised and scored rings), used by the ablation table
    if len(synth_ok):
        out["pg_success"] = float(synth_ok.success.mean())
        out["pg_bucket"] = float(synth_ok.bucket.mean())
        out["rel_above_pct"] = 100.0 * float(((synth_ok.status == "released") & (synth_ok.above == 1.0)).mean())
        out["pg_n"] = int(len(synth_ok))
        out["pg_tau"] = float(synth_ok.tau_hat.mean())
        out["pg_h2"] = float(synth_ok.h2.mean())
        out["pg_S"] = float(synth_ok.n.mean())
        out["pg_cover"] = float((synth_ok.covers / synth_ok.n).mean())
        out["pg_R"] = float(synth_ok.real.mean())
        out["pg_gap"] = float(synth_ok.gap.max())
        lo, hi = wilson(synth_ok.success.sum(), len(synth_ok))
        out["pg_ci_lo"], out["pg_ci_hi"] = lo, hi
    if collect:
        out["_rows"] = df
    return out


def builders_experiment(dom, P, rng, runs):
    active = np.ones(dom.N, dtype=bool)
    slots = resolve_target(dom.bucket, active, P["tau"], P["Hstar"], P["k"])
    if slots is None:
        raise RuntimeError(f"{dom.name}: target infeasible for builders experiment")
    n = int(slots.sum())
    elig = slots[dom.bucket] > 0
    prior = elig / elig.sum()
    res = {b: [] for b in ["Uniform", "Neighbour", "Same-bucket", "PRIME"]}
    budget = int(math.floor(P["cover_frac_max"] * n))
    for _ in range(runs):
        key, G = dom.snapshot(rng)
        near = dom.near(key, G, P["d0"])
        ball = dom.ball(key, G, elig, n - 1)
        u0 = rng.choice(np.nonzero(elig)[0])
        for name, fn in [("Uniform", builder_uniform), ("Same-bucket", builder_same_bucket)]:
            R, post = fn(dom, u0, n, elig, rng)
            res[name].append(score(dom, R, post, u0, rng, P["k"], P["tau"]))
        R, post = builder_neighbour(dom, u0, n, elig, rng, ball)
        res["Neighbour"].append(score(dom, R, post, u0, rng, P["k"], P["tau"]))
        ring, _ = synthesize(dom, u0, slots, near, active, rng, budget)
        if ring is not None:
            post = certify(dom, ring.real, slots, near, active, prior)
            if post is not None:
                res["PRIME"].append(score(dom, ring.real, post, u0, rng, P["k"], P["tau"]))
    out = {}
    for b, lst in res.items():
        d = pd.DataFrame(lst)
        out[b] = dict(maxpost=d.maxpost.mean(), success=d.success.mean(), bucket=d.bucket.mean(),
                      passed=100.0 * d.passed.mean(), runs=len(d))
    out["_n"] = n
    return out


# --------------------------------------------------------------------------------------
# No-shape leakage: plug-in classifier on the unpadded transcript length (policy size class)
# --------------------------------------------------------------------------------------


def noshape_recovery(log, cfg, rng, seeds=5):
    pol, att = cfg["log_policy_col"], cfg["log_target_attr_col"]
    res_col = cfg.get("log_resource_col")
    if cfg.get("log_policy_size_col"):
        size = log[cfg["log_policy_size_col"]].astype(str)
        how = f"policy size column {cfg['log_policy_size_col']}"
    else:
        # proxy: number of distinct attribute tuples granted under each policy handle
        tup = log[cfg.get("log_size_proxy_cols", cfg["log_attr_cols"])].astype(str).agg("|".join, axis=1)
        sz = tup.groupby(log[pol]).nunique()
        size = log[pol].map(sz).astype(str)
        how = "PROXY: distinct granted attribute tuples per policy handle"
    accs_p, accs_a, ch_p, ch_a = [], [], [], []
    for s in range(seeds):
        r = np.random.default_rng(1000 + s)
        if res_col:
            keys = log[res_col].unique()
            test_keys = set(r.choice(keys, size=len(keys) // 2, replace=False))
            te = log[res_col].isin(test_keys).values
        else:
            te = r.random(len(log)) < 0.5
        tr = ~te
        for target, accs, ch in [(pol, accs_p, ch_p), (att, accs_a, ch_a)]:
            y = log[target].astype(str)
            m = pd.DataFrame({"c": size[tr], "y": y[tr]}).groupby("c")["y"].agg(lambda s: s.value_counts().idxmax())
            maj = y[tr].value_counts().idxmax()
            pred = size[te].map(m).fillna(maj)
            accs.append(float((pred.values == y[te].values).mean()))
            ch.append(float((y[te].values == maj).mean()))
    return dict(policy=np.mean(accs_p), attr=np.mean(accs_a), chance_policy=np.mean(ch_p),
                chance_attr=np.mean(ch_a), how=how)


# --------------------------------------------------------------------------------------
# Lattice OR-proof prototype (Table 13). Linear-size ring membership proof, Fiat-Shamir with
# aborts, CDS OR-composition, parallel repetition. Covers ring membership only.
# --------------------------------------------------------------------------------------


class LatticeRing:
    def __init__(self, k_rows=256, m=512, q=8380417, kappa=16, reps=8, seed=0):
        self.k, self.m, self.q, self.C, self.reps = k_rows, m, q, 2 ** kappa, reps
        self.kappa = kappa
        self.B = self.C * m * reps          # per-rep abort rate ~ 1/reps, overall ~ e^-1
        rng = np.random.default_rng(seed)
        self.A = rng.integers(0, q, size=(k_rows, m), dtype=np.int64)

    def keygen(self, rng):
        s = rng.integers(-1, 2, size=self.m, dtype=np.int64)
        return (self.A @ s) % self.q, s

    def _hash(self, msg, ring, W):
        h = hashlib.sha256()
        h.update(msg)
        for t in ring:
            h.update(t.tobytes())
        for w in W:
            h.update(w.tobytes())
        d = h.digest()
        out, ctr = [], 0
        while len(out) < self.reps:
            blk = hashlib.sha256(d + ctr.to_bytes(4, "little")).digest()
            for j in range(0, 32, 2):
                if len(out) < self.reps:
                    out.append(int.from_bytes(blk[j:j + 2], "little") % self.C)
            ctr += 1
        return out

    def sign(self, msg, ring, i, s, rng):
        n, q, C, B, m = len(ring), self.q, self.C, self.B, self.m
        while True:
            Z = np.zeros((self.reps, n, m), dtype=np.int64)
            Cs = np.zeros((self.reps, n), dtype=np.int64)
            W = []
            Y = rng.integers(-B, B + 1, size=(self.reps, m), dtype=np.int64)
            for r in range(self.reps):
                for j in range(n):
                    if j == i:
                        W.append((self.A @ Y[r]) % q)
                    else:
                        Cs[r, j] = rng.integers(0, C)
                        Z[r, j] = rng.integers(-(B - C), B - C + 1, size=m, dtype=np.int64)
                        W.append((self.A @ Z[r, j] - Cs[r, j] * ring[j]) % q)
            chal = self._hash(msg, ring, W)
            ok = True
            for r in range(self.reps):
                Cs[r, i] = (chal[r] - (Cs[r].sum() - Cs[r, i])) % C
                Z[r, i] = Y[r] + Cs[r, i] * s
                if np.abs(Z[r, i]).max() > B - C:
                    ok = False
                    break
            if ok:
                return Cs, Z

    def verify(self, msg, ring, sig):
        Cs, Z = sig
        n, q = len(ring), self.q
        if np.abs(Z).max() > self.B - self.C:
            return False
        W = []
        for r in range(self.reps):
            for j in range(n):
                W.append((self.A @ Z[r, j] - Cs[r, j] * ring[j]) % q)
        chal = self._hash(msg, ring, W)
        return all((Cs[r].sum() % self.C) == chal[r] for r in range(self.reps))

    def encode(self, sig):
        Cs, Z = sig
        return zlib.compress(Cs.astype(np.uint16).tobytes() + Z.astype(np.int32).tobytes(), 6)


def scaling_experiment(dom, P, ns, seeds, sigs_per_n, rng_master):
    HDR = 4 + 4 + 32 + 32 + 4 + 16           # planID, t, statusID, h_P, tau, nonce (bytes)
    rows = []
    for n in ns:
        m_sup = min(P["scaling_support"], n)
        lens, signs, verifs, synths, certs = [], [], [], [], []
        for sd in range(seeds):
            rng = np.random.default_rng(int(rng_master.integers(1 << 31)) + sd)
            lr = LatticeRing(seed=sd)
            iss_pk, iss_sk = lr.keygen(rng)
            keys = [lr.keygen(rng) for _ in range(n)]
            ring = [k[0] for k in keys]
            for t in range(sigs_per_n):
                i = int(rng.integers(n))
                msg = rng.bytes(32)
                t0 = time.perf_counter(); sig = lr.sign(msg, ring, i, keys[i][1], rng); t1 = time.perf_counter()
                assert lr.verify(msg, ring, sig)
                t2 = time.perf_counter()
                cert = lr.sign(msg + b"cert", [iss_pk], 0, iss_sk, rng)  # issuer certificate, size only
                L_S = HDR + len(lr.encode(sig)) + len(lr.encode(cert))
                lens.append(L_S); signs.append(t1 - t0); verifs.append(t2 - t1)
            # synthesis and certification at this ring size on the E-health graph
            active = np.ones(dom.N, dtype=bool)
            slots = resolve_target(dom.bucket, active, P["tau"], P["Hstar"], P["k"], n_override=n, m_override=m_sup)
            if slots is None:
                continue
            elig = slots[dom.bucket] > 0
            prior = elig / elig.sum()
            for _ in range(P["scaling_synth_runs"]):
                key, G = dom.snapshot(rng)
                near = dom.near(key, G, P["d0"])
                u0 = rng.choice(np.nonzero(elig)[0])
                t0 = time.perf_counter()
                ring_s, _ = synthesize(dom, u0, slots, near, active, rng, n)
                t1 = time.perf_counter()
                if ring_s is None:
                    continue
                certify(dom, ring_s.real, slots, near, active, prior)
                t2 = time.perf_counter()
                synths.append(t1 - t0); certs.append(t2 - t1)
        L = int(np.percentile(lens, 99))                 # plan length: 99th percentile of L_S
        acc = [x for x in lens if x <= L]
        rows.append(dict(n=n, sigma_kB=L / 1000.0, pad_B=float(L - np.mean(acc)),
                         synth_ms=1000 * np.mean(synths) if synths else float("nan"),
                         cert_ms=1000 * np.mean(certs) if certs else float("nan"),
                         sign_ms=1000 * np.mean(signs), verify_ms=1000 * np.mean(verifs),
                         reject_pct=100.0 * (1 - len(acc) / len(lens))))
    return rows


