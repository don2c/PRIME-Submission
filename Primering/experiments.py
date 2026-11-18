"""Transcript-level experiments: AOV hiding, constant shape, freshness and linking, governed opening,
attack search and stage costs. All transcripts are signed and verified with the lattice prototype in core.py."""
import hashlib
import math
import time
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from . import opening as op
from .core import LatticeRing, certify, resolve_target, synthesize

warnings.filterwarnings("ignore")
HDR = 4 + 4 + 32 + 32 + 4 + 16            # planID, t, statusID, h_P, tau, nonce (bytes)


# --------------------------------------------------------------------------------------
# Keys, transcripts, verifier
# --------------------------------------------------------------------------------------


class KeyStore:
    """Independent per-epoch user keys (pk^(t), sk^(t)), derived deterministically from (domain, node, t)."""

    def __init__(self, lr, name):
        self.lr, self.name, self.cache = lr, name, {}

    def get(self, node, t):
        k = (int(node), int(t))
        if k not in self.cache:
            seed = int.from_bytes(hashlib.sha256(f"{self.name}|{k[0]}|{k[1]}".encode()).digest()[:8], "little")
            self.cache[k] = self.lr.keygen(np.random.default_rng(seed))
        return self.cache[k]

    def cover(self, t, idx):
        return self.get(-1 - idx, t)

    def key_id(self, node, t):
        return hashlib.sha256(self.get(node, t)[0].tobytes()).digest()


class Issuer:
    def __init__(self, lr, seed):
        self.lr = lr
        self.pk, self.sk = lr.keygen(np.random.default_rng(seed))

    def cert(self, ring_pks, plan, t, status, tau, k, rng):
        h = hashlib.sha256(b"".join(p.tobytes() for p in ring_pks) + f"|{plan}|{t}|{status}|{tau}|{k}".encode()).digest()
        return self.lr.sign(h, [self.pk], 0, self.sk, rng), h


class Transcript:
    __slots__ = ("nodes", "pks", "t", "status", "plan", "nonce", "cert", "cert_msg", "sig", "ctx", "L_S",
                 "sign_s", "signer")


def ring_pks(ks, nodes, covers, t):
    return [ks.get(v, t)[0] for v in nodes] + [ks.cover(t, j)[0] for j in range(covers)]


def make_transcript(lr, ks, issuer, nodes, covers, signer, t, status, plan, P, msg, rng):
    tr = Transcript()
    tr.nodes = np.sort(np.asarray(nodes, dtype=np.int64))       # canonical order: position reveals nothing
    tr.pks = ring_pks(ks, tr.nodes, covers, t)
    tr.t, tr.status, tr.plan, tr.signer = int(t), status, plan, int(signer)
    tr.nonce = rng.bytes(16)
    tr.cert, tr.cert_msg = issuer.cert(tr.pks, plan, t, status, P["tau"], P["k"], rng)
    tr.ctx = hashlib.sha256(msg + f"|{plan}|{t}|{status}|{P['tau']}".encode() + tr.nonce + tr.cert_msg).digest()
    i = int(np.nonzero(tr.nodes == signer)[0][0])
    t0 = time.perf_counter()
    tr.sig = lr.sign(tr.ctx, tr.pks, i, ks.get(signer, t)[1], rng)
    tr.sign_s = time.perf_counter() - t0
    tr.L_S = HDR + len(lr.encode(tr.sig)) + len(lr.encode(tr.cert))
    return tr


def verify_padded(lr, tr, Theta):
    """SoK verification followed by dummy work up to the plan cost Theta. Returns (ok, elapsed seconds)."""
    t0 = time.perf_counter()
    ok = lr.verify(tr.ctx, tr.pks, tr.sig)
    while time.perf_counter() - t0 < Theta:
        pass
    return ok, time.perf_counter() - t0


def calibrate_plan(lr, ks, issuer, dom, P, rng, n_sig):
    """Fixes the plan (L, Theta) for ring size n: L = 99th percentile of L_S, Theta = theta_margin x max verify time."""
    active = np.ones(dom.N, dtype=bool)
    slots = resolve_target(dom.bucket, active, P["tau"], P["Hstar"], P["k"])
    elig = np.nonzero(slots[dom.bucket] > 0)[0]
    lens, vts = [], []
    while len(lens) < n_sig:
        t = epoch_list(dom)[int(rng.integers(len(epoch_list(dom))))]
        G = snapshot_at(dom, t)
        u0 = int(rng.choice(elig))
        ring, _ = synthesize(dom, u0, slots, dom.near(t, G, P["d0"]), active, rng, int(slots.sum()) // 2)
        if ring is None:
            continue
        tr = make_transcript(lr, ks, issuer, ring.real, ring.covers, u0, t, "cal", "PLAN", P, rng.bytes(32), rng)
        t0 = time.perf_counter(); lr.verify(tr.ctx, tr.pks, tr.sig); vts.append(time.perf_counter() - t0)
        lens.append(tr.L_S)
    return dict(n=int(slots.sum()), L=int(np.percentile(lens, P.get("plan_percentile", 99))),
                Theta=P.get("theta_margin", 1.5) * max(vts), slots=slots)


def epoch_list(dom):
    return sorted(dom.epochs) if dom.epochs else list(range(1, 13))


def snapshot_at(dom, t):
    return dom.epochs[t] if dom.epochs else dom.G


# --------------------------------------------------------------------------------------
# Transcript sample from a domain log (shared by AOV hiding and stage costs)
# --------------------------------------------------------------------------------------


def node_index(dom):
    return {str(x): i for i, x in enumerate(dom.ids)}


def sample_transcripts(dom, log, cols, plan, lr, ks, issuer, P, rng, n_events):
    idx = node_index(dom)
    slots = plan["slots"]
    active = np.ones(dom.N, dtype=bool)
    nodes = log[cols["id"]].astype(str).map(idx)
    ok = nodes.notna() & (slots[dom.bucket[nodes.fillna(0).astype(int)]] > 0)
    stats = dict(events=len(log), eligible=int(ok.sum()))
    sub = log[ok].sample(n=min(n_events, int(ok.sum())), random_state=int(rng.integers(1 << 30)))
    rows, trs = [], []
    for _, e in sub.iterrows():
        u0, t = int(idx[str(e[cols["id"]])]), int(e[cols["epoch"]])
        G = snapshot_at(dom, t)
        ring, _ = synthesize(dom, u0, slots, dom.near(t, G, P["d0"]), active, rng, int(slots.sum()) // 2)
        if ring is None:
            continue
        tr = make_transcript(lr, ks, issuer, ring.real, ring.covers, u0, t, e[cols["status"]], e[cols["plan"]], P,
                             str(e[cols["event"]]).encode(), rng)
        if tr.L_S > plan["L"]:
            rows.append(dict(rejected=1)); continue
        okv, theta = verify_padded(lr, tr, plan["Theta"])
        assert okv
        nb = np.frombuffer(tr.nonce, dtype=np.uint32) / 2.0 ** 32
        cb = np.frombuffer(hashlib.sha256(lr.encode(tr.cert)).digest()[:16], dtype=np.uint32) / 2.0 ** 32
        rows.append(dict(rejected=0, plan=tr.plan, t=tr.t, status=tr.status, L=plan["L"], theta_ms=1000 * theta,
                         nS=len(tr.pks), tau=P["tau"], **{f"nonce{j}": nb[j] for j in range(4)},
                         **{f"cert{j}": cb[j] for j in range(4)}, policy=e[cols["policy"]],
                         attr=e[cols["attr"]], sign_s=tr.sign_s, verify_s=theta, nodes=tr.nodes))
    df = pd.DataFrame(rows)
    stats["sampled"] = len(df)
    stats["length_rejected"] = int(df.rejected.sum())
    return df[df.rejected == 0].reset_index(drop=True), stats


# --------------------------------------------------------------------------------------
# AOV hiding (Table: tab:prime_aov_hiding) and positive control
# --------------------------------------------------------------------------------------


def _attackers():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    return {"logreg": lambda: LogisticRegression(max_iter=400),
            "gbt": lambda: HistGradientBoostingClassifier(max_iter=40, max_depth=3),
            "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=300)}


def _meta_matrix(df):
    X = pd.get_dummies(df[["plan", "t"]].astype(str), dtype=float)
    num = df[["L", "theta_ms", "nS", "tau"] + [f"nonce{j}" for j in range(4)] + [f"cert{j}" for j in range(4)]]
    num = (num - num.mean()) / num.std().replace(0, 1)
    return np.hstack([X.values, num.fillna(0).values])


def _ring_matrix(df, N):
    r, c = [], []
    for i, nodes in enumerate(df.nodes):
        r += [i] * len(nodes); c += list(nodes)
    return csr_matrix((np.ones(len(r)), (r, c)), shape=(len(df), N))


def _epoch_splits(df, seeds):
    ep = np.array(sorted(df.t.unique()))
    for s in range(seeds):
        test = np.random.default_rng(500 + s).choice(ep, size=max(1, len(ep) // 3), replace=False)
        te = df.t.isin(test).values
        yield s, ~te, te


def aov_experiment(df, N, registry, seeds):
    from sklearn.linear_model import LogisticRegression
    targets = {"Policy body": df.policy.astype(str),
               "Attribute value": df.attr.astype(str),
               "Policy mask": df.policy.map(lambda p: "+".join(registry[p]["mask"])).astype(str)}
    Xm = _meta_matrix(df)
    Xr = hstack([_ring_matrix(df, N), csr_matrix(Xm)]).tocsr()
    out = {}
    for name, y in targets.items():
        y = y.values
        acc = {a: [] for a in _attackers()}
        chance, ring_acc, ctrl = [], [], []
        for s, tr, te in _epoch_splits(df, seeds):
            maj = pd.Series(y[tr]).value_counts().idxmax()
            chance.append(float((y[te] == maj).mean()))
            for a, mk in _attackers().items():
                if len(np.unique(y[tr])) < 2:
                    acc[a].append(chance[-1]); continue
                clf = mk().fit(Xm[tr], y[tr])
                acc[a].append(float((clf.predict(Xm[te]) == y[te]).mean()))
            clf = LogisticRegression(max_iter=400).fit(Xr[tr], y[tr])
            ring_acc.append(float((clf.predict(Xr[te]) == y[te]).mean()))
            # positive control: the same view with the label in the clear
            # (scored on test events whose label occurs in training, since no classifier can output an unseen label)
            Xc = np.hstack([Xm, pd.get_dummies(pd.Series(y), dtype=float).values])
            clf = LogisticRegression(max_iter=400).fit(Xc[tr], y[tr])
            seen = te & np.isin(y, np.unique(y[tr]))
            ctrl.append(float((clf.predict(Xc[seen]) == y[seen]).mean()))
        best = max(acc, key=lambda a: np.mean(acc[a]))
        out[name] = dict(acc=float(np.mean(acc[best])), attacker=best, chance=float(np.mean(chance)),
                         adv=float(np.mean(acc[best]) - np.mean(chance)), ring_acc=float(np.mean(ring_acc)),
                         control=float(np.mean(ctrl)), per_attacker={a: float(np.mean(v)) for a, v in acc.items()})
    return out


# --------------------------------------------------------------------------------------
# Constant shape by churn (Figure: fig:prime_ring_shape)
# --------------------------------------------------------------------------------------


def shape_experiment(dom, plan, lr, ks, issuer, P, rng, churns, runs):
    rows = []
    for c in churns:
        res = dict(pool=0, length=0, time=0, ok=0, pads=[], dthetas=[])
        for _ in range(runs):
            t = epoch_list(dom)[int(rng.integers(len(epoch_list(dom))))]
            active = np.ones(dom.N, dtype=bool)
            if c > 0:
                active[rng.random(dom.N) < c] = False
            slots = resolve_target(dom.bucket, active, P["tau"], P["Hstar"], P["k"], n_override=plan["n"])
            elig = np.nonzero(active & (slots[dom.bucket] > 0))[0] if slots is not None else []
            ring = None
            if len(elig):
                u0 = int(rng.choice(elig))
                ring, _ = synthesize(dom, u0, slots, dom.near(t, snapshot_at(dom, t), P["d0"]), active, rng,
                                     int(math.floor(P["cover_frac_max"] * plan["n"])))
            if ring is None:
                res["pool"] += 1; continue
            tr = make_transcript(lr, ks, issuer, ring.real, ring.covers, u0, t, "s", "PLAN", P, rng.bytes(32), rng)
            if tr.L_S > plan["L"]:
                res["length"] += 1; continue
            okv, theta = verify_padded(lr, tr, plan["Theta"])
            dth = abs(theta - plan["Theta"])
            if not okv or dth > P["delta_theta_ms"] / 1000:
                res["time"] += 1; continue
            res["ok"] += 1; res["pads"].append(plan["L"] - tr.L_S); res["dthetas"].append(1000 * dth)
        rows.append(dict(domain=dom.name, churn=c, runs=runs, reject_pct=100 * (runs - res["ok"]) / runs,
                         pool_pct=100 * res["pool"] / runs, length_pct=100 * res["length"] / runs,
                         time_pct=100 * res["time"] / runs,
                         pad_B=float(np.mean(res["pads"])) if res["pads"] else float("nan"),
                         dtheta_ms=float(np.mean(res["dthetas"])) if res["dthetas"] else float("nan")))
    return rows


# --------------------------------------------------------------------------------------
# Freshness, replay, past-epoch forgery and linking (Table: tab:prime_epoch_fs)
# --------------------------------------------------------------------------------------


class Verifier:
    def __init__(self, lr, ks, plan, snapshots):
        self.lr, self.ks, self.plan, self.snap = lr, ks, plan, snapshots
        self.t_cur = self.status_cur = None
        self.issued, self.used = set(), set()

    def advance(self, t):
        self.t_cur, self.status_cur = t, self.snap[t]["status"]

    def verify(self, tr, length):
        if (tr.t, tr.status) != (self.t_cur, self.status_cur):
            return 0
        if tr.nonce not in self.issued or tr.nonce in self.used:
            return 0
        if length != self.plan["L"] or len(tr.pks) != self.plan["n"]:
            return 0
        if any(hashlib.sha256(p.tobytes()).digest() not in self.snap[self.t_cur]["keys"] for p in tr.pks):
            return 0
        if not self.lr.verify(tr.ctx, tr.pks, tr.sig):
            return 0
        self.used.add(tr.nonce)
        return 1


def freshness_experiment(dom, plan, lr, ks, issuer, P, rng, per_epoch, pool_size, link_pairs, seeds):
    T = epoch_list(dom)
    slots = plan["slots"]
    active = np.ones(dom.N, dtype=bool)
    elig = np.nonzero(slots[dom.bucket] > 0)[0]
    signers = rng.choice(elig, size=min(pool_size, len(elig)), replace=False)
    trs = {t: [] for t in T}
    for t in T:
        G = snapshot_at(dom, t)
        while len(trs[t]) < per_epoch:
            u0 = int(rng.choice(signers))
            ring, _ = synthesize(dom, u0, slots, dom.near(t, G, P["d0"]), active, rng, plan["n"] // 2)
            if ring is None:
                continue
            tr = make_transcript(lr, ks, issuer, ring.real, ring.covers, u0, t,
                                 hashlib.sha256(f"{dom.name}|{t}".encode()).hexdigest()[:16], "PLAN", P,
                                 rng.bytes(32), rng)
            if tr.L_S <= plan["L"]:
                trs[t].append(tr)
    snaps = {t: dict(status=hashlib.sha256(f"{dom.name}|{t}".encode()).hexdigest()[:16],
                     keys={hashlib.sha256(p.tobytes()).digest() for tr in trs[t] for p in tr.pks}) for t in T}
    V = Verifier(lr, ks, plan, snaps)
    fresh, replay_same, replay_old, stale_status, forge = [], [], [], [], []
    for j, t in enumerate(T):
        V.advance(t)
        for tr in trs[t]:
            V.issued.add(tr.nonce)
            fresh.append(V.verify(tr, plan["L"]))
            replay_same.append(V.verify(tr, plan["L"]))          # nonce already used
            # stale status digest inside the current epoch: the context no longer matches either
            old = tr.status
            tr.status = snaps[T[j - 1]]["status"] if j else "0" * 16
            stale_status.append(V.verify(tr, plan["L"]))
            tr.status = old
        if j:
            for tr in trs[T[j - 1]]:                               # previous-epoch transcripts at epoch t
                replay_old.append(V.verify(tr, plan["L"]))
    # compromise at the latest epoch: the adversary holds every sk^(T_last)
    tl = T[-1]
    V.advance(tl)
    for t in T[:-1]:
        for tr in trs[t][: max(2, per_epoch // 4)]:
            i = int(np.nonzero(tr.nodes == tr.signer)[0][0])
            forged = Transcript()
            for f in Transcript.__slots__:
                if hasattr(tr, f):
                    setattr(forged, f, getattr(tr, f))
            forged.nonce = rng.bytes(16)
            forged.ctx = hashlib.sha256(tr.ctx + forged.nonce).digest()
            forged.sig = lr.sign(forged.ctx, tr.pks, i, ks.get(tr.signer, tl)[1], rng)   # epoch-tl secret
            crypto_ok = lr.verify(forged.ctx, forged.pks, forged.sig)
            V.issued.add(forged.nonce)
            forge.append(int(crypto_ok) + V.verify(forged, plan["L"]))
    # linking across epochs
    allt = [tr for t in T for tr in trs[t]]
    pairs, lab = [], []
    by_signer = {}
    for k, tr in enumerate(allt):
        by_signer.setdefault(tr.signer, []).append(k)
    while len(pairs) < link_pairs:
        a, b = rng.integers(len(allt), size=2)
        same = len(pairs) % 2 == 0
        if same:
            cand = [x for x in by_signer[allt[a].signer] if allt[x].t != allt[a].t]
            if not cand:
                continue
            b = int(rng.choice(cand))
        elif allt[a].signer == allt[b].signer or allt[a].t == allt[b].t:
            continue
        pairs.append((int(a), int(b))); lab.append(int(same))
    lab = np.array(lab)

    def crypto_feats(x, y):
        kx = {hashlib.sha256(p.tobytes()).digest() for p in x.pks}
        ky = {hashlib.sha256(p.tobytes()).digest() for p in y.pks}
        hx = np.frombuffer(hashlib.sha256(lr.encode(x.sig)).digest()[:8], dtype=np.uint8)
        hy = np.frombuffer(hashlib.sha256(lr.encode(y.sig)).digest()[:8], dtype=np.uint8)
        return [len(kx & ky), float(np.unpackbits(hx ^ hy).sum()), abs(x.L_S - y.L_S)]

    Xc = np.array([crypto_feats(allt[a], allt[b]) for a, b in pairs], dtype=float)
    Xr = np.array([[len(set(allt[a].nodes) & set(allt[b].nodes))] for a, b in pairs], dtype=float)
    from sklearn.linear_model import LogisticRegression
    acc_c, acc_r = [], []
    for s in range(seeds):
        te = np.random.default_rng(900 + s).random(len(pairs)) < 0.5
        for X, acc in [(Xc, acc_c), (Xr, acc_r)]:
            clf = LogisticRegression().fit(X[~te], lab[~te])
            acc.append(float((clf.predict(X[te]) == lab[te]).mean()))
    return dict(epochs=len(T), transcripts=len(allt), fresh=float(np.mean(fresh)),
                replay_same=float(np.mean(replay_same)), replay_old=float(np.mean(replay_old)),
                stale_status=float(np.mean(stale_status)), forge=float(np.mean(np.array(forge) > 0)),
                link_crypto=float(np.mean(acc_c)), link_ring=float(np.mean(acc_r)), link_chance=0.5,
                pairs=len(pairs))


# --------------------------------------------------------------------------------------
# Governed opening and audit log
# --------------------------------------------------------------------------------------


def opening_experiment(dom, log, cols, ks, P, rng):
    idx = node_index(dom)
    pke = op.AuditorPKE(int(rng.integers(1 << 30)))
    gov = rng.bytes(32)
    opener = op.GovernedOpener(pke, gov)
    users = log[cols["id"]].astype(str).map(idx)
    keep = users.notna().values
    users = users[keep].astype(int).values
    epochs = log[cols["epoch"]].values[keep]
    kids = [ks.key_id(u, t) for u, t in zip(users, epochs)]
    U, V, C2 = [], [], []
    for s in range(0, len(kids), 5000):
        u, v, c = op.make_handles(pke, kids[s:s + 5000], rng)
        U.append(u); V.append(v); C2 += c
    U, V = np.vstack(U), np.vstack(V)
    n = len(kids)
    req = np.nonzero(rng.random(n) < P["open_request_rate"])[0]
    res = dict(transcripts=n, requests=len(req), opened=0, correct=0, judge_ok=0, unauth_opened=0,
               tampered=0, tampered_opened=0, nonsigner_judged=0, wrong_proof_judged=0)
    for i in req:
        tid = int(i)
        tag = op.issue_authz(gov, tid, 0)
        if opener.open(tid, 1, rng.bytes(32), U[i], V[i], C2[i]) is not None:   # forged authorisation
            res["unauth_opened"] += 1
        if opener.open(tid, 2, None, U[i], V[i], C2[i]) is not None:            # no authorisation
            res["unauth_opened"] += 1
        r = opener.open(tid, 0, tag, U[i], V[i], C2[i])
        if r is None:
            continue
        kid, pi = r
        res["opened"] += 1
        res["correct"] += int(kid == kids[i])
        res["judge_ok"] += int(op.judge_open(pke, U[i], V[i], C2[i], kids[i], pi))
        other = kids[(i + 1) % n] if kids[(i + 1) % n] != kids[i] else kids[(i + 7) % n]
        res["nonsigner_judged"] += int(op.judge_open(pke, U[i], V[i], C2[i], other, pi))
        res["wrong_proof_judged"] += int(op.judge_open(pke, U[i], V[i], C2[i], kids[i], rng.bytes(32)))
        if rng.random() < P["tamper_rate"]:
            Vt = V[i].copy(); j = int(rng.integers(len(Vt))); Vt[j] = (Vt[j] + 1) % op.Q
            res["tampered"] += 1
            res["tampered_opened"] += int(opener.open(tid, 3, op.issue_authz(gov, tid, 3), U[i], Vt, C2[i]) is not None)
    logged = pd.Series([x[0] for x in opener.log if x[1] == 0]).value_counts()
    res["logged_once"] = int((logged == 1).sum())
    res["log_duplicates"] = int((logged > 1).sum())
    res["log_chain_ok"] = bool(opener.log_consistent())
    return res


# --------------------------------------------------------------------------------------
# Attack search (proposer-checker with a fixed proposal set) and stage costs
# --------------------------------------------------------------------------------------


def attack_search(dom, R, slots, near, G, active, prior, tau_hat, rng, sims):
    deg = np.diff(G.indptr)
    bsize = np.bincount(dom.bucket, minlength=len(slots))
    props = {
        "max-degree": lambda Rr: Rr[np.argmax(deg[Rr])],
        "min-degree": lambda Rr: Rr[np.argmin(deg[Rr])],
        "largest-bucket": lambda Rr: Rr[np.argmax(bsize[dom.bucket[Rr]])],
        "smallest-bucket": lambda Rr: Rr[np.argmin(bsize[dom.bucket[Rr]])],
    }
    hits = {k: 0 for k in props}
    elig = np.nonzero(prior > 0)[0]
    done = 0
    for _ in range(sims):
        u = int(rng.choice(elig))
        ring, _ = synthesize(dom, u, slots, near, active, rng, int(slots.sum()) // 2)
        if ring is None:
            continue
        done += 1
        for k, f in props.items():
            hits[k] += int(f(ring.real) == u)
    best = max(hits.values()) / max(done, 1)
    return max(tau_hat, best), hits


def stage_costs(dom, plan, P, rng, runs, sims):
    slots = plan["slots"]
    active = np.ones(dom.N, dtype=bool)
    elig = slots[dom.bucket] > 0
    prior = elig / elig.sum()
    syn, cer, sea, tas = [], [], [], []
    for _ in range(runs):
        t = epoch_list(dom)[int(rng.integers(len(epoch_list(dom))))]
        G = snapshot_at(dom, t)
        near = dom.near(t, G, P["d0"])
        u0 = int(rng.choice(np.nonzero(elig)[0]))
        t0 = time.perf_counter()
        ring, _ = synthesize(dom, u0, slots, near, active, rng, plan["n"] // 2)
        t1 = time.perf_counter()
        if ring is None:
            continue
        post = certify(dom, ring.real, slots, near, active, prior)
        t2 = time.perf_counter()
        tau_as, _ = attack_search(dom, ring.real, slots, near, G, active, prior, float(post.max()), rng, sims)
        t3 = time.perf_counter()
        syn.append(t1 - t0); cer.append(t2 - t1); sea.append(t3 - t2); tas.append(tau_as >= post.max() - 1e-12)
    return dict(synth_s=float(np.mean(syn)), cert_s=float(np.mean(cer)), search_s=float(np.mean(sea)),
                tau_as_ge_tau=float(np.mean(tas)), runs=len(syn))
