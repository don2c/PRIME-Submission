#!/usr/bin/env python3
"""Runs every experiment of the evaluation and writes LaTeX tables, the shape figure (PDF),
summary.md (numbers quoted in the text) and provenance.json to the output directory.

    python reproduce.py                       # full run, writes results/
    python reproduce.py --config config/quick.json --out results_quick
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from primering import core
from primering import experiments as ex
from primering import latex as tx
from primering.figures import shape_figure


def graph_stats(G):
    A = G.astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    tri = np.asarray((A @ A).multiply(A).sum(1)).ravel() / 2
    c = np.where(deg >= 2, 2 * tri / np.maximum(deg * (deg - 1), 1), 0.0)
    return dict(nodes=G.shape[0], edges=int(G.nnz // 2), deg=float(deg.mean()), clust=float(c.mean()))


def log_step(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = json.load(open(args.config))
    P, H = cfg["params"], dict(cfg["params"], **cfg["hardcase"])
    os.makedirs(args.out, exist_ok=True)
    W = lambda name, txt: open(os.path.join(args.out, name), "w").write(txt)
    t0 = time.time()
    seed = P["seed"]
    rng = np.random.default_rng(seed)
    doms = {d["name"]: core.load_domain(d) for d in cfg["domains"]}
    dcfg = {d["name"]: d for d in cfg["domains"]}
    order_c, order_a = cfg["order"]["calib"], cfg["order"]["aov"]
    eh_name = cfg["ehealth_domain"]
    eh = doms[eh_name]
    registry = json.load(open(cfg["policy_registry"]))
    S, prov = {}, {"config": cfg, "python": sys.version.split()[0], "steps": {}}
    log_step("data loaded", t0)

    # ---- Datasets -------------------------------------------------------------------
    rows = []
    for d in order_c:
        dom = doms[d]
        if dom.epochs:
            st = [graph_stats(G) for G in dom.epochs.values()]
            s = dict(nodes=dom.N, edges=int(round(np.mean([x["edges"] for x in st]))),
                     deg=float(np.mean([x["deg"] for x in st])), clust=float(np.mean([x["clust"] for x in st])))
        else:
            s = graph_stats(dom.G)
        rows.append(dict(s, label=dcfg[d]["label"], buckets=len(dom.bucket_names)))
    W("tab_datasets.tex", tx.datasets(rows))
    S["datasets"] = rows

    # ---- Q1 calibration, sensitivity --------------------------------------------------
    cal = {d: core.prime_runs(doms[d], P, rng, P["calib_runs"]) for d in order_c}
    W("tab_calib.tex", tx.calib(cal, order_c, P["calib_runs"], P["k"]))
    S["calib"] = {d: {k: v for k, v in cal[d].items() if not k.startswith("_")} for d in order_c}
    log_step("calibration", t0)

    sens = []
    orig = core.resolve_target
    for n in P["sens_ns"]:
        for d0 in P["sens_d0"]:
            core.resolve_target = (lambda b, a, t, Hs, k, support="uniform", n_override=None, m_override=None,
                                   _n=n: orig(b, a, t, Hs, k, support, _n, m_override))
            try:
                st = core.prime_runs(eh, dict(P, d0=d0), np.random.default_rng(seed + 200), P["sens_runs"])
            finally:
                core.resolve_target = orig
            sens.append(dict(n=n, d0=d0, tau=st["pg_tau"], succ=st["pg_success"], rel=st["released_pct"]))
    W("tab_sensitivity.tex", tx.sensitivity(sens, P["tau"], P["sens_runs"]))
    S["sensitivity"] = sens
    log_step("sensitivity", t0)

    # ---- Q2 builders ----------------------------------------------------------------
    bld = {d: core.builders_experiment(doms[d], P, rng, P["builder_runs"]) for d in order_c}
    W("tab_builders.tex", tx.builders(bld, order_c, bld[order_c[0]]["_n"]))
    S["builders"] = {d: {b: v for b, v in bld[d].items() if b != "_n"} for d in order_c}
    log_step("builders", t0)

    # ---- Sparsity and churn ---------------------------------------------------------
    std, hard, churn_detail = {}, {}, []
    for tag, PP, store in [("standard", P, std), ("hard", H, hard)]:
        for d in order_c:
            store[d] = {}
            for c in P["churn_levels"]:
                s = core.prime_runs(doms[d], PP, rng, P["churn_runs"], churn=c)
                store[d][c] = s.get("pg_tau")
                churn_detail.append(dict(setting=tag, domain=d, churn=c, tau=s.get("pg_tau"), h2=s.get("pg_h2"),
                                         n=s.get("pg_S"), cover=s.get("pg_cover"), empty=s.get("empty_all"),
                                         released_pct=s["released_pct"]))
    cd = pd.DataFrame(churn_detail)
    W("tab_prime_sparsity_churn_summary.tex",
      tx.churn(std, hard, order_c, P["churn_levels"], dict(P, n=int(cd[cd.setting == "standard"].n.max())),
               dict(H, n=int(cd[cd.setting == "hard"].n.max()))))
    cd.to_csv(os.path.join(args.out, "churn_detail.csv"), index=False)
    S["churn"] = churn_detail
    log_step("churn", t0)

    # ---- Scaling --------------------------------------------------------------------
    sc = core.scaling_experiment(eh, P, P["scaling_ns"], P["scaling_seeds"], P["scaling_sigs_per_n"], rng)
    W("tab_scaling.tex", tx.scaling(sc))
    pd.DataFrame(sc).to_csv(os.path.join(args.out, "scaling.csv"), index=False)
    S["scaling"] = sc
    log_step("scaling", t0)

    # ---- Transcript experiments per domain ------------------------------------------
    aov, shape_rows, fs, opn, plans, tstats = {}, [], {}, {}, {}, {}
    for d in order_a:
        dom = doms[d]
        lr = core.LatticeRing(seed=0)
        ks, issuer = ex.KeyStore(lr, d), ex.Issuer(lr, seed + 31)
        r = np.random.default_rng(seed + 1000 + order_a.index(d))
        plan = ex.calibrate_plan(lr, ks, issuer, dom, P, r, P["plan_calibration_sigs"])
        plans[d] = {k: v for k, v in plan.items() if k != "slots"}
        lc = cfg["logs"][d]
        log = pd.read_csv(lc["path"])
        df, st = ex.sample_transcripts(dom, log, lc, plan, lr, ks, issuer, P, r, P["aov_events"])
        tstats[d] = dict(st, mean_sign_s=float(df.sign_s.mean()), mean_verify_s=float(df.verify_s.mean()))
        aov[d] = ex.aov_experiment(df, dom.N, registry[d], P["aov_seeds"])
        log_step(f"{d}: AOV", t0)
        shape_rows += ex.shape_experiment(dom, plan, lr, ks, issuer, P, r, P["shape_churns"], P["shape_runs"])
        log_step(f"{d}: shape", t0)
        fs[d] = ex.freshness_experiment(dom, plan, lr, ks, issuer, P, r, P["fresh_per_epoch"], P["fresh_pool"],
                                        P["link_pairs"], P["link_seeds"])
        log_step(f"{d}: freshness", t0)
        opn[d] = ex.opening_experiment(dom, log, lc, ks, P, r)
        log_step(f"{d}: opening", t0)
        if d == eh_name:
            costs = ex.stage_costs(dom, plan, P, r, P["cost_runs"], P["search_sims"])
            costs.update(sign_s=tstats[d]["mean_sign_s"], verify_s=tstats[d]["mean_verify_s"], n=plan["n"])
    W("tab_prime_aov_hiding.tex", tx.aov(aov, order_a))
    shape_figure(shape_rows, order_c, os.path.join(args.out, "prime-ring-shape.pdf"))
    W("fig_prime_ring_shape.tex", tx.figure_snippet())
    pd.DataFrame(shape_rows).to_csv(os.path.join(args.out, "shape.csv"), index=False)
    W("tab_prime_epoch_fs.tex", tx.epoch_fs(fs, order_a))
    W("tab_opening.tex", tx.opening(opn, order_a))
    W("tab_overhead_by_scheme.tex", tx.overhead(costs))
    S.update(aov=aov, shape=shape_rows, freshness=fs, opening=opn, plans=plans, transcripts=tstats, costs=costs)

    # ---- BBS+ baseline (released-view recovery) -------------------------------------
    # A BBS+ presentation carries the verifier's proof request, which names the policy, so the BBS+ view is the
    # released view with the policy in the clear. Its recovery is the positive-control accuracy of the same attackers.
    bbs_policy = aov[eh_name]["Policy body"]["control"]
    bbs_attr = aov[eh_name]["Attribute value"]["chance"]
    prime_b = dict(policy=aov[eh_name]["Policy body"]["acc"], attr=aov[eh_name]["Attribute value"]["acc"],
                   tau=cal[eh_name]["pg_tau"], refusal=100.0 - cal[eh_name]["released_pct"])
    W("tab_bbs-baseline.tex", tx.bbs(prime_b, dict(policy=bbs_policy, attr=bbs_attr)))
    S["bbs"] = dict(bbs_policy=bbs_policy, bbs_attr=bbs_attr, prime=prime_b)

    # ---- Q3 ablations ---------------------------------------------------------------
    arng = lambda: np.random.default_rng(seed + 100)
    pr, at = aov[eh_name]["Policy body"]["acc"], aov[eh_name]["Attribute value"]["acc"]

    def row(st, **kw):
        return dict(policy=pr, attr=at, success=st["pg_success"], bucket=st["pg_bucket"],
                    above=st["rel_above_pct"], release=st["released_pct"], **kw)
    full = core.prime_runs(eh, P, arng(), P["calib_runs"])
    abl = {"Full PRIME-Ring": row(full)}
    ns = core.noshape_recovery(eh.log, dcfg[eh_name], rng)
    abl["No shape"] = dict(row(full), policy=ns["policy"], attr=ns["attr"], dagger=True)
    abl["No distance ($d_0=0$)"] = row(core.prime_runs(eh, P, arng(), P["calib_runs"], d0=0))
    abl["No certificate"] = row(core.prime_runs(eh, P, arng(), P["calib_runs"], gate=False))
    abl["No support control"] = row(core.prime_runs(eh, P, arng(), P["calib_runs"], support="raw"))
    abl["Signer not counted"] = row(core.prime_runs(eh, P, arng(), P["calib_runs"], signer_counted=False))
    W("tab_ablation.tex", tx.ablation(abl))
    S["ablation"] = abl
    S["noshape"] = ns
    log_step("ablation", t0)

    # ---- Summary of numbers quoted in the text --------------------------------------
    W("summary.md", summary(S, P, order_c, order_a, eh_name))
    prov["runtime_s"] = round(time.time() - t0, 1)
    json.dump(dict(prov, results=S), open(os.path.join(args.out, "provenance.json"), "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    bad = [f for f in os.listdir(args.out) if f.endswith(".tex") and ("nan" in open(os.path.join(args.out, f)).read())]
    if bad:
        print("WARNING: empty cells in", bad, file=sys.stderr)
    log_step(f"done -> {args.out}", t0)


def rng_str(vals, fmt="{:.3f}"):
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}--{fmt.format(hi)}"


def summary(S, P, order_c, order_a, eh):
    L = ["# Numbers quoted in the text", ""]
    cal = S["calib"]
    L += ["## Q1 calibration",
          f"- mean tau_hat per domain: " + ", ".join(f"{d} {cal[d]['pg_tau']:.3f}" for d in order_c),
          f"- mean tau_hat range: {rng_str([cal[d]['pg_tau'] for d in order_c], '{:.2f}')}",
          f"- mean |R|: " + ", ".join(f"{d} {cal[d]['pg_R']:.1f}" for d in order_c),
          f"- max over runs of p(u0) - tau_hat: " + ", ".join(f"{d} {cal[d]['pg_gap']:.3f}" for d in order_c),
          f"- released (%): " + ", ".join(f"{d} {cal[d]['released_pct']:.1f}" for d in order_c),
          f"- mean tau_hat inside Bayes-success CI: " + ", ".join(
              f"{d} {cal[d]['pg_ci_lo'] <= cal[d]['pg_tau'] <= cal[d]['pg_ci_hi']}" for d in order_c), ""]
    b = S["builders"]
    L += ["## Q2 builders",
          f"- Neighbour Bayes success range: {rng_str([b[d]['Neighbour']['success'] for d in order_c])}",
          f"- Uniform mean max posterior range: {rng_str([b[d]['Uniform']['maxpost'] for d in order_c])}",
          f"- PRIME mean max posterior range: {rng_str([b[d]['PRIME']['maxpost'] for d in order_c], '{:.2f}')}",
          f"- pass at tau (%): " + "; ".join(f"{d}: " + ", ".join(f"{k} {v['passed']:.1f}" for k, v in b[d].items())
                                             for d in order_c), ""]
    ch = pd.DataFrame(S["churn"])
    stdc = ch[ch.setting == "standard"]
    dev = max(g.tau.max() - g.tau.min() for _, g in ch.groupby(["setting", "domain"]))
    L += ["## Sparsity and churn",
          f"- H2 values: {sorted(set(round(x, 3) for x in ch.h2))}",
          f"- max cover fraction: {ch.cover.max():.3f}",
          f"- pool depletion (share of runs) by cell: " + "; ".join(
              f"{r.setting}/{r.domain}/{r.churn}: {r.empty:.3f}" for r in ch.itertuples() if r.empty > 0),
          f"- max change of mean tau_hat across churn within a setting and domain: {dev:.3f}",
          f"- released (%) max over all cells: {ch.released_pct.max():.1f}",
          f"- hard setting mean tau_hat range: {rng_str(ch[ch.setting == 'hard'].tau.tolist(), '{:.2f}')}", ""]
    sh = pd.DataFrame(S["shape"])
    L += ["## Constant shape (figure)",
          f"- mean timing deviation (ms) range over accepted runs: {rng_str(sh.dtheta_ms.dropna().tolist())}",
          f"- rejection causes (%): pool max {sh.pool_pct.max():.1f}, length max {sh.length_pct.max():.1f}, "
          f"time max {sh.time_pct.max():.1f}",
          "- per cell: " + "; ".join(f"{r.domain}/{r.churn}: reject {r.reject_pct:.1f} (pool {r.pool_pct:.1f}, "
                                     f"len {r.length_pct:.1f}, time {r.time_pct:.1f}), pad {r.pad_B:.1f} B"
                                     for r in sh.itertuples()),
          "- plans: " + "; ".join(f"{d}: n={p['n']}, L={p['L']} B, Theta={1000 * p['Theta']:.2f} ms"
                                  for d, p in S["plans"].items()), ""]
    c = S["costs"]
    sc = S["scaling"]
    L += ["## Overhead and scaling",
          f"- synthesis {1000 * c['synth_s']:.2f} ms, certification {1000 * c['cert_s']:.2f} ms, attack search "
          f"{1000 * c['search_s']:.1f} ms, sign {1000 * c['sign_s']:.1f} ms, verify incl. padding {1000 * c['verify_s']:.1f} ms",
          f"- tau_AS >= tau_hat in {100 * c['tau_as_ge_tau']:.0f}% of searches",
          f"- scaling synthesis (ms): " + ", ".join(f"n={r['n']}: {r['synth_ms']:.2f}" for r in sc),
          f"- certification at largest n: {sc[-1]['cert_ms']:.1f} ms", ""]
    o = S["opening"]
    L += ["## Governed opening"] + [
        f"- {d}: {o[d]['transcripts']} transcripts, {o[d]['requests']} requests, opened {o[d]['opened']}, correct "
        f"{o[d]['correct']}, JudgeOpen accepted {o[d]['judge_ok']}, unauthorised opened {o[d]['unauth_opened']}, "
        f"non-signer accepted {o[d]['nonsigner_judged']}, wrong proof accepted {o[d]['wrong_proof_judged']}, "
        f"tampered {o[d]['tampered']} (opened {o[d]['tampered_opened']}), logged once {o[d]['logged_once']}, "
        f"duplicates {o[d]['log_duplicates']}, hash chain ok {o[d]['log_chain_ok']}" for d in order_a] + [""]
    f = S["freshness"]
    L += ["## Freshness and forward security"] + [
        f"- {d}: fresh {f[d]['fresh']:.3f}, same-epoch replay {f[d]['replay_same']:.3f}, older-epoch replay "
        f"{f[d]['replay_old']:.3f}, stale status {f[d]['stale_status']:.3f}, past-epoch forgery {f[d]['forge']:.3f}, "
        f"crypto linker {f[d]['link_crypto']:.3f}, ring-overlap linker {f[d]['link_ring']:.3f} "
        f"({f[d]['transcripts']} transcripts, {f[d]['pairs']} pairs)" for d in order_a] + [""]
    a = S["aov"]
    L += ["## AOV hiding"] + [
        f"- {d}/{t}: best attacker {a[d][t]['attacker']} acc {a[d][t]['acc']:.3f}, chance {a[d][t]['chance']:.3f}, "
        f"with ring membership {a[d][t]['ring_acc']:.3f}, positive control {a[d][t]['control']:.3f}"
        for d in order_a for t in a[d]] + [
        "- transcripts: " + "; ".join(f"{d}: {v['sampled']} sampled, {v['length_rejected']} length-rejected"
                                      for d, v in S["transcripts"].items()), ""]
    ab = S["ablation"]
    L += ["## Ablations",
          f"- No shape policy {ab['No shape']['policy']:.3f}, attr {ab['No shape']['attr']:.3f} "
          f"(policy-size model: {S['noshape']['how']})",
          f"- Signer not counted: Bayes success {ab['Signer not counted']['success']:.3f}, bucket "
          f"{ab['Signer not counted']['bucket']:.3f}",
          f"- No support control: Bayes success {ab['No support control']['success']:.3f}, release "
          f"{ab['No support control']['release']:.1f}%", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
