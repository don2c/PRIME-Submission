"""LaTeX writers. Labels, columns and row order follow the manuscript."""
import math


def f3(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def f1(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def num(x):
    return f"{x:,}".replace(",", "{,}")


def wrap(label, caption, colspec, header, body, resize=True, size=r"\small"):
    out = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size]
    out += [r"\resizebox{\linewidth}{!}{%"] if resize else []
    out += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r"\\", r"\midrule"] + body
    out += [r"\bottomrule", r"\end{tabular}" + ("}" if resize else ""), r"\end{table}"]
    return "\n".join(out) + "\n"


def datasets(rows):
    body = [f"{r['label']} & {num(r['nodes'])} & {num(r['edges'])} & {r['deg']:.1f} & {r['clust']:.3f}\\\\" for r in rows]
    nb = sorted({r["buckets"] for r in rows})
    cap = (f"Datasets. Each domain has {nb[0]} attribute buckets. E-health statistics are per epoch snapshot $G_t$."
           if len(nb) == 1 else "Datasets. E-health statistics are per epoch snapshot $G_t$.")
    return wrap("tab:datasets", cap, "lrrrr", r"Domain & Nodes & Edges & Avg.\ deg. & Clustering", body, resize=False)


def aov(res, order):
    body = []
    for d in order:
        for tgt in ["Policy body", "Attribute value", "Policy mask"]:
            s = res[d][tgt]
            body.append(f"{d} & {tgt} & {s['acc']:.3f} & {s['chance']:.3f} & {s['adv']:.3f} \\\\")
    return wrap("tab:prime_aov_hiding", "Recovering policy/attribute content from the PRIME-Ring verifier view (AOV on).",
                "llrrr", r"Domain & Target & Acc. & Chance & Advantage", body)


def calib(cal, order, runs, k):
    body = [f"{d} & {f3(cal[d]['pg_tau'])} & {f3(cal[d]['pg_success'])} & [{f3(cal[d]['pg_ci_lo'])}, {f3(cal[d]['pg_ci_hi'])}]\\\\"
            for d in order]
    return wrap("tab:calib", f"Certificate calibration ($k={k}$, {num(runs)} runs per domain, all synthesised rings). "
                "Lemma~1 predicts Bayes success equal to mean $\\hat{\\tau}$.", "lrrr",
                r"Domain & Mean $\hat{\tau}$ & Bayes success & 95\% CI", body, resize=False)


def sensitivity(rows, tau, runs):
    body = [f"{r['n']} & {r['d0']} & {r['tau']:.3f} & {r['succ']:.3f} & {r['rel']:.1f} \\\\" for r in rows]
    return wrap("tab:sensitivity", f"Sensitivity of the certificate to ring size $n$ and distance rule $d_0$ "
                f"(E-health, $\\tau={tau}$, {runs} runs each).", "rrrrr",
                r"$n$ & $d_0$ & Mean $\hat{\tau}$ & Bayes success & Released (\%)", body, resize=False)


def builders(bld, order, n):
    body, first = [], True
    for d in order:
        if not first:
            body.append(r"\midrule")
        first = False
        for b in ["Uniform", "Neighbour", "Same-bucket", "PRIME"]:
            s = bld[d][b]
            body.append(f"{d} & {b} & {s['maxpost']:.3f} & {s['success']:.3f} & {s['bucket']:.3f}\\\\")
    return wrap("tab:builders", f"Signer inference by ring builder (same pools, signers and $n={n}$). Lower is better.",
                "llrrr", r"Domain & Builder & Mean max posterior & Bayes success & Bucket recovery", body)


def churn(std, hard, order, levels, P, H):
    lv = " & ".join(f"{c:.1f}" for c in levels)
    body = [f"{d} & " + " & ".join(f3(std[d][c]) for c in levels) + " & " + " & ".join(f3(hard[d][c]) for c in levels) + r"\\"
            for d in order]
    m = len(levels)
    cap = (f"Mean $\\hat{{\\tau}}$ over all synthesised rings, before the release gate, by churn level. "
           f"Standard: $\\tau={P['tau']}$, $d_0={P['d0']}$, $|S'|={P['n']}$. Hard: $\\tau={H['tau']}$, $d_0={H['d0']}$, "
           f"$|S'|={H['n']}$. Both use $H^\\star={P['Hstar']}$.")
    out = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{cap}}}", r"\label{tab:prime_sparsity_churn_summary}",
           r"\small", r"\setlength{\tabcolsep}{4pt}", rf"\begin{{tabular}}{{l{'r' * 2 * m}}}", r"\toprule",
           rf" & \multicolumn{{{m}}}{{c}}{{Standard}} & \multicolumn{{{m}}}{{c}}{{Hard}}\\",
           rf"\cmidrule(lr){{2-{m + 1}}}\cmidrule(lr){{{m + 2}-{2 * m + 1}}}", rf"Domain & {lv} & {lv}\\", r"\midrule"]
    out += body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def overhead(c):
    ms = {k: 1000 * c[k] for k in ["sign_s", "verify_s", "synth_s", "cert_s", "search_s"]}
    body = [f"PRIME-Ring & {ms['sign_s']:.1f} & {ms['verify_s']:.1f} & {ms['synth_s']:.2f} & {ms['cert_s']:.2f} & {ms['search_s']:.1f}\\\\"]
    out = [r"\begin{table}[htbp]", r"\centering",
           rf"\caption{{Mean wall-clock time per transcript (E-health, $n={c['n']}$). Verify includes plan padding to $\Theta$.}}",
           r"\label{tab:overhead_by_scheme}", r"\small", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrr}",
           r"\toprule", r" & \multicolumn{2}{c}{{Native protocol}} & \multicolumn{3}{c}{{PRIME-Ring stages}}\\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}", r"Scheme & Sign (ms) & Verify (ms) & Synthesis (ms) & Cert.\ (ms) & Search (ms)\\",
           r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out) + "\n"


def scaling(rows):
    body = [f"{r['n']} & {r['sigma_kB']:.1f} & {r['pad_B']:.0f}  & {r['cert_ms']:.1f} & {r['sign_ms']:.1f} & {r['verify_ms']:.1f}\\\\"
            for r in rows]
    return wrap("tab:scaling", "Cost by ring size $n$ (E-health, mean of five seeds). $|\\Sigma|$ is the plan length $L$ "
                "(99th percentile of $L_S$).", "rrrrrr",
                r"$n$ & $|\Sigma|$ (kB) & Pad (B) & Certify (ms) & Sign (ms) & Verify (ms)", body)


def epoch_fs(fs, order):
    body = [f"{d} & {fs[d]['fresh']:.3f} & {fs[d]['link_crypto']:.3f} & {fs[d]['link_ring']:.3f} & "
            f"{fs[d]['link_chance']:.3f} \\\\" for d in order]
    ep = fs[order[0]]["epochs"]
    return wrap("tab:prime_epoch_fs", f"Freshness and past-epoch linking (implementation, {ep} epochs). "
                "Key linker uses keys and proof bytes. Ring linker uses shared ring members.", "lrrrr",
                r"Domain & Fresh acc. & Link acc.\ (keys) & Link acc.\ (ring) & Link chance", body)


def bbs(prime, bbs_):
    out = [r"\begin{table}[htbp]", r"\centering", r"\caption{Anonymous-credential baseline under the same attacker view.}",
           r"\label{tab:bbs-baseline}", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.05}",
           r"\begin{tabular}{lcc}", r"\toprule",
           r"Metric & BBS+ cred.~\cite{flamini2024selective,w3c2025bbs} & PRIME-Ring \\", r"\midrule",
           rf"Policy recovery & {bbs_['policy']:.3f}$^{{\dagger}}$ & {prime['policy']:.3f} \\",
           rf"Hidden-attribute recovery & {bbs_['attr']:.3f}$^{{\ddagger}}$ & {prime['attr']:.3f} \\",
           rf"Signer inference & Uncertified & {prime['tau']:.3f} \\",
           rf"Release refusal & No gate & {prime['refusal']:.0f}\% \\", r"\bottomrule", r"\end{tabular}", "",
           r"\vspace{2pt}", r"\begin{minipage}{0.95\columnwidth}", r"\scriptsize",
           r"$^{\dagger}$Disclosed policy claims are recoverable by selective disclosure. "
           r"$^{\ddagger}$Equals the e-health chance level (Table~\ref{tab:prime_aov_hiding}).",
           r"\end{minipage}", r"\end{table}"]
    return "\n".join(out) + "\n"


def ablation(abl):
    body = []
    for name, s in abl.items():
        dag = r"$^{\dagger}$" if s.get("dagger") else ""
        body.append(f"{name} & {s['policy']:.3f}{dag} & {s['attr']:.3f}{dag} & {s['success']:.3f} & {s['bucket']:.3f} & "
                    f"{s['above']:.1f} & {s['release']:.1f}\\\\")
    return wrap("tab:ablation", r"Ablations on E-health, one component removed per row. $^{\dagger}$Classifier on unpadded length.",
                "lrrrrrr", r"Variant & Policy rec. & Attr.\ rec. & Bayes success & Bucket rec. & Released above $\tau$ (\%) & Release rate (\%)",
                body)


def opening(res, order):
    body = [f"{d} & {num(res[d]['transcripts'])} & {num(res[d]['requests'])} & {num(res[d]['opened'])} & "
            f"{res[d]['unauth_opened']} & {res[d]['nonsigner_judged']} & {res[d]['tampered_opened']} & "
            f"{num(res[d]['logged_once'])} & {res[d]['log_duplicates']}\\\\" for d in order]
    return wrap("tab:opening", "Governed opening and audit log.", "lrrrrrrrr",
                r"Domain & Transcripts & Requests & Opened & Unauth.\ opened & Non-signer & Tampered opened & Logged once & Duplicates",
                body)


def figure_snippet():
    return "\n".join([r"\begin{figure}[htbp]", r"\centering", r"\includegraphics[width=0.7\linewidth]{prime-ring-shape.pdf}",
                      r"\caption{Constant-shape enforcement by churn level. Bars show the share of transcripts rejected (left axis), "
                      r"and lines show mean pad bytes $\bar b$ (right axis).}", r"\label{fig:prime_ring_shape}",
                      r"\end{figure}"]) + "\n"
