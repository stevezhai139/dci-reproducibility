#!/usr/bin/env python3
"""contrast_preflight.py — design search for a live CONTRAST schedule (PG).

Paper 3C §6.10 extension: find a post-onset query mix whose deviation from
the steady calibration is a contrast of the sampling noise — per-axis z
below the Bonferroni cheap thresholds (union structurally quiet), joint
statistic over threshold (full fires), cumulative alignment R < rho with
significant whitened presence (router escalates and catches it live).

Pure offline (no DB): mirrors calibrate_dci_gate()/run_block() template
draws with the synthetic exec times of part2_validate_offline.py (imported,
not copied). NO live run until this preflight passes on the locked env.
Sandbox previews are DEVELOPMENT data; the official preflight record is the
locked-env run (RUN_HANDOFF rule).

Schedule shape searched: 24 windows = 12 steady (tpch_mixed MixBase,
qpw=20) + 12 post-onset candidate windows (same templates or one
substitution, same qpw -> S_V frozen). Knobs: weight profile (rank-tie
structure), permutation (which template is heavy), one-template
substitution (S_A/S_P level).

Usage: python3 contrast_preflight.py <repro_root> [--seeds 20] [--cal 64]
                                     [--out contrast_preflight.csv]
"""
from __future__ import annotations
import argparse, csv, random, sys
from pathlib import Path
import numpy as np

STEADY_N, POST_N = 12, 12
RHO, ALPHA, AUDIT = 0.35, 0.05, 8

PROFILES = {
    "cal":   [3.0, 2.0, 2.0, 2.0, 1.0],
    "tie":   [2.2, 2.1, 2.0, 1.9, 1.8],
    "sharp": [6.0, 3.0, 1.5, 1.0, 0.5],
    "two":   [4.0, 4.0, 1.0, 0.6, 0.4],
    "mild":  [2.6, 2.2, 2.0, 1.7, 1.5],
    "flat":  [1.0, 1.0, 1.0, 1.0, 1.0],
    "topt":  [3.2, 3.0, 2.8, 1.0, 0.8],
    "topt2": [2.8, 2.6, 2.4, 1.4, 1.2],
}
PERMS = {"id": [0, 1, 2, 3, 4], "rev": [4, 3, 2, 1, 0], "hswap": [4, 1, 2, 3, 0]}
# (name, profile, perm, substitute Q12->Q4?, mode, lam_max, ramp_len)
# mode "step": post windows drawn from the candidate mix outright.
# mode "ramp": per-query Bernoulli mixture (the live analogue of the Redset
# partial migrations, Sec 6.5): window t of the post segment draws each query
# from the candidate mix w.p. lam_t = lam_max*min(1,(t+1)/ramp_len), else
# from steady. Sustained sub-sigma per-axis steps in a fixed direction ->
# the accumulating router integrates what per-window axis tests cannot see;
# lam_max below the union's saturation point keeps the union blind for good.
def _grid_profiles():
    """Structured profile grid: k near-tied heavy templates carrying mass M,
    tail sharing 1-M with minimum probability p_min. Axis signatures:
    k ties -> S_R churn; M vs cal's 0.7 top-3 -> S_T level; p_min vs cal's
    0.1 -> S_A absence churn."""
    out = {}
    for k in (2, 3):
        for M in (0.75, 0.82, 0.90):
            for pmin in (0.04, 0.06, 0.08):
                top = [M / k * (1 + 0.02 * (k - i)) for i in range(k)]
                nt = 5 - k
                rest = 1.0 - sum(top)
                tail = np.linspace(rest - pmin * (nt - 1) / 2 * 0 + 0, 0, 0)  # placeholder
                # tail: linear from (rest - pmin*(nt-1)) .. down to pmin, normalised to rest
                raw = np.linspace(1.0, 0.4, nt); raw = raw / raw.sum() * rest
                raw = np.maximum(raw, pmin); raw = raw / raw.sum() * rest
                w = np.array(top + raw.tolist())
                out[f"g{k}_{int(M*100)}_{int(pmin*100)}"] = (w / w.sum()).tolist()
    return out

PROFILES.update(_grid_profiles())

# ── engineered table-coincident family ────────────────────────────────
# The S_A axis is quasi-deterministic: a low-mass template absent from a
# window removes its tables from the union -> single-axis A spike that the
# union test sees (measured: dbar_A -1.4..-6.2 dominates every strong mix
# candidate). Kill the channel structurally: active set chosen so absence
# is table-invisible. Top pair (Q14,Q17) have IDENTICAL tables
# {lineitem,part} -> near-tie rank churn moves S_R without touching S_A;
# tail (Q1,Q6) are lineitem-only (subset of any window's union) -> their
# absence moves nothing; Q3 alone carries {customer,orders} -> its p sets
# a small, tunable A component. Signature: R down (top-tie churn),
# T up (mass concentration), A ~ 0(+eps) -- opposing signs on the
# r=0.67 (S_R,S_T) calibration correlation = whitened contrast.
# S_A reads COLUMNS too (sa_v2(tables, cols)): the (Q14,Q17) "identical
# tables" pair still churns A through disjoint column sets. The truly
# A-invisible tie pair is (Q1,Q6): same table {lineitem} and Q6's columns
# are a SUBSET of Q1's -- their rank flips move nothing in A. The fade
# family therefore glides to a {Q1,Q6,Q3} mix: top-tie churn (R down),
# concentration 0.5->~0.85 (T up), absence churn collapses (A UP -- no
# low-p table carriers left). Sign pattern (-,+,+) is off-ridge on BOTH
# strong calibration correlations (R,T)=0.67, (R,A)=0.63 -> maximal
# whitened boost at sub-threshold per-axis z. "glide" = ramp with
# lam_max=1: per-query Bernoulli fade over ramp_len windows (kills the
# boundary transient the union was catching), then hold.
ENG = {
    "e2_78_15": (["Q14","Q17","Q3","Q1","Q6"], [0.397,0.383,0.15,0.07,0.05]),
    "f3_85_15": (["Q1","Q6","Q3"], [0.44,0.41,0.15]),
    "f3_80_20": (["Q1","Q6","Q3"], [0.42,0.38,0.20]),
    "f3_88_12": (["Q1","Q6","Q3"], [0.45,0.43,0.12]),
    "f4_80_15": (["Q1","Q6","Q3","Q12"], [0.42,0.38,0.15,0.05]),
}
# Official preflight set (design frozen from the 2026-08-12 dev search;
# see the candidate-family comments above for the elimination trail):
# NULL = FA floor probe; X0r = pure query-rewrite canary (B templates
# textually new, same tables/cols/plans); X1r = one-relation view clone.
CANDIDATES = [
    ("NULL",    "cal", "id", False, "step",   0.0,  1),
    ("X0r_l05", "cal", "0r", False, "twopop", 0.05, 2),
    ("X0r_l06", "cal", "0r", False, "twopop", 0.06, 2),
    ("X0r_l07", "cal", "0r", False, "twopop", 0.07, 2),
    ("X1r_l06", "cal", "1r", False, "twopop", 0.06, 2),
]


def load_module(path: Path, name: str, argv=None):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    if argv is not None:
        save, sys.argv = sys.argv, argv
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.argv = save
    else:
        spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repro", type=Path)
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--cal", type=int, default=64)
    ap.add_argument("--out", default="contrast_preflight.csv")
    ap.add_argument("--steady", type=int, default=STEADY_N)
    ap.add_argument("--post", type=int, default=POST_N)
    ap.add_argument("--prompt", type=int, default=3,
                    help="detection = fired within first PROMPT post windows")
    ap.add_argument("--compose", action="store_true",
                    help="measure atom signatures and append composed candidates")
    a = ap.parse_args()

    sys.path.insert(0, str(a.repro))
    pv = load_module(a.repro / "part2_validate_offline.py", "pvo")
    pgdir = a.repro / "end_to_end" / "postgres"
    sys.path.insert(0, str(pgdir))
    pga = load_module(pgdir / "pg_adaptation.py", "pga_contrast",
                      argv=["pg_adaptation.py", "--workload", "tpch_mixed"])
    from dci_gate_v3 import DCIGateV3

    ph0 = pga.PHASES[0]
    QS0 = list(ph0["qs"]); QPW = int(ph0.get("qpw", pga.QUERIES_PW))
    W0 = np.array(ph0["w"], float); W0 /= W0.sum()

    def draw(qs, w, trng):
        qn = list(np.random.choice(qs, size=QPW, p=w))
        ex = [pv.synth_ms(q, trng) for q in qn]
        return pga.make_window_features(qn, ex)

    def feats(prev, cur):
        b = pga.compute_hsm_breakdown(prev, cur)
        return np.array([b["S_R"], b["S_V"], b["S_T"], b["S_A"], b["S_P"]])

    def cand_mix(profile, perm, sub):
        if perm in RELSETS:            # twopop: perm slot encodes clone set
            w = np.array(PROFILES[profile], float)
            return list(QS0), w / w.sum()
        if profile in ENG:
            qs, w = ENG[profile]
            w = np.array(w, float)
            w = np.maximum(w, 1e-9)
            return qs, w / w.sum()
        qs = list(QS0)
        if sub:
            qs[qs.index("Q12")] = "Q4"
        w = np.array([PROFILES[profile][PERMS[perm].index(i)] for i in range(5)], float)
        return qs, w / w.sum()

    def draw_ramp(qs_c, w_c, lam, trng):
        n_b = int(round(lam * QPW))
        qn = (list(np.random.choice(QS0, size=QPW - n_b, p=W0)) +
              list(np.random.choice(qs_c, size=n_b, p=w_c)))
        ex = [pv.synth_ms(q, trng) for q in qn]
        return pga.make_window_features(qn, ex)

    # cal-noise structure (one probe seed): the exploitable contrast boost
    np.random.seed(909999); trng0 = random.Random(909999)
    prev = draw(QS0, W0, trng0); Xp = []
    for _ in range(200):
        cur = draw(QS0, W0, trng0); Xp.append(feats(prev, cur)); prev = cur
    C4 = np.corrcoef(np.asarray(Xp)[:, [0, 2, 3]].T)
    print("cal corr (S_R,S_T,S_A):")
    print(np.round(C4, 2))

    # ── stage A: atomic mix perturbations, measured signatures ──────
    # Each atom is a full mix (support may differ from steady). The search
    # measures dbar (per-axis standardised post-hold deviation) per atom,
    # then solves a box-constrained least-squares for a composition
    # alpha (in mix space, renormalised) whose PREDICTED signature points
    # off-ridge; the composed candidate is then verified directly through
    # the full pipeline (no linearity assumption in the verdict).
    ATOMS = {
        "sup3":  (["Q1","Q6","Q3"], [0.44,0.41,0.15]),
        "sup7":  (["Q1","Q6","Q14","Q3","Q12","Q5","Q10"],
                  [0.24,0.16,0.16,0.16,0.08,0.10,0.10]),
        "conc":  (["Q1","Q6","Q14","Q3","Q12"], [0.42,0.40,0.08,0.06,0.04]),
        "flat":  (["Q1","Q6","Q14","Q3","Q12"], [0.2,0.2,0.2,0.2,0.2]),
        "addQ4": (["Q1","Q6","Q14","Q3","Q12","Q4"],
                  [0.25,0.17,0.17,0.17,0.09,0.15]),
        "swapT": (["Q1","Q6","Q17","Q3","Q12"], [0.3,0.2,0.2,0.2,0.1]),
        "q3up":  (["Q1","Q6","Q14","Q3","Q12"], [0.22,0.14,0.14,0.40,0.10]),
        "tailup":(["Q1","Q6","Q14","Q3","Q12"], [0.16,0.16,0.16,0.16,0.36]),
    }

    def measure_sig(qs_c, w_c, n_seeds=4, n_post=10):
        """Mean standardised deviation of hold windows for one mix."""
        acc = np.zeros(5); n = 0
        for seed in range(n_seeds):
            np.random.seed(920000 + seed); trng = random.Random(920000 + seed)
            prev = draw(QS0, W0, trng); X = []
            for _ in range(a.cal):
                cur = draw(QS0, W0, trng); X.append(feats(prev, cur)); prev = cur
            X = np.asarray(X)
            mu0, sd0 = X.mean(0), X.std(0, ddof=1)
            sd0[sd0 < 1e-12] = 1e-12
            prev = draw(qs_c, np.asarray(w_c) / np.sum(w_c), trng)
            for _ in range(n_post):
                cur = draw(qs_c, np.asarray(w_c) / np.sum(w_c), trng)
                acc += (feats(prev, cur) - mu0) / sd0; n += 1
                prev = cur
        return acc / n

    if a.compose:
        print("[atoms] measuring signatures (dev heuristic stage)...")
        base_u = sorted({q for q in QS0} | {q for qs_c, _ in ATOMS.values() for q in qs_c})
        def to_vec(qs_c, w_c):
            v = np.zeros(len(base_u))
            for q, w in zip(qs_c, np.asarray(w_c, float) / np.sum(w_c)):
                v[base_u.index(q)] = w
            return v
        v0 = to_vec(QS0, W0)
        sigs, dirs = {}, {}
        for nm, (qs_c, w_c) in ATOMS.items():
            sig = measure_sig(qs_c, w_c)
            sigs[nm] = sig; dirs[nm] = to_vec(qs_c, w_c) - v0
            print(f"  {nm:<8} dbar/sd {np.round(sig, 2).tolist()}")
        # target: off-ridge cheap pattern, S_V untouched, |z|<=1.3
        names = list(ATOMS)
        S = np.stack([sigs[n][[0, 2, 3]] for n in names])   # (n_atoms, 3) R,T,A
        for tgt in ([-1.3, 1.0, 1.0], [1.3, -1.0, -1.0], [-1.3, 1.2, 0.0], [1.0, -1.3, 1.0]):
            t = np.asarray(tgt, float)
            # box-constrained LS via projected gradient (alpha in [0,0.9])
            al = np.full(len(names), 0.1)
            for _ in range(4000):
                r = (al @ S) - t
                al = np.clip(al - 0.01 * (S @ r), 0.0, 0.9)
            pred = al @ S
            mix = v0 + sum(al[i] * dirs[n] for i, n in enumerate(names))
            mix = np.clip(mix, 0.0, None)
            if mix.sum() <= 0:
                continue
            mix /= mix.sum()
            keep = mix > 0.005
            qs_c = [base_u[i] for i in range(len(base_u)) if keep[i]]
            w_c = [float(mix[i]) for i in range(len(base_u)) if keep[i]]
            key = f"C_{'_'.join(f'{x:+.1f}' for x in tgt)}"
            ENG[key] = (qs_c, w_c)
            CANDIDATES.append((key, key, "id", False, "ramp", 1.0, 4))
            print(f"  {key}: pred(R,T,A)={np.round(pred,2).tolist()} "
                  f"alpha={np.round(al,2).tolist()}")
            print(f"    mix: {dict(zip(qs_c, np.round(w_c,3)))}")

    # ── two-population splice (blue-green schema migration) ─────────
    # Hypothesis from three exhausted single-population searches + Redset:
    # off-axis directions need CROSS-POPULATION mixes. Population B = the
    # same five steady templates re-pointed at a cloned schema (live:
    # CREATE VIEW s2.* -> tables; zero storage): distinct template ids,
    # distinct table/col ids, same synth-time base (same plans). Post
    # onset: per-query Bernoulli mixture glide 4 -> hold at lam_max, i.e.
    # the live analogue of Sec 6.5's partial migrations.
    CLONE = {}
    for q in QS0:
        q2 = q + "v2"
        CLONE[q2] = q
        if q2 not in pga.ALL_TEMPLATES:
            pga.ALL_TEMPLATES.append(q2)
    _synth_orig = pv.synth_ms
    def synth2(key, trng, **kw):
        return _synth_orig(CLONE.get(key, key), trng, **kw)
    QS2 = [q + "v2" for q in QS0]
    _COLPFX = {"l_": "lineitem", "o_": "orders", "p_": "part", "c_": "customer",
               "ps_": "partsupp", "s_": "supplier", "n_": "nation", "r_": "region"}

    def set_clone(rels):
        """B-population metadata: same logical footprint except `rels`,
        which move to the s2.* schema (live: views over the migrated
        relations). Template ids always differ (new query shapes)."""
        def cmap(c):
            for pfx, tab in _COLPFX.items():
                if c.startswith(pfx):
                    return ("s2." + c) if tab in rels else c
            return c
        for q in QS0:
            q2 = q + "v2"
            pga.QUERY_TABLES[q2] = {("s2." + t if t in rels else t)
                                    for t in pga.QUERY_TABLES.get(q, set())}
            pga.QUERY_COLS[q2] = {cmap(c) for c in pga.QUERY_COLS.get(q, set())}
    RELSETS = {"0r": set(), "1r": {"lineitem"}, "2r": {"lineitem", "orders"},
               "4r": {"lineitem", "orders", "part", "customer"}}

    def draw_twopop(lam, trng, w_b=None):
        n_b = int(np.random.binomial(QPW, lam))   # honest canary jitter
        wb = W0 if w_b is None else w_b
        qn = (list(np.random.choice(QS0, size=QPW - n_b, p=W0)) +
              (list(np.random.choice(QS2, size=n_b, p=wb)) if n_b else []))
        ex = [synth2(q, trng) for q in qn]
        return pga.make_window_features(qn, ex)

    rows = []
    floor = None
    for name, prof, perm, sub, mode, lam_max, ramp_len in CANDIDATES:
        if mode == "twopop":
            set_clone(RELSETS.get(perm, RELSETS["4r"]))
        qs_c, w_c = cand_mix(prof, perm, sub)
        agg = {k: [] for k in ("fa_ch", "fa_fu", "fa_gt", "det_ch", "det_fu", "det_gt",
                               "dly_fu", "dly_gt", "esc", "R_end", "pres", "margin")}
        dbar = np.zeros(5); nd = 0
        for seed in range(a.seeds):
            np.random.seed(910000 + seed); trng = random.Random(910000 + seed)
            # calibration (own stream, frozen gates — mirrors live)
            prev = draw(QS0, W0, trng); X = []
            for _ in range(a.cal):
                cur = draw(QS0, W0, trng); X.append(feats(prev, cur)); prev = cur
            X = np.asarray(X)
            gates = {"gt": DCIGateV3(rho=RHO, alpha=ALPHA, audit_every=AUDIT).fit(X),
                     "ch": DCIGateV3(rho=RHO, alpha=ALPHA, audit_every=None, force="cheap").fit(X),
                     "fu": DCIGateV3(rho=RHO, alpha=ALPHA, audit_every=None, force="full").fit(X)}
            mu0 = gates["gt"].mu0.copy(); sd0 = gates["gt"].sigma0.copy()
            # trajectory: 12 steady + 12 post, same stream to all arms
            fires = {k: [] for k in gates}; esc = 0; margins = []
            prev = draw(QS0, W0, trng)
            for t in range(a.steady + a.post):
                post = t >= a.steady
                if not post:
                    cur = draw(QS0, W0, trng)
                elif mode == "step":
                    cur = draw(qs_c, w_c, trng)
                elif mode == "twopop":
                    lam_t = lam_max * min(1.0, (t - a.steady + 1) / ramp_len)
                    wb = np.array(PROFILES[prof], float)[:len(QS2)]
                    cur = draw_twopop(lam_t, trng, wb / wb.sum())
                else:
                    lam_t = lam_max * min(1.0, (t - a.steady + 1) / ramp_len)
                    cur = draw_ramp(qs_c, w_c, lam_t, trng)
                fv = feats(prev, cur); prev = cur
                for k, g in gates.items():
                    f = g.decide(fv)
                    fires[k].append((t, f, dict(g.last)))
                if post:
                    L = gates["gt"].last
                    if L["regime"] == "full" and not L["audit"]:
                        esc += 1
                    margins.append(gates["ch"].last["statistic"] / gates["ch"].last["threshold_F"])
                    dbar += (fv - mu0) / sd0; nd += 1
            for k in gates:
                agg["fa_" + k].append(sum(f for t, f, _ in fires[k][:a.steady]))
                post_f = [t - a.steady for t, f, _ in fires[k][a.steady:] if f]
                agg["det_" + k].append(int(bool(post_f and post_f[0] < a.prompt)))
                if k in ("fu", "gt") and post_f:
                    agg["dly_" + k].append(post_f[0])
            agg["esc"].append(esc / a.post)
            Lend = gates["gt"].last
            agg["R_end"].append(Lend["R4s"]); agg["pres"].append(int(Lend["has_signal"]))
            agg["margin"].append(float(np.median(margins)))
        m = {k: (float(np.mean(v)) if v else float("nan")) for k, v in agg.items()}
        m["R_end"] = float(np.median(agg["R_end"]))
        if name == "NULL":
            floor = {k: m["det_" + k] for k in ("ch", "fu", "gt")}
            floor["esc"] = m["esc"]
            ok = False
        else:
            # Frontier criteria (2026-08-12 official rev): for PERSISTENT
            # changes any nonzero-mean axis makes the union consistent via
            # its noise tail (P(fire) = 1-(1-Phi(|z|-z_a))^n; verified
            # against the official run: z=-1.38 -> 0.51 predicted, 0.52
            # measured). Binary blindness is a transient-only notion; the
            # correct live statistic is net-of-floor detection RATE and
            # DELAY. Routing must dominate the frontier, not zero out the
            # union.
            fl = floor if floor else {"ch": 0.0, "fu": 0.0, "gt": 0.0, "esc": 0.0}
            net_gt = m["det_gt"] - fl["gt"]
            net_ch = max(m["det_ch"] - fl["ch"], 0.04)
            ok = (net_gt >= 0.55 and net_gt / net_ch >= 2.5
                  and (not np.isnan(m["dly_gt"]) and m["dly_gt"] <= 3.5)
                  and m["R_end"] < RHO and m["pres"] >= 0.7
                  and m["esc"] >= 2.0 * fl.get("esc", 0.0))
        rows.append({"cand": name, "PASS": int(ok), **{k: round(v, 3) for k, v in m.items()},
                     "dbar": np.round(dbar / max(nd, 1), 2).tolist()})

    rows.sort(key=lambda r: (r["cand"] != "NULL", -r["PASS"], r["det_ch"] - r["det_gt"]))
    hdr = (f'{"cand":<10}{"PASS":>5}{"det_ch":>7}{"det_fu":>7}{"det_gt":>7}{"dly_gt":>7}'
           f'{"esc":>6}{"R_end":>7}{"pres":>6}{"margin":>8}{"fa_ch":>7}{"fa_gt":>6}  dbar/sd [R,V,T,A,P]')
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f'{r["cand"]:<10}{r["PASS"]:>5}{r["det_ch"]:>7}{r["det_fu"]:>7}{r["det_gt"]:>7}'
              f'{r["dly_gt"]:>7}{r["esc"]:>6}{r["R_end"]:>7}{r["pres"]:>6}{r["margin"]:>8}'
              f'{r["fa_ch"]:>7}{r["fa_gt"]:>6}  {r["dbar"]}')
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[out] {a.out}")
    passing = [r for r in rows if r["PASS"]]
    if passing:
        best = passing[0]
        nm, prof, perm, sub, mode, lam_max, ramp_len = next(
            c for c in CANDIDATES if c[0] == best["cand"])
        qs_c, w_c = cand_mix(prof, perm, sub)
        print("\n[winner] tpch_contrast post-onset target "
              f"(mode={mode}, lam_max={lam_max}, ramp_len={ramp_len}):")
        print(f'  {{"name": "Contrast_{nm}", "qs": {qs_c}, "w": {np.round(w_c*10,2).tolist()}, "qpw": {QPW}}}')
    else:
        print("\n[no PASS] no candidate meets the contrast triple; widen the design grid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
