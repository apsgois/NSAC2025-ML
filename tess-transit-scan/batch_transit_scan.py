
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_transit_scan_v2.py — varredura leve + plots dos top-N por planet_score

Uso:
  python batch_transit_scan_v2.py --targets targets.txt --sector 13 --out scored.csv --plot_top_n 12

Requisitos:
  pip install lightkurve astropy numpy scipy matplotlib
  # opcional: pip install transitleastsquares
"""

import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
from lightkurve import search_lightcurve

# TLS opcional
try:
    from transitleastsquares import transitleastsquares as TLS
    HAS_TLS = True
except Exception:
    HAS_TLS = False

def robust_normalize_to_one(flux):
    f = np.array(flux, dtype=float)
    # Se for ppm-like (amplitude grande, mediana ~0), converte aprox. p/ relativo
    if np.nanstd(f) > 1e3 and abs(np.nanmedian(f)) < 1e3:
        f = 1.0 + f/1e6
    med = np.nanmedian(f) if np.isfinite(np.nanmedian(f)) else 1.0
    if med == 0: med = 1.0
    return f / med

def fetch_lc(target, sector=None, mission="TESS", author_priority=("SPOC","QLP","TESS-SPOC")):
    sr = search_lightcurve(target, mission=mission, sector=sector)
    if len(sr) == 0:
        raise RuntimeError(f"Nenhuma light curve encontrada (mission={mission}, sector={sector}).")
    # prioriza autores
    order = []
    for a in author_priority:
        order += [r for r in sr if getattr(r, "author", None) == a]
    order += [r for r in sr if r not in order]
    lc = order[0].download().remove_nans()
    try:
        lc = lc.flatten(window_length=101)
    except Exception:
        pass
    lc = lc.remove_outliers(sigma=5)
    t = lc.time.value
    f = robust_normalize_to_one(lc.flux.value)
    return t, f

def run_tls_or_bls(t, f, pmin, pmax, tls_depth_min=5e-6, use_tls=False):
    if use_tls and HAS_TLS:
        try:
            model = TLS(np.array(t, float), np.array(f, float))
            res = model.power(period_min=pmin, period_max=pmax, transit_depth_min=tls_depth_min)
            transit_count = int(getattr(res, "transit_count", 0) or 0)
            SDE = float(getattr(res, "SDE", np.nan))
            SNR = float(getattr(res, "SNR", 0.0)) if np.isfinite(getattr(res, "SNR", np.nan)) else 0.0
            if transit_count == 0 or not np.isfinite(SDE):
                raise RuntimeError("TLS não encontrou trânsitos confiáveis")
            return {
                "engine":"TLS","period":float(res.period),"t0":float(res.T0),
                "duration":float(res.duration),"depth":float(res.depth),
                "SDE":SDE,"SNR":SNR,"num_transits":transit_count
            }
        except Exception:
            pass  # cai para BLS
    # BLS
    y = f/np.nanmedian(f)
    bls = BoxLeastSquares(t, y)
    periods = np.linspace(pmin, pmax, 20000)
    power = bls.power(periods, 0.02)
    i = int(np.nanargmax(power.power))
    P = float(power.period[i]); t0 = float(power.transit_time[i])
    dur = float(power.duration[i]); depth = float(power.depth[i])
    sde = float((power.power[i]-np.nanmean(power.power))/(np.nanstd(power.power)+1e-9))
    scatter = np.nanstd(y)
    base = max(t) - min(t)
    ntr = max(1, int(base/P)) if np.isfinite(base) and P>0 else 1
    snr = float(abs(depth)/(scatter+1e-9)*math.sqrt(ntr))
    return {"engine":"BLS","period":P,"t0":t0,"duration":dur,"depth":depth,"SDE":sde,"SNR":snr,"num_transits":ntr}

def fold_features(t, f, P, t0, dur):
    phi = ((t - t0 + 0.5*P) % P)/P - 0.5
    order = np.argsort(phi); phi = phi[order]; y = f[order]
    w = 2.0*(dur/P)
    mask_local = (phi >= -w) & (phi <= +w)
    yin, yout = y[mask_local], y[~mask_local]
    depth_est = float(np.nanmedian(yin) - np.nanmedian(yout))
    rms_out = float(np.nanstd(yout))
    snr_local = float(abs(depth_est)/(rms_out+1e-9)*math.sqrt(max(1, yin.size)))
    mask_odd = (phi < 0)
    m_odd = float(np.nanmedian(y[mask_local & mask_odd]))
    m_even= float(np.nanmedian(y[mask_local & ~mask_odd]))
    odd_even = float(abs(m_odd - m_even) / (abs(depth_est)+1e-9))
    vshape = float(np.nanstd(yin) / (abs(depth_est)+1e-9))
    return {"phi":phi, "y":y, "odd_even":odd_even, "vshape":vshape, "snr_local":snr_local}

def planet_score(SDE, SNR, feats, num_transits):
    def sig(x): return 1/(1+math.exp(-x))
    s_sde = sig(((SDE or 0.0)-6.0)/2.0)
    s_snr = sig(((SNR or 0.0)-7.0)/3.0)
    pen_odd = math.exp(-min(5.0, max(0.0, feats["odd_even"])))
    pen_vsh = math.exp(-min(5.0, max(0.0, feats["vshape"])))
    bonus   = 1.0 if (num_transits or 1) >= 2 else 0.7
    score = (0.45*s_sde + 0.45*s_snr) * pen_odd * pen_vsh * bonus
    return max(0.0, min(1.0, score))

def phase_plot(phi, y, P, out_png, title=None, bins=200):
    # scatter
    plt.figure()
    plt.plot(phi, y, ".", markersize=2)
    # binned median overlay
    try:
        edges = np.linspace(-0.5, 0.5, bins+1)
        idx = np.digitize(phi, edges) - 1
        yb = np.zeros(bins); xb = 0.5*(edges[:-1]+edges[1:])
        for k in range(bins):
            sel = (idx==k)
            yb[k] = np.nanmedian(y[sel]) if np.any(sel) else np.nan
        plt.plot(xb, yb, linewidth=1.0)
    except Exception:
        pass
    plt.xlabel("Phase"); plt.ylabel("Flux (rel)")
    if title is None: title = f"Phase-folded (P={P:.5f} d)"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def time_plot(t, f, out_png, title):
    plt.figure()
    plt.plot(t, f, ".", markersize=2)
    plt.xlabel("Time (d)"); plt.ylabel("Flux (rel)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="Arquivo txt com um alvo por linha")
    ap.add_argument("--sector", type=int, default=None, help="Setor TESS (opcional)")
    ap.add_argument("--out", default="batch_scored.csv", help="CSV de saída")
    ap.add_argument("--min_period", type=float, default=0.5)
    ap.add_argument("--max_period", type=float, default=30.0)
    ap.add_argument("--use_tls", type=lambda s:s.lower()=='true', default=False, help="Tentar TLS antes do BLS")
    ap.add_argument("--plot_top_n", type=int, default=0, help="Gera plots para os top-N por planet_score")
    ap.add_argument("--plots_dir", default="plots_topN", help="Pasta para salvar os PNGs")
    ap.add_argument("--bins", type=int, default=200, help="Bins para a curva em fase")
    args = ap.parse_args()

    targets = [ln.strip() for ln in Path(args.targets).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not targets:
        print("Arquivo de alvos vazio.")
        sys.exit(1)

    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["target","sector","engine","period_d","t0_d","duration_d","depth_rel","SDE","SNR","num_transits",
                    "odd_even","vshape","snr_local","planet_score","status","error"])
        for tgt in targets:
            try:
                t, f = fetch_lc(tgt, sector=args.sector)
                det = run_tls_or_bls(t, f, args.min_period, args.max_period, use_tls=args.use_tls)
                feats_all = fold_features(t, f, det["period"], det["t0"], det["duration"])
                feats = {"odd_even":feats_all["odd_even"], "vshape":feats_all["vshape"], "snr_local":feats_all["snr_local"]}
                score = planet_score(det.get("SDE",0.0), det.get("SNR",0.0), feats, det.get("num_transits",1))
                w.writerow([tgt, args.sector, det["engine"], det["period"], det["t0"], det["duration"], det["depth"],
                            det.get("SDE",None), det.get("SNR",None), det.get("num_transits",None),
                            feats["odd_even"], feats["vshape"], feats["snr_local"], score, "OK",""])
            except Exception as e:
                w.writerow([tgt, args.sector, "", "", "", "", "", "", "", "", "", "", "", "", "FAIL", str(e)])
                sys.stderr.write(f"[FAIL] {tgt}: {e}\n")
    print(f"Concluído. CSV salvo em: {out_path}")

    # Plots dos top-N
    if args.plot_top_n > 0:
        rows = []
        with out_path.open("r", encoding="utf-8") as fr:
            rdr = csv.DictReader(fr)
            for r in rdr:
                if r["status"] == "OK" and r["planet_score"] not in ("", None):
                    try:
                        rows.append((r["target"], float(r["planet_score"]), float(r["period_d"]), float(r["t0_d"])))
                    except Exception:
                        continue
        rows.sort(key=lambda x: x[1], reverse=True)
        sel = rows[:args.plot_top_n]
        pdir = Path(args.plots_dir); pdir.mkdir(parents=True, exist_ok=True)
        print(f"Gerando plots para top-{len(sel)} em {pdir} ...")
        for tgt, score, P, t0 in sel:
            try:
                t, f = fetch_lc(tgt, sector=args.sector)
                title_time = f"{tgt} — LC (score={score:.3f})"
                time_plot(t, f, pdir / f"{tgt.replace(' ','_')}_time.png", title_time)
                # fold
                phi = ((t - t0 + 0.5*P) % P)/P - 0.5
                order = np.argsort(phi); phi = phi[order]; y = f[order]
                title_phase = f"{tgt} — Phase (P={P:.5f} d, score={score:.3f})"
                phase_plot(phi, y, P, pdir / f"{tgt.replace(' ','_')}_phase.png", title_phase, bins=args.bins)
            except Exception as e:
                sys.stderr.write(f"[WARN] plot fail {tgt}: {e}\n")
        print("Plots prontos.")
    
if __name__ == "__main__":
    main()
