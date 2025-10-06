
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lite_transit_checker_v2.py — checker leve e robusto (sem treinamento)

- Baixa curva (Lightkurve)
- Normaliza para média≈1 (sempre), de-trend básico
- Tenta TLS com parâmetros ajustáveis; se falhar ou não encontrar trânsitos, cai para BLS
- Extrai métricas + planet_score (0–1)
- Plota e salva JSON opcionalmente

Instalação:
  pip install lightkurve astropy numpy scipy matplotlib
  # Opcional (melhor): pip install transitleastsquares
"""

import argparse, json, math, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares

# TLS (opcional)
try:
    from transitleastsquares import transitleastsquares as TLS
    HAS_TLS = True
except Exception:
    HAS_TLS = False

from lightkurve import search_lightcurve

def sigmoid(x): return 1/(1+math.exp(-x))

def robust_normalize_to_one(flux):
    """Retorna fluxo com mediana=1.0, mesmo que já esteja em ppm/percent/etc."""
    f = np.array(flux, dtype=float)
    # Se parece ppm (amplitude >> 1e3 e média ~0), recentra em 1
    if np.nanstd(f) > 1e3 and abs(np.nanmedian(f)) < 1e3:
        f = 1.0 + f/1e6  # ppm → relativo (aprox.)
    # Se média for muito grande, normaliza por mediana
    med = np.nanmedian(f) if np.isfinite(np.nanmedian(f)) else 1.0
    med = med if med != 0 else 1.0
    f = f / med
    return f

def fetch_lightcurve(target: str, sector=None, mission="TESS", author_priority=("SPOC","QLP","TESS-SPOC")):
    sr = search_lightcurve(target, mission=mission, sector=sector)
    if len(sr) == 0:
        raise RuntimeError(f"Nenhuma light curve encontrada para '{target}' (mission={mission}, sector={sector}).")
    # prioriza autores
    order = []
    for a in author_priority:
        order += [r for r in sr if getattr(r, "author", None) == a]
    order += [r for r in sr if r not in order]
    lc = order[0].download()
    # limpeza + detrending leve + normalização robusta
    try:
        lc = lc.remove_nans()
        try:
            lc = lc.flatten(window_length=101)  # remove tendências
        except Exception:
            pass
        lc = lc.remove_outliers(sigma=5)
        f = robust_normalize_to_one(lc.flux.value)
        t = lc.time.value
        return t, f, lc
    except Exception as e:
        raise RuntimeError(f"Falha ao normalizar LC: {e}")

def run_tls_or_bls(t, f, period_min, period_max, tls_depth_min=5e-6, force_bls=False):
    if HAS_TLS and not force_bls:
        try:
            model = TLS(np.array(t, dtype=float), np.array(f, dtype=float))
            res = model.power(period_min=period_min, period_max=period_max, transit_depth_min=tls_depth_min)
            # TLS às vezes não encontra ajuste — tratar como falha e cair p/ BLS
            transit_count = int(getattr(res, "transit_count", 0) or 0)
            SDE = float(getattr(res, "SDE", np.nan))
            SNR = float(getattr(res, "SNR", np.nan)) if hasattr(res, "SNR") and np.isfinite(getattr(res, "SNR", np.nan)) else 0.0
            if transit_count == 0 or not np.isfinite(SDE):
                raise RuntimeError("TLS não encontrou trânsitos confiáveis (transit_count=0 ou SDE NaN).")
            return {
                "engine": "TLS",
                "period": float(res.period),
                "t0": float(res.T0),
                "duration_d": float(res.duration),
                "depth": float(res.depth),
                "sde": SDE,
                "snr": SNR,
                "power_series": (res.periods.tolist(), res.power.tolist()),
                "num_transits": transit_count,
                "in_transit": getattr(res, "intransit", None).tolist() if hasattr(res, "intransit") else None
            }
        except Exception as e:
            print("[INFO] TLS falhou ou não encontrou trânsito:", e, "→ usando BLS.")
            # continua para BLS

    # BLS robusto
    y = f/np.nanmedian(f)
    bls = BoxLeastSquares(t, y)
    periods = np.linspace(period_min, period_max, 20000)
    power = bls.power(periods, 0.02)  # duração ~2% P (heurística)
    i = int(np.nanargmax(power.power))
    P = float(power.period[i]); t0 = float(power.transit_time[i])
    dur = float(power.duration[i]); depth = float(power.depth[i])
    sde = float((power.power[i]-np.nanmean(power.power))/(np.nanstd(power.power)+1e-9))
    scatter = np.nanstd(y)
    base = max(t) - min(t)
    ntr = max(1, int(base/P)) if np.isfinite(base) and P>0 else 1
    snr = float(abs(depth)/(scatter+1e-9)*math.sqrt(ntr))
    return {
        "engine":"BLS","period":P,"t0":t0,"duration_d":dur,"depth":depth,
        "sde":sde,"snr":snr,"power_series":(power.period.tolist(), power.power.tolist()),
        "num_transits":ntr,"in_transit":None
    }

def fold_and_features(t, f, period, t0, dur_d):
    P = period
    phi = ((t - t0 + 0.5*P) % P)/P - 0.5
    order = np.argsort(phi); phi = phi[order]; y = f[order]
    w = 2.0*(dur_d/P)
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
    feats = {"depth_est":depth_est,"rms_out":rms_out,"snr_local":snr_local,
             "odd_even_ratio":odd_even,"vshape_proxy":vshape,"local_window_frac":float(2*w)}
    return feats, (phi, y), mask_local

def quick_planet_score(sde, snr, feats, num_transits):
    if not np.isfinite(sde): sde = 0.0
    if not np.isfinite(snr): snr = 0.0
    s_sde = sigmoid((sde-6.0)/2.0)
    s_snr = sigmoid((snr-7.0)/3.0)
    pen_odd_even = math.exp(-min(5.0, feats["odd_even_ratio"]))
    pen_vshape   = math.exp(-min(5.0, feats["vshape_proxy"]))
    bonus_multi  = 1.0 if num_transits>=2 else 0.7
    return max(0.0, min(1.0, (0.45*s_sde+0.45*s_snr)*pen_odd_even*pen_vshape*bonus_multi))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Nome/ID (ex.: TIC 150428135, TOI 700)")
    ap.add_argument("--sector", type=int, default=None, help="Setor TESS (opcional)")
    ap.add_argument("--min_period", type=float, default=0.5)
    ap.add_argument("--max_period", type=float, default=30.0)
    ap.add_argument("--tls_depth_min", type=float, default=5e-6, help="Mínimo de profundidade para TLS (ajuste se TLS não encontrar nada)")
    ap.add_argument("--force_bls", type=lambda s:s.lower()=='true', default=False)
    ap.add_argument("--plot", type=lambda s:s.lower()=='true', default=True)
    ap.add_argument("--save_json", type=lambda s:s.lower()=='true', default=True)
    args = ap.parse_args()

    # 1) Busca e normalização robusta
    t, f, lc = fetch_lightcurve(args.target, sector=args.sector)
    f = robust_normalize_to_one(f)
    print(f"[INFO] Fluxo normalizado: med={np.nanmedian(f):.4f}, std={np.nanstd(f):.4f}")

    # 2) TLS → BLS
    det = run_tls_or_bls(t, f, args.min_period, args.max_period, tls_depth_min=args.tls_depth_min, force_bls=args.force_bls)

    # 3) Vistas simples + features
    feats, (phi, y), mask_local = fold_and_features(t, f, det["period"], det["t0"], det["duration_d"])
    score = quick_planet_score(det.get("sde",0.0), det.get("snr",0.0), feats, det.get("num_transits",1))

    # 4) Print
    print("\n=== Resultado rápido (sem treino) ===")
    for k in ["engine","period","t0","duration_d","depth","sde","snr","num_transits"]:
        v = det.get(k, None)
        if v is None: continue
        if isinstance(v, float): print(f"{k:12s}: {v:.6f}")
        else: print(f"{k:12s}: {v}")
    for k,v in feats.items(): print(f"{k:12s}: {v:.6f}")
    print(f"planet_score: {score:.3f}\n")

    # 5) Plots/JSON
    if args.plot:
        plt.figure(); plt.plot(t, f, ".", markersize=2)
        plt.xlabel("Time (d)"); plt.ylabel("Flux (rel)")
        plt.title(f"{args.target} — Light curve (norm)")
        plt.tight_layout(); plt.savefig("lc_time.png", dpi=160); plt.close()

        plt.figure(); plt.plot(phi, y, ".", markersize=2)
        plt.xlabel("Phase"); plt.ylabel("Flux (rel)")
        plt.title(f"{args.target} — Folded @ P={det['period']:.5f} d")
        plt.tight_layout(); plt.savefig("lc_phase.png", dpi=160); plt.close()
        print("Figuras salvas: lc_time.png, lc_phase.png")

    if args.save_json:
        report = {
            "target": args.target, "sector": args.sector,
            "engine": det["engine"], "period_d": det["period"], "t0_d": det["t0"],
            "duration_d": det["duration_d"], "depth_rel": det["depth"],
            "SDE": det.get("sde", None), "SNR": det.get("snr", None),
            "num_transits": det.get("num_transits", None),
            "features": feats, "planet_score": score
        }
        with open("report.json","w",encoding="utf-8") as fjs:
            json.dump(report, fjs, indent=2, ensure_ascii=False)
        print("Relatório salvo: report.json")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print("Erro:", e); sys.exit(1)
