## What the script does: 
It bulk-scans a list of targets (TESS TIC/TOI or star names), downloads each target’s light curve from MAST via Lightkurve, applies basic cleaning/detrending, searches for periodic box-shaped dips (candidate transits) using BLS (or TLS if you opt in), computes a handful of diagnostics, assigns a rule-based score (not learned), writes everything to a CSV, and (optionally) generates PNG plots for the top-N candidates so you can review them by eye.


## How to install & run: 
```
pip install lightkurve astropy numpy scipy matplotlib
pip install transitleastsquares
```

## Inputs:

- A text file, one target per line (e.g. targets.txt)
```
TOI 700
TIC 150428135
Kepler-10
```

## Typical command:
Lightweight (BLS) + plots for top 12 scores
```
python batch_transit_scan_v2.py \
  --targets targets.txt \
  --sector 13 \
  --out scored.csv \
  --plot_top_n 12 \
  --plots_dir plots_topN
```

## Outputs:
**cored.csv** with one row per target containing period, depth, diagnostics, and the rule-based score.

**plots_topN/ folder with**:

<ID>_time.png — time-domain light curve

<ID>_phase.png — phase-folded light curve (with a binned-median overlay)

## Processing pipeline (step-by-step)

#### 1) Light curve retrieval and pre-processing

Source: The script uses lightkurve.search_lightcurve(...) and prefers SPOC/QLP products if available. You can constrain to a sector to reduce downloads.

- Cleaning:

Remove NaNs.

Flatten with a windowed filter (removes slow trends so periodic dips stand out).

Remove moderate outliers (sigma-clipping).

Robust normalization:

Converts flux to relative units with median ≈ 1 (even if the original is ppm or counts).

This makes all downstream calculations consistent and prevents “scale” issues.

Why: Detrending + normalization ensure dips are measured against a flat baseline, so BLS/TLS can focus on periodic box-like decrements rather than long-term variability.

### 2) Transit search (BLS by default, TLS optional)

 - BLS (Box Least Squares) — default:

Builds a grid of trial periods between --min_period and --max_period.

Assumes a box-shaped transit model with a fractional duration (fixed at ~2% of the period in the script).

Evaluates a power for each trial period; picks the peak as the best candidate (period P, epoch t0, duration D, and depth).

Computes:

SDE (signal detection efficiency, here an approximate z-score of the peak power).

SNR (approximate): |depth| / rms(out-of-transit) * sqrt(number_of_transits).

number_of_transits ≈ time baseline / period.

TLS (Transit Least Squares) — optional (--use_tls true):

Similar scan over periods, but uses a physically motivated template (rounded, limb-darkened transit), which can be more sensitive to shallow events.

If TLS fails to produce a reliable result (e.g., no transits fitted), the script falls back to BLS automatically.

When to use which: BLS is faster and robust for quick scans. TLS is heavier but can help on low-depth or short-duration events. If you’re compute-limited, stick to BLS.

### 3) Phase folding & diagnostic measurements (no ML)

Once the period is chosen:

Phase-fold the light curve at P with reference epoch t0.

Define a local window around the transit (± 2 transit durations) and split points into in-transit vs out-of-transit.

Compute simple diagnostics:

depth_est: median(in-transit) − median(out-of-transit)

rms_out: scatter out of transit

snr_local: |depth_est| / rms_out * sqrt(N_in) (SNR restricted to the local window)

odd_even_ratio: compares the median depth in alternating halves of phase (helps flag eclipsing binaries where odd/even events differ)

vshape_proxy: std(in-transit) / |depth_est| (higher values can indicate V-shaped events / grazing eclipses rather than flat-bottomed transits)

These are rule-based quality checks—no training involved.

### 4) Rule-based score (“planet_score”)
The script does not use a learned model. It combines the above diagnostics into a single heuristic score in [0,1]:

Compress SDE and SNR with smooth functions (so they don’t explode with scale),

Penalize large odd_even_ratio (likely eclipsing binaries),

Penalize large vshape_proxy (grazing/V-shaped),

Give a small bonus if there are ≥2 observed transits.

Think of it as a triage number to help you sort by “how transit-like” an object looks, not a classifier.

### 5) Plotting (top-N)
For the top-N by score (e.g., --plot_top_n 12):

Time plot: the normalized light curve vs time.

Phase plot: the folded light curve with a binned-median overlay to visualize the transit shape and depth quickly.

Plots are saved under --plots_dir.

### CLI options you’ll actually use:
--targets PATH — required; text file with one ID per line.

--sector INT — optional; speeds up downloads by limiting to one TESS sector.

--out PATH — CSV path (default batch_scored.csv).

--min_period, --max_period — period search bounds (days); tighten these if you know your range to save time.

--use_tls true|false — try TLS before BLS (heavier).

--plot_top_n INT — how many top scored targets to plot.

--plots_dir PATH — output directory for PNGs.

--bins INT — number of bins for the binned-median overlay on the phase plot.