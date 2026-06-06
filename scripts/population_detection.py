import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, c
from pycbc.waveform import get_fd_waveform
from pycbc.psd import aLIGOZeroDetHighPower, EinsteinTelescopeP1600143, CosmicExplorerP1600143
from pycbc.filter import sigma

# ── Reproduce population from your Aug 2025 project ────────────
YEAR   = 365.25 * 24 * 3600
MSUN   = 1.98847e30
RSUN   = 6.957e8
HUBBLE = 13.8e9

np.random.seed(42)  # reproducible

def sample_m1(size, mmin=5, mmax=40, alpha=2.35):
    r  = np.random.rand(size)
    a1 = 1 - alpha
    return ((r * (mmax**a1 - mmin**a1) + mmin**a1)) ** (1 / a1)

def merger_time_yr(m1_msun, m2_msun, a0_rsun):
    m1 = m1_msun * MSUN
    m2 = m2_msun * MSUN
    a0 = a0_rsun * RSUN
    num = 5 * c**5 * a0**4
    den = 256 * G**3 * m1 * m2 * (m1 + m2)
    return (num / den) / YEAR

def chirp_mass(m1, m2):
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

N  = 5000
m1 = sample_m1(N)
q  = np.random.uniform(0.3, 1.0, N)
m2 = q * m1
a0 = np.exp(np.random.uniform(np.log(1), np.log(100), N))

tmerge = merger_time_yr(m1, m2, a0)
mask   = tmerge < HUBBLE

m1c    = m1[mask]
m2c    = m2[mask]
chirp  = chirp_mass(m1c, m2c)

print(f"Population: {N} binaries, {mask.sum()} merge within Hubble time")

# ── Assign random luminosity distances (uniform in comoving volume) ─
# Simple approximation: d ~ uniform in volume means d ~ r^2 dr
# Draw distances uniform in volume out to d_max = 5000 Mpc
np.random.seed(99)
d_max = 5000.0  # Mpc
u     = np.random.uniform(0, 1, mask.sum())
distances = d_max * u**(1/3)   # uniform in volume

# ── SNR threshold ───────────────────────────────────────────────
SNR_THR = 8.0
f_lower = 10.0
delta_f = 1.0 / 32
flen    = int(2048 / delta_f) + 1

psd_aligo = aLIGOZeroDetHighPower(flen, delta_f, f_lower)
psd_et    = EinsteinTelescopeP1600143(flen, delta_f, f_lower)
psd_ce    = CosmicExplorerP1600143(flen, delta_f, f_lower)

def compute_snr(m1, m2, dist, psd):
    try:
        hp, hc = get_fd_waveform(approximant="IMRPhenomD",
                                  mass1=m1, mass2=m2,
                                  distance=dist,
                                  delta_f=delta_f,
                                  f_lower=f_lower)
        hp.resize(flen)
        return sigma(hp, psd=psd, low_frequency_cutoff=f_lower)
    except:
        return 0.0

# ── Compute SNR for each binary in each detector ────────────────
n = len(m1c)
snr_aligo = np.zeros(n)
snr_et    = np.zeros(n)
snr_ce    = np.zeros(n)

print(f"Computing SNR for {n} binaries...")
for i in range(n):
    snr_aligo[i] = compute_snr(m1c[i], m2c[i], distances[i], psd_aligo)
    snr_et[i]    = compute_snr(m1c[i], m2c[i], distances[i], psd_et)
    snr_ce[i]    = compute_snr(m1c[i], m2c[i], distances[i], psd_ce)
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{n} done...")

det_aligo = snr_aligo >= SNR_THR
det_et    = snr_et    >= SNR_THR
det_ce    = snr_ce    >= SNR_THR

print(f"\nDetection fractions (d_max={d_max} Mpc):")
print(f"  aLIGO O5: {det_aligo.mean():.3f}  ({det_aligo.sum()} / {n})")
print(f"  ET:       {det_et.mean():.3f}  ({det_et.sum()} / {n})")
print(f"  CE:       {det_ce.mean():.3f}  ({det_ce.sum()} / {n})")

# ── Plot detectable fraction vs chirp mass ──────────────────────
bins  = np.linspace(2, 30, 20)
cents = 0.5 * (bins[:-1] + bins[1:])

def det_fraction_per_bin(chirp, detected, bins):
    frac = []
    for i in range(len(bins)-1):
        in_bin = (chirp >= bins[i]) & (chirp < bins[i+1])
        if in_bin.sum() == 0:
            frac.append(np.nan)
        else:
            frac.append(detected[in_bin].mean())
    return np.array(frac)

frac_aligo = det_fraction_per_bin(chirp, det_aligo, bins)
frac_et    = det_fraction_per_bin(chirp, det_et,    bins)
frac_ce    = det_fraction_per_bin(chirp, det_ce,    bins)

plt.figure(figsize=(10, 6))
plt.plot(cents, frac_aligo, color="steelblue",  linewidth=2.5, label="aLIGO O5",  marker="o", ms=5)
plt.plot(cents, frac_et,    color="darkorange", linewidth=2.5, label="ET",         marker="s", ms=5)
plt.plot(cents, frac_ce,    color="green",      linewidth=2.5, label="CE",         marker="^", ms=5)
plt.xlabel("Chirp Mass (M☉)", fontsize=13)
plt.ylabel("Detectable Fraction", fontsize=13)
plt.title(f"Detectable Fraction vs Chirp Mass\n(population within {d_max} Mpc, SNR ≥ {SNR_THR})", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("results/detectable_fraction.png", dpi=160)
print("Plot saved to results/detectable_fraction.png")