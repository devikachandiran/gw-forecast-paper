import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_fd_waveform
from pycbc.psd import aLIGOZeroDetHighPower, EinsteinTelescopeP1600143, CosmicExplorerP1600143
from pycbc.filter import sigma

f_lower = 10.0
delta_f = 1.0 / 32
flen    = int(2048 / delta_f) + 1
snr_thr = 8.0
d_ref   = 1000.0

psd_aligo = aLIGOZeroDetHighPower(flen, delta_f, f_lower)
psd_et    = EinsteinTelescopeP1600143(flen, delta_f, f_lower)
psd_ce    = CosmicExplorerP1600143(flen, delta_f, f_lower)

total_masses = np.logspace(np.log10(5), np.log10(300), 50)

def compute_horizon(m1, m2, psd):
    try:
        hp, hc = get_fd_waveform(approximant="IMRPhenomD",
                                  mass1=m1, mass2=m2,
                                  distance=d_ref,
                                  delta_f=delta_f,
                                  f_lower=f_lower)
        hp.resize(flen)
        snr_ref = sigma(hp, psd=psd, low_frequency_cutoff=f_lower)
        return (d_ref * snr_ref / snr_thr) / 2.26
    except:
        return np.nan

print("Computing horizon distances...")
horizon_aligo = np.array([compute_horizon(M/2, M/2, psd_aligo) for M in total_masses])
horizon_et    = np.array([compute_horizon(M/2, M/2, psd_et)    for M in total_masses])
horizon_ce    = np.array([compute_horizon(M/2, M/2, psd_ce)    for M in total_masses])
print("Done.")

print(f"\nSky-averaged horizon distances:")
print(f"  aLIGO O5: {np.nanmax(horizon_aligo):.0f} Mpc  at M={total_masses[np.nanargmax(horizon_aligo)]:.0f} Msun")
print(f"  ET:       {np.nanmax(horizon_et):.0f} Mpc  at M={total_masses[np.nanargmax(horizon_et)]:.0f} Msun")
print(f"  CE:       {np.nanmax(horizon_ce):.0f} Mpc  at M={total_masses[np.nanargmax(horizon_ce)]:.0f} Msun")

fig, ax = plt.subplots(figsize=(10, 7))
ax.loglog(total_masses, horizon_aligo, color="steelblue",  linewidth=2.5, label="aLIGO O5")
ax.loglog(total_masses, horizon_et,    color="darkorange", linewidth=2.5, label="Einstein Telescope")
ax.loglog(total_masses, horizon_ce,    color="green",      linewidth=2.5, label="Cosmic Explorer")
ax.axhline(4300,  color="gray",   linestyle=":",  linewidth=1.5, label="z≈1 (4300 Mpc)")
ax.axhline(9700,  color="purple", linestyle="--", linewidth=1.5, label="z≈2 (9700 Mpc)")
ax.set_xlabel("Total Mass (M☉)", fontsize=13)
ax.set_ylabel("Sky-averaged Horizon Distance (Mpc)", fontsize=13)
ax.set_title("Horizon Distance vs Total Mass\naLIGO O5 · Einstein Telescope · Cosmic Explorer", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("results/horizon_distance.png", dpi=160)
print("Plot saved to results/horizon_distance.png")