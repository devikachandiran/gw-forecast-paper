import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_fd_waveform
from pycbc.psd import aLIGOZeroDetHighPower, EinsteinTelescopeP1600143, CosmicExplorerP1600143

# ── Setup ──────────────────────────────────────────────────────
f_lower = 10.0
delta_f = 1.0 / 32
flen    = int(2048 / delta_f) + 1

psd_aligo = aLIGOZeroDetHighPower(flen, delta_f, f_lower)
psd_et    = EinsteinTelescopeP1600143(flen, delta_f, f_lower)
psd_ce    = CosmicExplorerP1600143(flen, delta_f, f_lower)

def inner_product(h1, h2, psd, f_lower, delta_f):
    """Noise-weighted inner product <h1|h2>."""
    f_start = int(f_lower / delta_f)
    h1_arr  = h1.numpy()[f_start:]
    h2_arr  = h2.numpy()[f_start:]
    psd_arr = psd.numpy()[f_start:]
    mask    = psd_arr > 0
    integrand = np.real(h1_arr[mask] * np.conj(h2_arr[mask])) / psd_arr[mask]
    return 4.0 * np.real(np.sum(integrand)) * delta_f

def get_waveform(mc, q, dl):
    """Generate waveform from chirp mass, mass ratio, distance."""
    eta = q / (1 + q)**2
    M   = mc / eta**(3/5)
    m1  = M / (1 + q)
    m2  = q * m1
    try:
        hp, hc = get_fd_waveform(approximant="IMRPhenomD",
                                  mass1=m1, mass2=m2,
                                  distance=dl,
                                  delta_f=delta_f,
                                  f_lower=f_lower)
        hp.resize(flen)
        return hp
    except:
        return None

def numerical_derivative(mc, q, dl, param, psd, eps_frac=0.01):
    """Numerical derivative dh/d(param) using central differences."""
    if param == "mc":
        eps = eps_frac * mc
        h_plus  = get_waveform(mc + eps, q, dl)
        h_minus = get_waveform(mc - eps, q, dl)
        if h_plus is None or h_minus is None:
            return None
        deriv = (h_plus.numpy() - h_minus.numpy()) / (2 * eps)
    elif param == "q":
        eps = eps_frac * q
        h_plus  = get_waveform(mc, q + eps, dl)
        h_minus = get_waveform(mc, q - eps, dl)
        if h_plus is None or h_minus is None:
            return None
        deriv = (h_plus.numpy() - h_minus.numpy()) / (2 * eps)
    elif param == "dl":
        eps = eps_frac * dl
        h_plus  = get_waveform(mc, q, dl + eps)
        h_minus = get_waveform(mc, q, dl - eps)
        if h_plus is None or h_minus is None:
            return None
        deriv = (h_plus.numpy() - h_minus.numpy()) / (2 * eps)
    from pycbc.types import FrequencySeries
    return FrequencySeries(deriv, delta_f=delta_f)

def fisher_errors(mc, q, dl, psd):
    """Compute parameter uncertainties from Fisher matrix."""
    params = ["mc", "q", "dl"]
    derivs = {}
    for p in params:
        d = numerical_derivative(mc, q, dl, p, psd)
        if d is None:
            return None
        derivs[p] = d

    Gamma = np.zeros((3, 3))
    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
            Gamma[i, j] = inner_product(derivs[pi], derivs[pj], psd, f_lower, delta_f)

    try:
        Cov = np.linalg.inv(Gamma)
        sigmas = np.sqrt(np.diag(Cov))
        return sigmas  # [sigma_mc, sigma_q, sigma_dl]
    except:
        return None

# ── Representative sources ─────────────────────────────────────
# Chirp masses from 3 to 25 solar masses, fixed distance 500 Mpc
chirp_masses = np.linspace(3, 25, 12)
q_fixed      = 0.8
dl_fixed     = 500.0  # Mpc

sig_mc_aligo, sig_mc_et, sig_mc_ce = [], [], []
sig_q_aligo,  sig_q_et,  sig_q_ce  = [], [], []
sig_dl_aligo, sig_dl_et, sig_dl_ce = [], [], []

print("Computing Fisher matrix errors...")
for i, mc in enumerate(chirp_masses):
    print(f"  Chirp mass {mc:.1f} Msun ({i+1}/{len(chirp_masses)})...")

    for psd, sa, sq, sd in [
        (psd_aligo, sig_mc_aligo, sig_q_aligo, sig_dl_aligo),
        (psd_et,    sig_mc_et,    sig_q_et,    sig_dl_et),
        (psd_ce,    sig_mc_ce,    sig_q_ce,    sig_dl_ce),
    ]:
        result = fisher_errors(mc, q_fixed, dl_fixed, psd)
        if result is not None:
            sa.append(result[0])
            sq.append(result[1])
            sd.append(result[2])
        else:
            sa.append(np.nan)
            sq.append(np.nan)
            sd.append(np.nan)

sig_mc_aligo = np.array(sig_mc_aligo)
sig_mc_et    = np.array(sig_mc_et)
sig_mc_ce    = np.array(sig_mc_ce)
sig_q_aligo  = np.array(sig_q_aligo)
sig_q_et     = np.array(sig_q_et)
sig_q_ce     = np.array(sig_q_ce)
sig_dl_aligo = np.array(sig_dl_aligo)
sig_dl_et    = np.array(sig_dl_et)
sig_dl_ce    = np.array(sig_dl_ce)

print("\nSample results at Mc=10 Msun:")
idx = 4
print(f"  aLIGO: sigma_Mc={sig_mc_aligo[idx]:.3f} Msun, sigma_q={sig_q_aligo[idx]:.3f}, sigma_dL={sig_dl_aligo[idx]:.1f} Mpc")
print(f"  ET:    sigma_Mc={sig_mc_et[idx]:.3f} Msun, sigma_q={sig_q_et[idx]:.3f}, sigma_dL={sig_dl_et[idx]:.1f} Mpc")
print(f"  CE:    sigma_Mc={sig_mc_ce[idx]:.3f} Msun, sigma_q={sig_q_ce[idx]:.3f}, sigma_dL={sig_dl_ce[idx]:.1f} Mpc")

# ── Plot ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].semilogy(chirp_masses, sig_mc_aligo, color="steelblue",  linewidth=2, marker="o", ms=5, label="aLIGO O5")
axes[0].semilogy(chirp_masses, sig_mc_et,    color="darkorange", linewidth=2, marker="s", ms=5, label="ET")
axes[0].semilogy(chirp_masses, sig_mc_ce,    color="green",      linewidth=2, marker="^", ms=5, label="CE")
axes[0].set_xlabel("Chirp Mass (M☉)", fontsize=12)
axes[0].set_ylabel("σ(Mc) [M☉]", fontsize=12)
axes[0].set_title("Chirp Mass Uncertainty", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(chirp_masses, sig_q_aligo, color="steelblue",  linewidth=2, marker="o", ms=5, label="aLIGO O5")
axes[1].semilogy(chirp_masses, sig_q_et,    color="darkorange", linewidth=2, marker="s", ms=5, label="ET")
axes[1].semilogy(chirp_masses, sig_q_ce,    color="green",      linewidth=2, marker="^", ms=5, label="CE")
axes[1].set_xlabel("Chirp Mass (M☉)", fontsize=12)
axes[1].set_ylabel("σ(q)", fontsize=12)
axes[1].set_title("Mass Ratio Uncertainty", fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

axes[2].semilogy(chirp_masses, sig_dl_aligo, color="steelblue",  linewidth=2, marker="o", ms=5, label="aLIGO O5")
axes[2].semilogy(chirp_masses, sig_dl_et,    color="darkorange", linewidth=2, marker="s", ms=5, label="ET")
axes[2].semilogy(chirp_masses, sig_dl_ce,    color="green",      linewidth=2, marker="^", ms=5, label="CE")
axes[2].set_xlabel("Chirp Mass (M☉)", fontsize=12)
axes[2].set_ylabel("σ(dL) [Mpc]", fontsize=12)
axes[2].set_title("Luminosity Distance Uncertainty", fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.suptitle("Fisher Matrix Parameter Uncertainties\naLIGO O5 · Einstein Telescope · Cosmic Explorer\n(q=0.8, dL=500 Mpc)", fontsize=11)
plt.tight_layout()
plt.savefig("results/fisher_uncertainties.png", dpi=160)
print("Plot saved to results/fisher_uncertainties.png")