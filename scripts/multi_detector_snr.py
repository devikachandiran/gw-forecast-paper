import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_fd_waveform
from pycbc.psd import aLIGOZeroDetHighPower, EinsteinTelescopeP1600143, CosmicExplorerP1600143
from pycbc.filter import sigma

m1       = 36.0
m2       = 29.0
distance = 410.0
f_lower  = 10.0
delta_f  = 1.0 / 32
flen     = int(2048 / delta_f) + 1

hp, hc = get_fd_waveform(approximant="IMRPhenomD",
                          mass1=m1, mass2=m2,
                          distance=distance,
                          delta_f=delta_f,
                          f_lower=f_lower)
hp.resize(flen)

psd_aligo = aLIGOZeroDetHighPower(flen, delta_f, f_lower)
psd_et    = EinsteinTelescopeP1600143(flen, delta_f, f_lower)
psd_ce    = CosmicExplorerP1600143(flen, delta_f, f_lower)

snr_aligo = sigma(hp, psd=psd_aligo, low_frequency_cutoff=f_lower)
snr_et    = sigma(hp, psd=psd_et,    low_frequency_cutoff=f_lower)
snr_ce    = sigma(hp, psd=psd_ce,    low_frequency_cutoff=f_lower)

print(f"GW150914-like source at {distance} Mpc:")
print(f"  aLIGO O5  SNR: {snr_aligo:.2f}")
print(f"  ET        SNR: {snr_et:.2f}")
print(f"  CE        SNR: {snr_ce:.2f}")
print(f"\n  ET/aLIGO ratio:  {snr_et/snr_aligo:.1f}x")
print(f"  CE/aLIGO ratio:  {snr_ce/snr_aligo:.1f}x")

freqs = hp.sample_frequencies.numpy()
plt.figure(figsize=(10, 6))
plt.loglog(freqs, np.sqrt(psd_aligo.numpy()), label="aLIGO O5", color="steelblue")
plt.loglog(freqs, np.sqrt(psd_et.numpy()),    label="ET",        color="darkorange")
plt.loglog(freqs, np.sqrt(psd_ce.numpy()),    label="CE",        color="green")
plt.loglog(freqs, np.abs(hp.numpy()),         label="|hp(f)|",   color="red", linestyle="--", alpha=0.7)
plt.xlim(5, 2048)
plt.ylim(1e-25, 1e-20)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Strain (1/sqrt(Hz))")
plt.legend()
plt.title("Detector sensitivity curves vs GW150914-like waveform")
plt.tight_layout()
plt.savefig("results/multi_detector_psd.png", dpi=160)
print("Plot saved to results/multi_detector_psd.png")