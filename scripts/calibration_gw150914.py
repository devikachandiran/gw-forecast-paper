import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_fd_waveform
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.filter import sigma

# GW150914-like source parameters
m1 = 36.0        # solar masses, primary
m2 = 29.0        # solar masses, secondary
distance = 410.0 # Mpc, GW150914 luminosity distance
f_lower = 20.0   # Hz, aLIGO low-frequency cutoff
delta_f = 1.0 / 32
flen = int(2048 / delta_f) + 1

# Generate the waveform (IMRPhenomD)
hp, hc = get_fd_waveform(approximant="IMRPhenomD",
                          mass1=m1, mass2=m2,
                          distance=distance,
                          delta_f=delta_f,
                          f_lower=f_lower)

hp.resize(flen)

psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)

snr_opt = sigma(hp, psd=psd, low_frequency_cutoff=f_lower)
print(f"GW150914-like optimal SNR in aLIGO: {snr_opt:.2f}")

freqs = hp.sample_frequencies.numpy()
plt.figure(figsize=(8, 5))
plt.loglog(freqs, np.abs(hp.numpy()), label="|hp(f)|")
plt.loglog(freqs, np.sqrt(psd.numpy()), label="sqrt(PSD aLIGO)")
plt.xlim(10, 2048)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Strain (1/sqrt(Hz))")
plt.legend()
plt.title("GW150914-like waveform vs aLIGO noise")
plt.tight_layout()
plt.savefig("results/gw150914_calibration.png", dpi=160)
print("Plot saved to results/gw150914_calibration.png")
