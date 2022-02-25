import matplotlib.image as mpimg
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pl
import cv2

image=cv2.imread(r"E:\NCR\TestImages\2.jpg")

def GetPwrSpcDensity(image):
    #https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
    npix = image.shape[0]

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Length=min(len(fourier_amplitudes),len(knrm))
    Abins, _, _ = stats.binned_statistic(knrm[0:Length], fourier_amplitudes[0:Length],
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return Abins
    pl.loglog(kvals, Abins)
    pl.xlabel("$k$")
    pl.ylabel("$P(k)$")
    pl.tight_layout()
    pl.savefig(r"E:\NCR\TestImages\cloud_power_spectrum2.png", dpi = 300, bbox_inches = "tight")

GetPwrSpcDensity(image)