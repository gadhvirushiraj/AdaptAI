import numpy as np
import pandas as pd
from biosppy.signals import ecg


def hrv_metrics(rpeaks):
    """
    Compute heart rate variability (HRV) metrics from ECG data.

    Args:
        rpeaks: list of R-peak indices

    Returns:
        dict with HRV metrics (mean RR interval, SDNN, NN50, pNN50, LF power, HF power, LF/HF ratio, heart rate)9
    """
    sampling_rate = 125  # Hz
    rr_intervals = np.diff(rpeaks) * (1000 / sampling_rate)

    # Time-domain metrics
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    nn50 = np.sum(
        np.abs(np.diff(rr_intervals)) > 50
    )  # Number of pairs of successive RR intervals > 50ms
    pnn50 = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0.0

    # Frequency-domain metrics using FFT
    fft_freqs = np.fft.rfftfreq(len(rr_intervals), d=mean_rr / 1000)
    fft_power = np.abs(np.fft.rfft(rr_intervals)) ** 2

    # Frequency bands (in Hz)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    # Compute power in LF and HF bands
    lf_power = np.sum(fft_power[(fft_freqs >= lf_band[0]) & (fft_freqs < lf_band[1])])
    hf_power = np.sum(fft_power[(fft_freqs >= hf_band[0]) & (fft_freqs < hf_band[1])])

    # LF/HF ratio
    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else np.nan

    # heart rate
    heart_rate = 60000 / mean_rr

    return {
        "mean_rr": mean_rr,
        "sdnn": sdnn,
        "nn50": nn50,
        "pnn50": pnn50,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
        "heart_rate": heart_rate,
    }


def short_instance_stats(buffer_ecg, tsec=60):
    """
    Compute HRV metrics from last tsec seconds of ECG data.

    Args:
        buffer_ecg: list of ECG data points
        tsec: time window in seconds

    Returns:
        dict with HRV metrics
    """

    # TODO this can throw an error if the buffer_ecg is too small
    last_tsec = buffer_ecg[-tsec * 125 :]  # Last tsec seconds of ECG data
    output = ecg.ecg(signal=last_tsec, sampling_rate=125, show=False)

    return hrv_metrics(output[2])


def long_instance_stats(ecg_input, nlast_tsec=3600):
    """
    Compute the mean of each HRV metric from the last 15 stored HRV metrics.

    Args:
        stored_stats: list of HRV metrics dictionaries

    Returns:
        dict with mean HRV metrics
    """

    last_tsec = ecg_input[-nlast_tsec * 125 :]  # Last tsec seconds of ECG data
    output = ecg.ecg(signal=last_tsec, sampling_rate=125, show=False)

    hrv_out = hrv_metrics(output[2])
    stats_out = {"pNN50":hrv_out['pnn50'],"hr_interval":f"{min(output[6])} - {max(output[6])}","heart_rate":hrv_out['heart_rate']}

    return stats_out
