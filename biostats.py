import numpy as np
from biosppy.signals import ecg


def hrv_metrics(rpeaks):
    """
    Compute heart rate variability (HRV) metrics from ECG data.
    Args:
        - rpeaks: list of R-peak indices
    Returns:
        - dict with HRV metrics (mean RR interval, SDNN, NN50, pNN50, LF power, HF power, LF/HF ratio, heart rate)9
    """
    sampling_rate = 200  # Hz
    rr_intervals = np.diff(rpeaks) * (1000 / sampling_rate)
    # Time-domain metrics
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    nn50 = np.sum(
        np.abs(np.diff(rr_intervals)) > 50
    )  # Number of pairs of successive RR intervals > 50ms
    pnn50 = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0.0

    nn20 = np.sum(
        np.abs(np.diff(rr_intervals)) > 20
    )  # Number of pairs of successive RR intervals > 20ms
    nn30 = np.sum(
        np.abs(np.diff(rr_intervals)) > 30
    )  # Number of pairs of successive RR intervals > 30ms

    pnn20 = (nn20 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0.0
    pnn30 = (nn30 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0.0

    # heart rate
    heart_rate = 60000 / mean_rr

    result = {
        "metrics": {
            "pnn50": pnn50,
            "pnn30": pnn30,
            "pnn20": pnn20,
            "mean_rr": mean_rr,
            "hr": heart_rate,
        },
        "start_interval": None,
        "end_interval": None,
    }
    return result


def short_instance_stats(buffer_ecg, tsec=60):
    """
    Compute HRV metrics from last tsec seconds of ECG data.
    Args:
        - buffer_ecg: list of ECG data points
        - tsec: time window in seconds
    Returns:
        - dict with HRV metrics
    """
    # TODO this can throw an error if the buffer_ecg is too small
    ecg_signal = np.array([row[3] for row in buffer_ecg])
    output = ecg.ecg(signal=ecg_signal, sampling_rate=200, show=False)
    return hrv_metrics(output[2])


def long_instance_stats(buffer_ecg, tsec=60):
    """
    Compute HRV metrics from last tsec seconds of ECG data.
    Args:
        - buffer_ecg: list of ECG data points
        - tsec: time window in seconds
    Returns:
        - dict with HRV metrics
    """
    # TODO edit to long form
    ecg_signal = np.array([row[3] for row in buffer_ecg])
    output = ecg.ecg(signal=ecg_signal, sampling_rate=200, show=False)
    return hrv_metrics(output[2])
