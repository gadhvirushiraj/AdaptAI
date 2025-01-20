# from datetime import datetime, timedelta
# import pytz
# import numpy as np
# from scipy.signal import find_peaks

# def short_instance_stats(ecg_input, tsec=9):
#     """
#     Compute HRV metrics from the last tsec seconds (default 1 minute) of ECG data.

#     Args:
#         ecg_input: list of ECG data points
#         tsec: time window in seconds (default: 60 seconds)

#     Returns:
#         dict with HRV metrics, start and end intervals
#     """
#     # Current IST time
#     ist = pytz.timezone('Asia/Kolkata')
#     end_interval = datetime.now(ist)
#     start_interval = end_interval - timedelta(seconds=tsec)

#     # Ensure the buffer has enough data
#     # required_data_points = tsec * 125  # Assuming 125 Hz sampling rate
#     # if len(ecg_input) < required_data_points:
#     #     raise ValueError("ECG input buffer is too small for the requested time window.")

#     # # Select the last tsec seconds of data
#     # ecg_input = ecg_input[-required_data_points:]

#     # Extract the ECG signal and sensor timestamps
#     ecg_signal = np.array([row[3] for row in ecg_input])
#     timestamps_sensor = np.array([int(row[2]) for row in ecg_input])  # Use timestamp_sensor

#     print('time_sensor',timestamps_sensor)

#     # Detect R-peaks
#     peaks, _ = find_peaks(ecg_signal, height=0.4)  # Adjust 'height' as needed

#     print('peaks', peaks)

#     # Calculate RR intervals (differences between consecutive peaks in time)
#     rr_intervals = np.diff(timestamps_sensor[peaks])  # RR intervals in milliseconds

#     # Compute HRV metrics
#     pnn50 = np.sum(rr_intervals > 50) / len(rr_intervals) * 100
#     pnn30 = np.sum(rr_intervals > 30) / len(rr_intervals) * 100
#     pnn20 = np.sum(rr_intervals > 20) / len(rr_intervals) * 100

#     # Mean RR interval and Heart Rate (HR)
#     mean_rr = np.mean(rr_intervals)
#     hr = 60000 / mean_rr

#     # Include start and end intervals in the result
#     result = {
#         "metrics": {
#             "pnn50": pnn50,
#             "pnn30": pnn30,
#             "pnn20": pnn20,
#             "mean_rr": mean_rr,
#             "hr": hr
#         },
#         "start_interval": start_interval.strftime('%Y-%m-%d %H:%M:%S %Z'),
#         "end_interval": end_interval.strftime('%Y-%m-%d %H:%M:%S %Z')
#     }

#     return result


# def long_instance_stats(ecg_input):
#     """
#     Compute HRV metrics from the last 1 hour of ECG data.

#     Args:
#         ecg_input: list of ECG data points

#     Returns:
#         dict with HRV metrics, start and end intervals
#     """
#     # Current IST time
#     ist = pytz.timezone('Asia/Kolkata')
#     end_interval = datetime.now(ist)
#     start_interval = end_interval - timedelta(seconds=45)

#     # Ensure the buffer has enough data
#     # required_data_points = 60 * 60 * 125  # 1 hour of data at 125 Hz sampling rate
#     # if len(ecg_input) < required_data_points:
#     #     raise ValueError("ECG input buffer is too small for the requested time window.")

#     # # Select the last 1 hour of data
#     # ecg_input = ecg_input[-required_data_points:]

#     # Extract the ECG signal and sensor timestamps
#     ecg_signal = np.array([row[3] for row in ecg_input])
#     timestamps_sensor = np.array([int(row[2]) for row in ecg_input])  # Use timestamp_sensor

#     # Detect R-peaks
#     peaks, _ = find_peaks(ecg_signal, height=0.4)  # Adjust 'height' as needed

#     # Calculate RR intervals (differences between consecutive peaks in time)
#     rr_intervals = np.diff(timestamps_sensor[peaks])  # RR intervals in milliseconds

#     # Compute HRV metrics
#     pnn50 = np.sum(rr_intervals > 50) / len(rr_intervals) * 100
#     pnn30 = np.sum(rr_intervals > 30) / len(rr_intervals) * 100
#     pnn20 = np.sum(rr_intervals > 20) / len(rr_intervals) * 100

#     # Mean RR interval and Heart Rate (HR)
#     mean_rr = np.mean(rr_intervals)
#     hr = 60000 / mean_rr

#     # Include start and end intervals in the result
#     result = {
#         "metrics": {
#             "pnn50": pnn50,
#             "pnn30": pnn30,
#             "pnn20": pnn20,
#             "mean_rr": mean_rr,
#             "hr": hr
#         },
#         "start_interval": start_interval.strftime('%Y-%m-%d %H:%M:%S %Z'),
#         "end_interval": end_interval.strftime('%Y-%m-%d %H:%M:%S %Z')
#     }

#     return result



import numpy as np
import pandas as pd
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

    nn20 = np.sum(np.abs(np.diff(rr_intervals)) > 20)  # Number of pairs of successive RR intervals > 20ms
    nn30 = np.sum(np.abs(np.diff(rr_intervals)) > 30)  # Number of pairs of successive RR intervals > 30ms

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
            "hr": heart_rate
        },
        "start_interval": None,
        "end_interval": None
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
    # TODO this can throw an error if the buffer_ecg is too small
    ecg_signal = np.array([row[3] for row in buffer_ecg])
    output = ecg.ecg(signal=ecg_signal, sampling_rate=200, show=False)
    return hrv_metrics(output[2])