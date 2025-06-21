import numpy as np
from enum import Enum

class ActivityState(Enum):
    """
    Defines the different states of activity for data collection and classification.
    """
    IDLE = 0
    WRITING = 1
    OTHER = 2

class CollectionMode(Enum):
    """
    Defines the data collection modes.
    SINGLE_ACTIVITY: Collect data for one activity at a time.
    COMBINED_SEQUENCE: Collect data for a sequence of activities.
    """
    SINGLE_ACTIVITY = 0
    COMBINED_SEQUENCE = 4

def seconds_to_samples(seconds: float, sampling_rate: int) -> int:
    """
    Converts a duration in seconds to the number of samples based on the sampling rate.

    Args:
        seconds (float): The duration in seconds.
        sampling_rate (int): The sampling rate of the sensor (samples per second).

    Returns:
        int: The equivalent number of samples.
    """
    return int(seconds * sampling_rate)

def preprocess_eeg_data(channel_data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Applies preprocessing steps (detrend, bandpass, bandstop filters) to EEG channel data.

    Args:
        channel_data (np.ndarray): Raw EEG data for a single channel.
        sampling_rate (int): The sampling rate of the EEG data.

    Returns:
        np.ndarray: The preprocessed EEG data.
    """
    # Import DataFilter and filter types locally to avoid circular import issues if
    # utils were to import from armband_connection or vice versa, and to keep
    # the preprocessing logic encapsulated.
    from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations

    # Detrending to remove DC offset
    DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
    # Bandpass filter (2 Hz to 60 Hz) to remove very low and very high frequencies
    DataFilter.perform_bandpass(channel_data, sampling_rate, 2.0, 60.0, 4,
                                FilterTypes.BUTTERWORTH.value, 0)
    # Bandstop filter (48 Hz to 52 Hz) to remove power line noise (e.g., 50 Hz or 60 Hz)
    DataFilter.perform_bandstop(channel_data, sampling_rate, 48.0, 52.0, 4,
                                FilterTypes.BUTTERWORTH.value, 0)
    return channel_data

