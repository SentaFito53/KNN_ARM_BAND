import os
import csv
import time
from collections import deque
from typing import Deque

from utils import ActivityState, CollectionMode # Import enums

class EEGDataCollector:
    """
    Manages the collection and saving of EEG data to CSV files.
    Supports single activity and combined sequence recording modes.
    """
    def __init__(self, base_dir: str = r'C:\Users\Dell\Downloads\data'):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True) # Ensure base directory exists

        self.current_activity: ActivityState = ActivityState.IDLE
        self.is_recording: bool = False
        self.recorded_samples: int = 0
        self.target_samples: int = 0
        self.data_buffer: Deque[list] = deque() # Buffer to store data before writing to CSV

        self.collection_mode: CollectionMode = CollectionMode.SINGLE_ACTIVITY
        self.combined_session_active: bool = False
        self.mode_4_sequence: list[ActivityState] = [ActivityState.IDLE, ActivityState.WRITING, ActivityState.OTHER]
        self.mode_4_current_step: int = 0
        self.combined_csv_filename: str | None = None
        self.csv_filename: str | None = None
        self.last_write_time: float = 0.0

    def start_recording(self, target_samples: int, current_activity: ActivityState,
                        collection_mode: CollectionMode, exg_channels: list[int]) -> tuple[bool, str]:
        """
        Initiates the data recording process.

        Args:
            target_samples (int): The number of samples to record.
            current_activity (ActivityState): The activity label for the current recording.
            collection_mode (CollectionMode): The current collection mode.
            exg_channels (list[int]): List of EXG channel indices for CSV header.

        Returns:
            tuple[bool, str]: A tuple indicating success (True/False) and a status message.
        """
        if self.is_recording:
            return False, "Already recording. Please stop current recording first."
        if target_samples < 100:
            return False, "Minimum 100 samples required for recording."

        self.target_samples = target_samples
        self.current_activity = current_activity
        self.collection_mode = collection_mode
        self.recorded_samples = 0
        self.data_buffer.clear()
        self.is_recording = True

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_mode = 'w'

        if self.collection_mode == CollectionMode.SINGLE_ACTIVITY:
            self.csv_filename = os.path.join(self.base_dir, f"eeg_activity_{self.current_activity.name}_{timestamp}.csv")
        elif self.collection_mode == CollectionMode.COMBINED_SEQUENCE:
            if self.combined_csv_filename is None:
                # This is the first segment of a combined sequence
                self.combined_csv_filename = os.path.join(self.base_dir, f"eeg_activity_COMBINED_{timestamp}.csv")
                self.csv_filename = self.combined_csv_filename
            else:
                # Appending to an existing combined file
                self.csv_filename = self.combined_csv_filename
                file_mode = 'a'

        try:
            # Write header only if it's a new file ('w' mode)
            if file_mode == 'w':
                with open(self.csv_filename, file_mode, newline='') as f:
                    writer = csv.writer(f)
                    header = ['timestamp', 'activity_label'] + [f'ch_{ch}' for ch in exg_channels]
                    writer.writerow(header)
            
            self.last_write_time = time.time()
            return True, f"Recording {self.current_activity.name} started to {os.path.basename(self.csv_filename)}..."
        except Exception as e:
            self.is_recording = False
            return False, f"Error setting up CSV file: {str(e)}"

    def stop_recording(self) -> tuple[bool, str]:
        """
        Stops the data recording process and flushes any buffered data to CSV.

        Returns:
            tuple[bool, str]: A tuple indicating success (True/False) and a status message.
        """
        if not self.is_recording:
            return False, "Not currently recording."

        self.is_recording = False
        self.write_to_csv() # Flush remaining data

        status_msg = f"Recording completed. Saved {self.recorded_samples} samples."
        if self.csv_filename:
             status_msg += f" to {os.path.basename(self.csv_filename)}"
        
        return True, status_msg

    def record_data_point(self, timestamp: float, activity_label: ActivityState, channel_data: list[float]):
        """
        Adds a single data point to the internal buffer.

        Args:
            timestamp (float): The timestamp of the data point.
            activity_label (ActivityState): The activity state for this data point.
            channel_data (list[float]): A list of processed channel values for this sample.
        """
        if not self.is_recording or self.recorded_samples >= self.target_samples:
            return

        csv_row = [timestamp, activity_label.value] + channel_data
        self.data_buffer.append(csv_row)
        self.recorded_samples += 1

        # Automatically write to CSV every second or when buffer is full (if needed)
        if time.time() - self.last_write_time > 1.0 or len(self.data_buffer) >= 1000: # Example threshold
            self.write_to_csv()
            self.last_write_time = time.time()

    def write_to_csv(self):
        """
        Writes buffered data to the CSV file.
        """
        if not self.csv_filename or not self.data_buffer:
            return

        try:
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                while self.data_buffer:
                    writer.writerow(self.data_buffer.popleft())
        except Exception as e:
            # In a real GUI app, you'd signal this error to the main UI
            print(f"Error saving data to CSV: {e}")

    def get_progress(self) -> int:
        """
        Returns the current recording progress as a percentage.

        Returns:
            int: Progress percentage (0-100).
        """
        if self.target_samples == 0:
            return 0
        return int((self.recorded_samples / self.target_samples) * 100)

    def reset_combined_mode(self):
        """
        Resets the state variables specifically for the combined collection mode.
        """
        self.combined_session_active = False
        self.mode_4_current_step = 0
        self.combined_csv_filename = None
        self.data_buffer.clear()
        self.recorded_samples = 0
        self.target_samples = 0
        self.is_recording = False
        self.csv_filename = None

    def get_combined_sequence_info(self) -> tuple[int, int, ActivityState | None]:
        """
        Returns information about the current step in a combined sequence.

        Returns:
            tuple[int, int, ActivityState | None]: (current_step, total_steps, next_activity).
        """
        total_steps = len(self.mode_4_sequence)
        next_activity = self.mode_4_sequence[self.mode_4_current_step] if self.mode_4_current_step < total_steps else None
        return self.mode_4_current_step, total_steps, next_activity

    def advance_combined_step(self) -> bool:
        """
        Advances to the next step in the combined recording sequence.

        Returns:
            bool: True if there is a next step, False if the sequence is complete.
        """
        self.mode_4_current_step += 1
        return self.mode_4_current_step < len(self.mode_4_sequence)

    # Properties to expose internal state read-only if needed by UI
    @property
    def is_recording_active(self) -> bool:
        return self.is_recording

    @property
    def current_activity_state(self) -> ActivityState:
        return self.current_activity

    @property
    def current_target_samples(self) -> int:
        return self.target_samples

    @property
    def current_recorded_samples(self) -> int:
        return self.recorded_samples

    @property
    def get_current_csv_filename(self) -> str | None:
        return self.csv_filename

    @property
    def is_combined_session_active(self) -> bool:
        return self.combined_session_active

    @property
    def get_mode_4_sequence(self) -> list[ActivityState]:
        return self.mode_4_sequence

    @property
    def get_mode_4_current_step(self) -> int:
        return self.mode_4_current_step

