import numpy as np
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations

from utils import seconds_to_samples, preprocess_eeg_data # Import utility functions

class ArmbandConnection:
    """
    Manages the connection and data acquisition from the MindRove armband.
    """
    def __init__(self):
        self.board_shim = None
        self.exg_channels = []
        self.sampling_rate = 0 # Will be updated after board initialization
        self.num_points = 0 # Number of samples to get at each update (e.g., 4 seconds of data)

    def setup_board(self, num_seconds_for_plot: int = 4) -> tuple[bool, str]:
        """
        Initializes the MindRove board session and prepares it for streaming.

        Args:
            num_seconds_for_plot (int): The number of seconds of data to keep for plotting.

        Returns:
            tuple[bool, str]: A tuple where the first element is True if setup was successful,
                              False otherwise. The second element is a status message.
        """
        try:
            params = MindRoveInputParams()
            # Initialize the BoardShim object with the MindRove WiFi board ID
            self.board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
            # Prepare the board session
            self.board_shim.prepare_session()
            # Start data streaming
            self.board_shim.start_stream()

            # Get EXG (EEG) channel indices for the board
            self.exg_channels = BoardShim.get_exg_channels(self.board_shim.get_board_id())
            # Get the sampling rate of the board
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_shim.get_board_id())
            # Calculate the number of points (samples) for real-time plotting
            self.num_points = seconds_to_samples(num_seconds_for_plot, self.sampling_rate)

            return True, "Board initialized. Ready to record/classify."
        except Exception as e:
            # Handle any errors during board initialization
            self.board_shim = None
            return False, f"Error initializing board: {str(e)}. Please check MindRove setup."

    def get_and_preprocess_data(self) -> tuple[np.ndarray, list[float]]:
        """
        Gets the current board data, preprocesses it, and prepares it for plotting
        and feature extraction.

        Returns:
            tuple[np.ndarray, list[float]]: A tuple containing:
                - np.ndarray: Processed data for all EXG channels, suitable for plotting.
                              Shape: (num_channels, num_points).
                - list[float]: Latest feature vector (last sample of each processed channel).
        """
        if not self.board_shim or not self.board_shim.is_prepared():
            return np.array([]), [] # Return empty if board is not ready

        # Get current data from the board
        data = self.board_shim.get_current_board_data(self.num_points)
        if data.shape[1] == 0: # If no new data
            return np.array([]), []

        processed_data_for_plot = []
        latest_sample_features = []

        for ch_idx in self.exg_channels:
            channel_data = data[ch_idx].copy()
            # Preprocess the channel data using the utility function
            processed_channel = preprocess_eeg_data(channel_data, self.sampling_rate)
            processed_data_for_plot.append(processed_channel)

            if processed_channel.size > 0:
                latest_sample_features.append(processed_channel[-1])

        # Convert list of arrays to a single NumPy array for consistent plotting
        if processed_data_for_plot:
            return np.array(processed_data_for_plot), latest_sample_features
        else:
            return np.array([]), []

    def get_sampling_rate(self) -> int:
        """
        Returns the sampling rate of the connected armband.

        Returns:
            int: The sampling rate in Hz.
        """
        return self.sampling_rate

    def get_exg_channels(self) -> list[int]:
        """
        Returns the list of EXG (EEG) channel indices.

        Returns:
            list[int]: A list of integers representing the EXG channel indices.
        """
        return self.exg_channels

    def stop_and_release(self) -> tuple[bool, str]:
        """
        Stops the data stream and releases the board session.

        Returns:
            tuple[bool, str]: A tuple where the first element is True if successful,
                              False otherwise. The second element is a status message.
        """
        if self.board_shim and self.board_shim.is_prepared():
            try:
                self.board_shim.stop_stream()
                self.board_shim.release_session()
                return True, "Board session released."
            except Exception as e:
                return False, f"Error releasing board session: {str(e)}"
        return True, "No board session to release." # Already not prepared or None

