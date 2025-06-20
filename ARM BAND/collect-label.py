import os
import sys
import csv
import time
from enum import Enum
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QGroupBox, QStatusBar,
                             QLabel, QLineEdit, QProgressBar, QMessageBox) # Import QMessageBox for pop-ups
from PyQt5.QtCore import QTimer, Qt
from PyQt5 import QtGui
import pyqtgraph as pg
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations

class ActivityState(Enum):
    IDLE = 0
    WRITING = 1
    OTHER = 2

# Define collection modes
class CollectionMode(Enum):
    SINGLE_ACTIVITY = 0 # Default mode for individual Idle, Writing, Other
    COMBINED_SEQUENCE = 4 # Mode 4 for combined collection

class DataCollector(QMainWindow):
    def __init__(self):
        super().__init__()

        # Data collection variables - Initialize these first
        self.current_activity = ActivityState.IDLE # This will reflect the activity being recorded
        self.is_recording = False
        self.recorded_samples = 0
        self.target_samples = 0
        self.data_buffer = deque()

        # New variables for mode management
        self.collection_mode = CollectionMode.SINGLE_ACTIVITY # Default to single activity mode
        self.combined_session_active = False # True when Mode 4 is active
        self.mode_4_sequence = [ActivityState.IDLE, ActivityState.WRITING, ActivityState.OTHER] # Default sequence
        self.mode_4_current_step = 0 # Index for the current activity in the sequence
        self.combined_csv_filename = None # Stores the base filename for Mode 4

        # Initialize UI
        self.init_ui()

        # Board setup
        self.board_shim = None
        self.exg_channels = []
        self.sampling_rate = 250  # Default, will be updated
        self.setup_board()

        # Update the samples/sec label after sampling_rate is set
        self.update_samples_per_sec_label()

        # Set initial UI state based on default mode
        self.set_ui_for_mode(self.collection_mode)


    def init_ui(self):
        self.setWindowTitle('EEG Writing Activity Collector')
        self.setGeometry(100, 100, 800, 600)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Mode Selection Panel
        mode_panel = QGroupBox("Collection Mode")
        mode_layout = QHBoxLayout()

        self.mode1_btn = QPushButton("Mode 1 (Idle)")
        self.mode1_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.SINGLE_ACTIVITY, ActivityState.IDLE))
        mode_layout.addWidget(self.mode1_btn)

        self.mode2_btn = QPushButton("Mode 2 (Writing)")
        self.mode2_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.SINGLE_ACTIVITY, ActivityState.WRITING))
        mode_layout.addWidget(self.mode2_btn)

        self.mode3_btn = QPushButton("Mode 3 (Other)")
        self.mode3_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.SINGLE_ACTIVITY, ActivityState.OTHER))
        mode_layout.addWidget(self.mode3_btn)

        self.mode4_btn = QPushButton("Mode 4 (Combined)")
        self.mode4_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.COMBINED_SEQUENCE))
        mode_layout.addWidget(self.mode4_btn)

        mode_panel.setLayout(mode_layout)
        layout.addWidget(mode_panel)


        # Recording Control Panel
        control_panel = QGroupBox("Recording Control")
        control_layout = QVBoxLayout()

        # Sample Count Input
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Target Samples:"))
        self.sample_input = QLineEdit("1000")
        self.sample_input.setValidator(QtGui.QIntValidator(100, 100000))
        sample_layout.addWidget(self.sample_input)

        # Placeholder for samples/sec label, will be updated after board setup
        self.samples_per_sec_label = QLabel()
        sample_layout.addWidget(self.samples_per_sec_label)

        control_layout.addLayout(sample_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.progress_bar)

        # Activity Buttons (now only used in Combined Mode for selection, or for direct single mode indication)
        self.activity_selection_group = QGroupBox("Select Activity (for Combined Mode)")
        self.activity_selection_layout = QHBoxLayout()

        self.idle_btn = QPushButton("Idle")
        self.idle_btn.clicked.connect(lambda: self.set_activity_and_enable_start(ActivityState.IDLE))
        self.activity_selection_layout.addWidget(self.idle_btn)

        self.writing_btn = QPushButton("Writing")
        self.writing_btn.clicked.connect(lambda: self.set_activity_and_enable_start(ActivityState.WRITING))
        self.activity_selection_layout.addWidget(self.writing_btn)

        self.other_btn = QPushButton("Other")
        self.other_btn.clicked.connect(lambda: self.set_activity_and_enable_start(ActivityState.OTHER))
        self.activity_selection_layout.addWidget(self.other_btn)

        self.activity_selection_group.setLayout(self.activity_selection_layout)
        control_layout.addWidget(self.activity_selection_group) # Add this group to the control layout


        # Start/Stop Button
        self.start_btn = QPushButton("Start Recording")
        self.start_btn.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.start_btn)

        # >>> MODIFIKASI DIMULAI DI SINI <<<
        # Reset Button for Combined Mode
        self.reset_combined_btn = QPushButton("Reset Combined Mode")
        self.reset_combined_btn.clicked.connect(self.reset_combined_mode)
        control_layout.addWidget(self.reset_combined_btn)
        # >>> MODIFIKASI BERAKHIR DI SINI <<<

        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)

        # Visualization
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status()

        # Timer for data update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # Update every 50ms

    def set_ui_for_mode(self, mode):
        # Reset current state for new mode selection
        self.is_recording = False
        self.start_btn.setText("Start Recording")
        self.progress_bar.setValue(0)
        self.recorded_samples = 0
        self.target_samples = 0
        self.data_buffer.clear()
        self.combined_session_active = False # Reset combined session status
        self.mode_4_current_step = 0
        self.combined_csv_filename = None

        if mode == CollectionMode.SINGLE_ACTIVITY:
            self.activity_selection_group.hide() # Hide activity selection group for single modes
            self.start_btn.setEnabled(True) # Start button always enabled for single mode (set_activity handles current_activity)
            self.reset_combined_btn.hide() # Hide reset button in single activity modes
            self.update_status(f"Mode: {self.current_activity.name}. Ready to record.")
        elif mode == CollectionMode.COMBINED_SEQUENCE:
            self.activity_selection_group.show() # Show activity selection group for combined mode
            self.start_btn.setEnabled(False) # Start button disabled until an activity is selected
            self.reset_combined_btn.show() # Show reset button in combined mode
            self.update_status(f"Mode: Combined. Select the first activity to record.")


    def set_collection_mode(self, mode: CollectionMode, initial_activity: ActivityState = None):
        if self.is_recording:
            QMessageBox.warning(self, "Recording Active", "Please stop the current recording before changing modes.")
            return

        self.collection_mode = mode
        if initial_activity: # For Mode 1, 2, 3 selection
            self.current_activity = initial_activity
            self.set_ui_for_mode(self.collection_mode)
        else: # For Mode 4
            self.set_ui_for_mode(self.collection_mode)
            self.combined_session_active = True
            # Re-enable all activity selection buttons for combined mode start
            self.idle_btn.setEnabled(True)
            self.writing_btn.setEnabled(True)
            self.other_btn.setEnabled(True)


    def set_activity_and_enable_start(self, activity: ActivityState):
        if self.combined_session_active:
            self.current_activity = activity # This is the activity for the current segment in combined mode
            self.start_btn.setEnabled(True)
            # Disable other activity buttons until this segment is done
            for btn in [self.idle_btn, self.writing_btn, self.other_btn]:
                if btn.text().upper() != activity.name:
                    btn.setEnabled(False)
                else:
                    btn.setEnabled(True) # Keep the current selected button enabled
            self.start_btn.setText(f"Start {activity.name} Recording")
            self.update_status(f"Mode: Combined. Ready to record {activity.name}. Press Start Recording.")
        else:
            # This logic should ideally not be hit if set_ui_for_mode manages button visibility/enabling correctly
            self.current_activity = activity
            self.update_status(f"Activity: {self.current_activity.name}. Ready to record.")


    def update_samples_per_sec_label(self):
        """Updates the label displaying samples per second."""
        self.samples_per_sec_label.setText(f"(≈ {self.sampling_rate} samples/sec)")

    def seconds_to_samples(self, seconds):
        return int(seconds * self.sampling_rate)

    def setup_board(self):
        try:
            params = MindRoveInputParams()
            self.board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
            self.board_shim.prepare_session()
            self.board_shim.start_stream()

            self.exg_channels = BoardShim.get_exg_channels(self.board_shim.get_board_id())
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_shim.get_board_id())
            self.num_points = self.seconds_to_samples(4)  # 4 seconds of data

            # Set background color to black
            self.plot_widget.setBackground('k') # 'k' for black

            # Set a fixed Y-axis range to prevent auto-scaling and large appearance
            # Adjust these values based on typical EEG signal amplitude (e.g., in microvolts)
            self.plot_widget.setYRange(-500, 500) # Example range: -500 to 500 microvolts

            # Add white grid lines for better readability
            self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
            self.plot_widget.getPlotItem().getAxis('bottom').setPen(pg.mkPen(color='w'))
            self.plot_widget.getPlotItem().getAxis('left').setPen(pg.mkPen(color='w'))

            # Initialize plots
            self.plots = []
            for i, ch in enumerate(self.exg_channels):
                pen = pg.mkPen(color=(i, len(self.exg_channels)*1.3), width=2)
                plot = self.plot_widget.plot(pen=pen)
                self.plots.append(plot)
            self.update_status("Board initialized. Ready.")
        except Exception as e:
            self.update_status(f"Error initializing board: {str(e)}. Please check MindRove setup.")
            self.start_btn.setEnabled(False) # Disable start button if board fails


    def toggle_recording(self):
        if not self.is_recording:
            # Start recording
            try:
                if not self.board_shim or not self.board_shim.is_prepared():
                    QMessageBox.warning(self, "Board Not Ready", "Board is not prepared or connected. Cannot start recording.")
                    self.update_status("Error: Board not ready.")
                    return

                # Always read target samples when recording starts (or a new segment starts)
                self.target_samples = int(self.sample_input.text())
                if self.target_samples < 100:
                    raise ValueError("Minimum 100 samples")

                self.recorded_samples = 0
                self.data_buffer = deque(maxlen=self.target_samples + self.sampling_rate * 5) # Buffer for target + 5 seconds margin
                self.is_recording = True

                base_dir = r'C:\Users\Dell\Downloads\data'
                os.makedirs(base_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_mode = 'w' # Default to write mode (create new file)

                if self.collection_mode == CollectionMode.SINGLE_ACTIVITY:
                    # Naming for individual modes (Mode 1, 2, 3)
                    self.csv_filename = os.path.join(base_dir, f"writing_activity_{self.current_activity.name}_{timestamp}.csv")
                elif self.collection_mode == CollectionMode.COMBINED_SEQUENCE:
                    if self.combined_csv_filename is None:
                        # First activity in combined mode creates the file
                        self.combined_csv_filename = os.path.join(base_dir, f"writing_activity_COMBINED_{timestamp}.csv")
                        self.csv_filename = self.combined_csv_filename
                    else:
                        # Subsequent activities in combined mode append to the same file
                        self.csv_filename = self.combined_csv_filename
                        file_mode = 'a' # Append mode
                else:
                    self.update_status("Error: Unknown collection mode selected.")
                    self.is_recording = False
                    return

                # Write header only if it's a new file or the first activity in combined mode
                if file_mode == 'w':
                    with open(self.csv_filename, file_mode, newline='') as f:
                        writer = csv.writer(f)
                        header = ['timestamp', 'activity_label'] + [f'ch_{ch}' for ch in self.exg_channels]
                        writer.writerow(header)
                # If in append mode, we assume header is already written by the first segment

                self.start_btn.setText("Stop Recording")
                self.update_status(f"Recording {self.current_activity.name} started...")
                self.last_write_time = time.time() # Reset last write time

            except ValueError as e:
                self.update_status(f"Error: {str(e)}")
                self.is_recording = False
            except Exception as e:
                self.update_status(f"An unexpected error occurred: {str(e)}")
                self.is_recording = False
        else:
            # Stop recording
            self.stop_recording()

    def stop_recording(self):
        if not self.is_recording:
            return # Already stopped or not recording

        self.is_recording = False
        self.write_to_csv()  # Save remaining data in buffer

        if self.collection_mode == CollectionMode.SINGLE_ACTIVITY:
            self.start_btn.setText("Start Recording")
            self.progress_bar.setValue(100)
            self.update_status(f"Recording completed. Saved {self.recorded_samples} samples to {self.csv_filename}")
            self.set_ui_for_mode(self.collection_mode) # Reset UI for next single recording
        elif self.collection_mode == CollectionMode.COMBINED_SEQUENCE:
            self.mode_4_current_step += 1
            if self.mode_4_current_step < len(self.mode_4_sequence):
                next_activity = self.mode_4_sequence[self.mode_4_current_step]
                self.current_activity = next_activity # Update current_activity for status/next segment
                self.start_btn.setText(f"Start {next_activity.name} Recording")
                self.start_btn.setEnabled(True) # Re-enable start button for next segment
                self.progress_bar.setValue(0) # Reset progress for next segment
                self.recorded_samples = 0 # Reset recorded samples for next segment
                self.data_buffer.clear() # Clear buffer for next segment
                # Re-enable next activity button, disable others
                for btn in [self.idle_btn, self.writing_btn, self.other_btn]:
                    if btn.text().upper() != next_activity.name:
                        btn.setEnabled(False)
                    else:
                        btn.setEnabled(True)
                self.update_status(f"{self.mode_4_sequence[self.mode_4_current_step-1].name} completed. Ready for {next_activity.name}. Saved to {self.combined_csv_filename}")
                QMessageBox.information(self, "Activity Complete",
                                        f"'{self.mode_4_sequence[self.mode_4_current_step-1].name}' data collected.\n"
                                        f"Click '{next_activity.name}' button and then 'Start Recording' for the next activity.")
            else:
                # All activities in combined sequence are done
                self.start_btn.setText("Start Recording")
                self.progress_bar.setValue(100)
                self.update_status(f"Combined recording completed. All activities saved to {self.combined_csv_filename}")
                QMessageBox.information(self, "Combined Recording Complete",
                                        f"All combined activities recorded and saved to:\n{self.combined_csv_filename}")
                self.combined_session_active = False # End combined session
                self.combined_csv_filename = None # Clear combined filename
                self.set_ui_for_mode(CollectionMode.SINGLE_ACTIVITY) # Return to default UI state
                # Re-enable all mode buttons
                self.mode1_btn.setEnabled(True)
                self.mode2_btn.setEnabled(True)
                self.mode3_btn.setEnabled(True)
                self.mode4_btn.setEnabled(True)
        self.update_status() # Final update


    # >>> MODIFIKASI DIMULAI DI SINI <<<
    def reset_combined_mode(self):
        if self.is_recording:
            # Stop current recording segment and save data
            self.stop_recording()
            QMessageBox.information(self, "Recording Stopped", "Current segment recording stopped. Combined mode will now reset.")

        # Reset all combined mode specific variables
        self.combined_session_active = False
        self.mode_4_current_step = 0
        self.combined_csv_filename = None
        self.data_buffer.clear() # Ensure data buffer is clear
        self.recorded_samples = 0
        self.progress_bar.setValue(0) # Reset progress bar visually

        # Re-initialize UI for combined mode
        self.set_ui_for_mode(CollectionMode.COMBINED_SEQUENCE)
        # Ensure all activity buttons are re-enabled for a fresh start in combined mode
        self.idle_btn.setEnabled(True)
        self.writing_btn.setEnabled(True)
        self.other_btn.setEnabled(True)
        self.start_btn.setText("Start Recording") # Reset start button text
        self.start_btn.setEnabled(False) # Disable until an activity is selected

        self.update_status("Combined mode has been reset. Select the first activity to begin a new sequence.")
        QMessageBox.information(self, "Combined Mode Reset",
                                "Combined mode has been reset. You can now start a new combined recording sequence.")
    # >>> MODIFIKASI BERAKHIR DI SINI <<<


    def update(self):
        if not self.board_shim or not self.exg_channels:
            return

        data = self.board_shim.get_current_board_data(self.num_points)
        if data.shape[1] == 0:
            return

        # Process and visualize data
        processed_data = []
        for i, ch in enumerate(self.exg_channels):
            channel_data = data[ch].copy()

            # Preprocessing
            DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(channel_data, self.sampling_rate, 2.0, 60.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(channel_data, self.sampling_rate, 48.0, 52.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)

            self.plots[i].setData(channel_data)
            processed_data.append(channel_data)

        # Save to buffer if recording
        if self.is_recording:
            current_time = time.time()
            num_new_samples = data.shape[1]

            for i in range(num_new_samples):
                # Use current_activity for labeling, which is set by mode or combined sequence
                row = [current_time, self.current_activity.value]
                row.extend(ch_data[i] for ch_data in processed_data)
                self.data_buffer.append(row)
                self.recorded_samples += 1

                # Check if target reached
                if self.recorded_samples >= self.target_samples:
                    self.stop_recording()
                    break # Break the loop to avoid processing more samples after target is hit

            # Update progress
            if self.target_samples > 0:
                progress = int((self.recorded_samples / self.target_samples) * 100)
                self.progress_bar.setValue(min(progress, 100))

            # Periodic save to CSV
            if time.time() - getattr(self, 'last_write_time', 0) > 1.0:
                self.write_to_csv()
                self.last_write_time = time.time()

    def write_to_csv(self):
        if not hasattr(self, 'csv_filename') or not self.data_buffer:
            return

        try:
            # Always append for subsequent writes after initial header for combined mode
            file_mode = 'a'
            with open(self.csv_filename, file_mode, newline='') as f:
                writer = csv.writer(f)
                while self.data_buffer:
                    writer.writerow(self.data_buffer.popleft())
        except Exception as e:
            self.update_status(f"Error saving data: {str(e)}")

    def update_status(self, message=None):
        if message:
            self.status_bar.showMessage(message)
        else:
            mode_str = "Mode 1,2,3 (Single)" if self.collection_mode == CollectionMode.SINGLE_ACTIVITY else "Mode 4 (Combined)"
            activity_str = self.current_activity.name
            if self.is_recording:
                status = f"Mode: {mode_str}, Activity: {activity_str} | Recording: {self.recorded_samples}/{self.target_samples} samples"
            else:
                status = f"Mode: {mode_str}, Activity: {activity_str} | Ready"

            if self.collection_mode == CollectionMode.COMBINED_SEQUENCE and self.combined_session_active:
                if self.mode_4_current_step < len(self.mode_4_sequence):
                    next_activity_name = self.mode_4_sequence[self.mode_4_current_step].name
                    status += f" | Next: {next_activity_name}"
                else:
                    status += f" | Sequence Complete"

            self.status_bar.showMessage(status)

    def closeEvent(self, event):
        if self.is_recording:
            self.stop_recording() # Ensure data is saved and session released
            QMessageBox.information(self, "Recording Stopped", "Recording has been stopped and data saved due to application exit.")

        if self.board_shim and self.board_shim.is_prepared():
            try:
                self.board_shim.stop_stream()
                self.board_shim.release_session()
                self.update_status("Board session released.")
            except Exception as e:
                self.update_status(f"Error releasing board session: {str(e)}")

        event.accept()

def main():
    app = QApplication(sys.argv)
    collector = DataCollector()
    collector.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()