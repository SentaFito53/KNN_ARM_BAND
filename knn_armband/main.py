import sys
import os
import time
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QGroupBox, QStatusBar,
                             QLabel, QLineEdit, QProgressBar, QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5 import QtGui
import pyqtgraph as pg

# Import custom modules
from utils import ActivityState, CollectionMode, seconds_to_samples
from armband_connection import ArmbandConnection
from classification import EEGClassifier
from collect_data import EEGDataCollector

class DataCollectorApp(QMainWindow):
    """
    Main application window for EEG data collection and real-time classification.
    """
    def __init__(self):
        super().__init__()

        # Initialize core components
        self.armband_conn = ArmbandConnection()
        self.eeg_classifier = EEGClassifier(model_path='model_knn.pkl', scaler_path='scaler_knn.pkl')
        self.data_collector = EEGDataCollector(base_dir=r'C:\Users\Dell\Downloads\data')

        # UI components initialization
        self.init_ui()

        # Board and model setup (initial calls)
        self.setup_board_and_models()

        # Update labels and UI state based on initial setup
        self.update_samples_per_sec_label()
        self.set_ui_for_mode(self.data_collector.collection_mode)

        # QTimer for real-time data updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_eeg_data_and_ui)
        self.timer.start(50) # Update every 50ms (20 FPS)

    def init_ui(self):
        """Initializes the main graphical user interface."""
        self.setWindowTitle('EEG Activity Collector & Classifier')
        self.setGeometry(100, 100, 1000, 700) # Set initial window size and position

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Top control layout (contains Mode, Recording, and Classification panels)
        top_control_layout = QHBoxLayout()

        # --- Mode Panel ---
        mode_panel = QGroupBox("Collection Mode")
        mode_layout = QVBoxLayout()
        mode_btn_layout = QHBoxLayout()

        # Buttons for single activity modes
        self.mode1_btn = QPushButton("Mode 1 (Idle)")
        self.mode1_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.SINGLE_ACTIVITY, ActivityState.IDLE))
        mode_btn_layout.addWidget(self.mode1_btn)

        self.mode2_btn = QPushButton("Mode 2 (Writing)")
        self.mode2_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.SINGLE_ACTIVITY, ActivityState.WRITING))
        mode_btn_layout.addWidget(self.mode2_btn)

        self.mode3_btn = QPushButton("Mode 3 (Other)")
        self.mode3_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.SINGLE_ACTIVITY, ActivityState.OTHER))
        mode_btn_layout.addWidget(self.mode3_btn)

        mode_btn_layout.addStretch(1) # Pushes buttons to the left
        mode_layout.addLayout(mode_btn_layout)

        # Button for combined sequence mode
        self.mode4_btn = QPushButton("Mode 4 (Combined Sequence)")
        self.mode4_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.COMBINED_SEQUENCE))
        mode_layout.addWidget(self.mode4_btn)

        mode_panel.setLayout(mode_layout)
        top_control_layout.addWidget(mode_panel)

        # --- Recording Control Panel ---
        control_panel = QGroupBox("Recording Control")
        control_layout = QVBoxLayout()

        # Target samples input and sampling rate display
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Target Samples:"))
        self.sample_input = QLineEdit("1000") # Default target samples
        self.sample_input.setValidator(QtGui.QIntValidator(100, 100000)) # Validator for input
        sample_layout.addWidget(self.sample_input)
        self.samples_per_sec_label = QLabel() # Displays samples per second
        sample_layout.addWidget(self.samples_per_sec_label)
        control_layout.addLayout(sample_layout)

        self.progress_bar = QProgressBar() # Progress bar for recording
        self.progress_bar.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.progress_bar)

        # Activity selection group (for combined mode)
        self.activity_selection_group = QGroupBox("Select Current Activity (Combined Mode)")
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
        control_layout.addWidget(self.activity_selection_group)

        self.start_btn = QPushButton("Start Recording")
        self.start_btn.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.start_btn)

        self.reset_combined_btn = QPushButton("Reset Combined Mode")
        self.reset_combined_btn.clicked.connect(self.reset_combined_mode)
        control_layout.addWidget(self.reset_combined_btn)

        control_panel.setLayout(control_layout)
        top_control_layout.addWidget(control_panel)

        # --- KNN Classification Panel ---
        classification_panel = QGroupBox("KNN Classification")
        classification_layout = QVBoxLayout()

        self.classify_start_btn = QPushButton("Start Classification")
        self.classify_start_btn.clicked.connect(self.toggle_classification)
        classification_layout.addWidget(self.classify_start_btn)

        self.prediction_label = QLabel("Prediction: N/A") # Displays real-time prediction
        self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        classification_layout.addWidget(self.prediction_label)

        classification_panel.setLayout(classification_layout)
        top_control_layout.addWidget(classification_panel)

        layout.addLayout(top_control_layout) # Add the top control layout to the main layout

        # --- Plot Widget (Real-time EEG signal display) ---
        self.plot_widget = pg.PlotWidget()
        self.configure_plot_widget() # Set up plot aesthetics
        layout.addWidget(self.plot_widget)

        # Status bar at the bottom
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status() # Initial status message

    def configure_plot_widget(self):
        """Configures the aesthetics and initial state of the pyqtgraph plot widget."""
        self.plot_widget.setBackground('k') # Set background color to black
        self.plot_widget.setYRange(-500, 500) # Fixed Y-axis range for EEG signals (e.g., in microvolts)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.5) # Add white grid lines
        self.plot_widget.getPlotItem().getAxis('bottom').setPen(pg.mkPen(color='w'))
        self.plot_widget.getPlotItem().getAxis('left').setPen(pg.mkPen(color='w'))

        self.plots = [] # List to hold individual channel plots
        # Initial plots will be created in setup_board_and_models after channels are known

    def setup_board_and_models(self):
        """
        Initializes the armband connection and loads the classification models.
        Updates UI based on success/failure.
        """
        # Setup Armband Connection
        success, message = self.armband_conn.setup_board()
        self.update_status(message, is_error=not success)
        if not success:
            QMessageBox.critical(self, "Board Error", f"Failed to initialize MindRove board:\n{message}\n"
                                                      "Please ensure board is connected and drivers are installed.")
            self.start_btn.setEnabled(False)
            self.classify_start_btn.setEnabled(False)
            return
        
        # Initialize plots with the color scheme
        exg_channels = self.armband_conn.get_exg_channels()
        for i, _ in enumerate(exg_channels):
            # Use pyqtgraph's built-in color cycle for different channels
            pen = pg.mkPen(color=(i, len(exg_channels) * 1.3), width=2)
            plot = self.plot_widget.plot(pen=pen)
            self.plots.append(plot)

        # Load KNN Classification Model
        success, message = self.eeg_classifier.load_models()
        self.update_status(message, is_error=not success)
        if not success:
            QMessageBox.critical(self, "Model Load Error", f"Model/Scaler Load Error:\n{message}\n"
                                                          "Classification will be disabled.")
            self.classify_start_btn.setEnabled(False)
        else:
            self.classify_start_btn.setEnabled(True)


    def update_samples_per_sec_label(self):
        """Updates the label displaying the samples per second based on sampling rate."""
        self.samples_per_sec_label.setText(f"(≈ {self.armband_conn.get_sampling_rate()} samples/sec)")

    def set_ui_for_mode(self, mode: CollectionMode):
        """
        Adjusts the UI elements based on the selected collection mode.
        Resets recording state.
        """
        # Ensure recording and classification are stopped before changing modes
        if self.data_collector.is_recording_active:
            self.toggle_recording() # This will stop recording and update UI
        
        if self.classify_start_btn.text() == "Stop Classification":
            self.toggle_classification() # This will stop classification and update UI

        self.data_collector.collection_mode = mode # Set the mode in the data collector

        self.progress_bar.setValue(0)
        self.prediction_label.setText("Prediction: N/A")
        self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50;")


        # Enable all mode buttons
        self.mode1_btn.setEnabled(True)
        self.mode2_btn.setEnabled(True)
        self.mode3_btn.setEnabled(True)
        self.mode4_btn.setEnabled(True)

        if mode == CollectionMode.SINGLE_ACTIVITY:
            self.activity_selection_group.hide()
            self.start_btn.setEnabled(True) # Always enabled for single mode once activity is set
            self.reset_combined_btn.hide()
            # Restore activity button states if they were disabled by combined mode
            for btn in [self.idle_btn, self.writing_btn, self.other_btn]:
                btn.setEnabled(True) 
            self.start_btn.setText("Start Recording")
            self.update_status(f"Mode: {self.data_collector.current_activity_state.name}. Ready to record.")
        elif mode == CollectionMode.COMBINED_SEQUENCE:
            self.data_collector.combined_session_active = True
            self.data_collector.mode_4_current_step = 0 # Reset combined step
            self.data_collector.combined_csv_filename = None # Clear combined filename
            
            self.activity_selection_group.show()
            self.start_btn.setEnabled(False) # Disabled until an activity is selected
            self.reset_combined_btn.show()
            # Enable activity selection buttons for combined mode start
            for btn in [self.idle_btn, self.writing_btn, self.other_btn]:
                btn.setEnabled(True)
            self.update_status(f"Mode: Combined. Select first activity to record.")

    def set_collection_mode(self, mode: CollectionMode, initial_activity: ActivityState = None):
        """
        Sets the collection mode and optionally an initial activity state.
        Includes checks to prevent mode change during active operations.
        """
        if self.data_collector.is_recording_active:
            QMessageBox.warning(self, "Recording Active", "Please stop current recording before changing modes.")
            return
        if self.classify_start_btn.text() == "Stop Classification":
            QMessageBox.warning(self, "Classification Active", "Please stop classification before changing modes.")
            return

        self.data_collector.collection_mode = mode
        if initial_activity:
            self.data_collector.current_activity = initial_activity
        self.set_ui_for_mode(self.data_collector.collection_mode)

    def set_activity_and_enable_start(self, activity: ActivityState):
        """
        Sets the current activity for recording and enables the start button
        if in combined mode.
        """
        if self.data_collector.is_combined_session_active:
            self.data_collector.current_activity = activity
            self.start_btn.setEnabled(True)
            # Disable other activity buttons to indicate current selection
            for btn in [self.idle_btn, self.writing_btn, self.other_btn]:
                btn_text_enum = getattr(ActivityState, btn.text().upper(), None)
                if btn_text_enum != activity:
                    btn.setEnabled(False)
                else:
                    btn.setEnabled(True) # Keep current selected activity button enabled
            self.start_btn.setText(f"Start Recording {activity.name}")
            self.update_status(f"Mode: Combined. Ready to record {activity.name}. Press 'Start Recording'.")
        else:
            self.data_collector.current_activity = activity
            self.update_status(f"Activity: {self.data_collector.current_activity_state.name}. Ready to record.")

    def toggle_recording(self):
        """Toggles the data recording state (start/stop)."""
        if not self.armband_conn.board_shim or not self.armband_conn.board_shim.is_prepared():
            QMessageBox.warning(self, "Board Not Ready", "EEG device not prepared or connected. Cannot start recording.")
            self.update_status("Error: EEG device not ready.", is_error=True)
            return

        if self.classify_start_btn.text() == "Stop Classification":
            QMessageBox.warning(self, "Classification Active", "Please stop real-time classification before starting recording.")
            return

        if not self.data_collector.is_recording_active:
            # Start Recording
            try:
                target_samples = int(self.sample_input.text())
                success, message = self.data_collector.start_recording(
                    target_samples,
                    self.data_collector.current_activity_state,
                    self.data_collector.collection_mode,
                    self.armband_conn.get_exg_channels()
                )
                if success:
                    self.start_btn.setText("Stop Recording")
                    self.update_status(message)
                    # Disable mode buttons when recording
                    self.mode1_btn.setEnabled(False)
                    self.mode2_btn.setEnabled(False)
                    self.mode3_btn.setEnabled(False)
                    self.mode4_btn.setEnabled(False)
                    # For combined mode, disable activity buttons once recording starts
                    if self.data_collector.is_combined_session_active:
                        for btn in [self.idle_btn, self.writing_btn, self.other_btn]:
                            btn.setEnabled(False)

                else:
                    QMessageBox.warning(self, "Recording Error", message)
                    self.update_status(message, is_error=True)
            except ValueError as e:
                QMessageBox.warning(self, "Input Error", f"Invalid target samples: {e}")
                self.update_status(f"Error: {e}", is_error=True)
            except Exception as e:
                QMessageBox.critical(self, "Recording Error", f"An unexpected error occurred during recording setup: {e}")
                self.update_status(f"Unexpected error: {e}", is_error=True)
        else:
            # Stop Recording
            self._finalize_recording_segment()

    def _finalize_recording_segment(self):
        """Internal method to stop recording a segment and handle UI updates."""
        success, message = self.data_collector.stop_recording()
        self.start_btn.setText("Start Recording")
        self.progress_bar.setValue(100) # Ensure progress bar is full on stop
        self.update_status(message)

        # Re-enable mode buttons after recording stops (for single mode)
        self.mode1_btn.setEnabled(True)
        self.mode2_btn.setEnabled(True)
        self.mode3_btn.setEnabled(True)
        self.mode4_btn.setEnabled(True)


        if self.data_collector.collection_mode == CollectionMode.SINGLE_ACTIVITY:
            QMessageBox.information(self, "Recording Complete", message)
            self.set_ui_for_mode(CollectionMode.SINGLE_ACTIVITY) # Reset UI for next single recording
        elif self.data_collector.collection_mode == CollectionMode.COMBINED_SEQUENCE:
            if self.data_collector.advance_combined_step():
                # Not the last step in combined sequence
                current_step, total_steps, next_activity = self.data_collector.get_combined_sequence_info()
                self.data_collector.current_activity = next_activity # Set next activity for collector
                self.start_btn.setText(f"Start Recording {next_activity.name}")
                self.start_btn.setEnabled(True) # Enable start for next segment
                self.progress_bar.setValue(0)
                # Re-enable only the next activity's button
                for btn in [self.idle_btn, self.writing_btn, self.other_btn]:
                    btn_text_enum = getattr(ActivityState, btn.text().upper(), None)
                    if btn_text_enum != next_activity:
                        btn.setEnabled(False)
                    else:
                        btn.setEnabled(True)
                QMessageBox.information(self, "Activity Complete",
                                        f"'{self.data_collector.get_mode_4_sequence[current_step - 1].name}' data collected.\n"
                                        f"Click '{next_activity.name}' then 'Start Recording' for the next activity.")
                self.update_status(f"{self.data_collector.get_mode_4_sequence[current_step-1].name} complete. Ready for {next_activity.name}. Combined Step: {current_step+1}/{total_steps}")
            else:
                # All steps in combined sequence are complete
                QMessageBox.information(self, "Combined Recording Complete", message)
                self.data_collector.reset_combined_mode() # Reset collector state
                self.set_ui_for_mode(CollectionMode.SINGLE_ACTIVITY) # Revert UI to single activity mode
                # Ensure all mode buttons are enabled as combined session is over
                self.mode1_btn.setEnabled(True)
                self.mode2_btn.setEnabled(True)
                self.mode3_btn.setEnabled(True)
                self.mode4_btn.setEnabled(True)
                # Ensure all activity selection buttons are enabled too
                for btn in [self.idle_btn, self.writing_btn, self.other_btn]:
                    btn.setEnabled(True)

    def reset_combined_mode(self):
        """Resets the combined recording mode, stopping any active recording if necessary."""
        if self.data_collector.is_recording_active:
            self._finalize_recording_segment() # Stops the current segment
            QMessageBox.information(self, "Recording Stopped", "Current segment stopped. Combined mode will be reset.")

        self.data_collector.reset_combined_mode() # Reset the data collector's state
        self.set_ui_for_mode(CollectionMode.COMBINED_SEQUENCE) # Re-configure UI for combined mode start
        self.start_btn.setText("Start Recording")
        self.start_btn.setEnabled(False) # Disable start button until activity is selected

        QMessageBox.information(self, "Combined Mode Reset",
                               "Combined mode has been reset. You can now start a new combined recording sequence.")
        self.update_status("Combined mode reset. Select first activity to start new sequence.")

    def toggle_classification(self):
        """Toggles real-time classification on/off."""
        if not self.eeg_classifier.is_ready:
            QMessageBox.warning(self, "Model Not Loaded", "KNN model not loaded. Cannot start classification.")
            return

        # If recording is active, prevent classification
        if self.data_collector.is_recording_active:
            QMessageBox.warning(self, "Recording Active", "Please stop data recording before starting real-time classification.")
            return

        if self.classify_start_btn.text() == "Stop Classification":
            # Stop classification
            self.classify_start_btn.setText("Start Classification")
            self.prediction_label.setText("Prediction: N/A")
            self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50;")
            self.update_status("Real-time classification stopped.")
        else:
            # Start classification
            self.classify_start_btn.setText("Stop Classification")
            self.update_status("Real-time classification started...")

    def update_eeg_data_and_ui(self):
        """
        Called by the QTimer to periodically get EEG data, update plots,
        perform classification (if active), and record data (if active).
        """
        # Get and preprocess data from the armband
        processed_data_for_plot, latest_sample_features = self.armband_conn.get_and_preprocess_data()

        if processed_data_for_plot.size == 0:
            return # No new data

        # Update real-time EEG plots
        for i, plot in enumerate(self.plots):
            if i < processed_data_for_plot.shape[0]: # Ensure channel exists
                plot.setData(processed_data_for_plot[i])

        # Perform KNN Classification if active
        if self.classify_start_btn.text() == "Stop Classification" and self.eeg_classifier.is_ready:
            if len(latest_sample_features) == len(self.armband_conn.get_exg_channels()):
                predicted_activity, predicted_name = self.eeg_classifier.predict(np.array(latest_sample_features))
                if predicted_activity is not None:
                    self.prediction_label.setText(f"Prediction: {predicted_name}")
                    # Update label color based on prediction
                    if predicted_activity == ActivityState.WRITING:
                        self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFD700;") # Gold
                    elif predicted_activity == ActivityState.IDLE:
                        self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00BFFF;") # Deep Sky Blue
                    else: # OTHER
                        self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FF6347;") # Tomato
                else:
                    self.prediction_label.setText("Prediction Error!")
                    self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: red;")
                    self.update_status(f"Error during KNN prediction.", is_error=True)
            else:
                self.prediction_label.setText("Insufficient Features!")
                self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: gray;")


        # Record Data if active
        if self.data_collector.is_recording_active:
            current_timestamp = time.time()
            num_new_samples_to_process = processed_data_for_plot.shape[1] # Number of new samples received

            for i in range(num_new_samples_to_process):
                if self.data_collector.current_recorded_samples >= self.data_collector.current_target_samples:
                    self._finalize_recording_segment() # Stop recording if target reached
                    break

                # Extract one sample across all channels for recording
                sample_for_csv = [processed_data_for_plot[ch_idx][i] for ch_idx in range(processed_data_for_plot.shape[0])]
                self.data_collector.record_data_point(
                    current_timestamp,
                    self.data_collector.current_activity_state,
                    sample_for_csv
                )
            
            # Update progress bar even if recording is not finished in this loop iteration
            self.progress_bar.setValue(min(self.data_collector.get_progress(), 100))
            self.update_status() # Update status bar with current recording info

    def update_status(self, message: str = None, is_error: bool = False):
        """
        Updates the application's status bar with a given message or current operational status.

        Args:
            message (str, optional): A specific message to display. If None,
                                     a dynamic status based on app state is generated.
            is_error (bool, optional): If True, the status message is displayed in red.
        """
        if message:
            self.status_bar.showMessage(message)
        else:
            mode_str = "Single Activity" if self.data_collector.collection_mode == CollectionMode.SINGLE_ACTIVITY else "Combined Sequence"
            
            activity_str = self.data_collector.current_activity_state.name
            if self.data_collector.is_combined_session_active and not self.data_collector.is_recording_active:
                _, _, next_activity = self.data_collector.get_combined_sequence_info()
                if next_activity:
                    activity_str = f"{next_activity.name} (Next)"
                else:
                    activity_str = "Complete"

            status_parts = [f"Mode: {mode_str}"]
            if self.data_collector.is_recording_active:
                status_parts.append(f"Recording: {activity_str} {self.data_collector.current_recorded_samples}/{self.data_collector.current_target_samples} samples")
            elif self.classify_start_btn.text() == "Stop Classification":
                status_parts.append("Classifying: ON")
            else:
                status_parts.append(f"Ready. Activity: {activity_str}")

            if self.data_collector.is_combined_session_active:
                current_step, total_steps, _ = self.data_collector.get_combined_sequence_info()
                status_parts.append(f"Combined Step: {current_step + 1}/{total_steps}")

            self.status_bar.showMessage(" | ".join(status_parts))

        # Apply error styling if needed
        if is_error:
            self.status_bar.setStyleSheet("QStatusBar::item {border: none;} QLabel {color: red;}")
        else:
            self.status_bar.setStyleSheet("") # Clear custom style

    def closeEvent(self, event):
        """
        Handles the application close event, ensuring proper shutdown of recording
        and armband connection.
        """
        if self.data_collector.is_recording_active:
            self.data_collector.stop_recording()
            QMessageBox.information(self, "Recording Stopped", "Recording stopped and data saved due to application closing.")

        success, message = self.armband_conn.stop_and_release()
        self.update_status(message, is_error=not success)
        if not success:
            QMessageBox.critical(self, "Board Close Error", f"Error closing MindRove board:\n{message}")

        event.accept() # Accept the close event

def main():
    app = QApplication(sys.argv)
    collector = DataCollectorApp()
    collector.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

