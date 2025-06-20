import os
import sys
import csv
import time
from enum import Enum
from collections import deque
import numpy as np
import joblib # Import joblib to load the KNN model and scaler

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QGroupBox, QStatusBar,
                             QLabel, QLineEdit, QProgressBar, QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5 import QtGui
import pyqtgraph as pg
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations

class ActivityState(Enum):
    IDLE = 0
    WRITING = 1
    OTHER = 2

class CollectionMode(Enum):
    SINGLE_ACTIVITY = 0
    COMBINED_SEQUENCE = 4

class DataCollector(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Data Collection Variables ---
        self.current_activity = ActivityState.IDLE
        self.is_recording = False
        self.recorded_samples = 0
        self.target_samples = 0
        self.data_buffer = deque()

        # --- Mode Management Variables ---
        self.collection_mode = CollectionMode.SINGLE_ACTIVITY
        self.combined_session_active = False
        self.mode_4_sequence = [ActivityState.IDLE, ActivityState.WRITING, ActivityState.OTHER]
        self.mode_4_current_step = 0
        self.combined_csv_filename = None

        # --- KNN Classification Variables ---
        self.knn_model = None
        self.scaler_model = None # Untuk menyimpan model scaler (misalnya StandardScaler)
        self.is_classifying = False

        # --- DEFINISIKAN PATH MODEL DAN SCALER DI SINI ---
        # Pastikan file-file ini berada di direktori yang sama dengan script ini,
        # atau berikan path absolut/relatif yang benar.
        self.model_path = 'c:/Users/Dell/Downloads/ARM BAND/model_knn.pkl'
        self.scaler_path = 'c:/Users/Dell/Downloads/ARM BAND/scaler_model.pkl' # Ganti jika nama file scaler Anda berbeda atau jika tidak menggunakan scaler

        # --- Initialize UI first ---
        self.init_ui()

        # --- Board Setup ---
        self.board_shim = None
        self.exg_channels = []
        self.sampling_rate = 250
        self.setup_board()

        self.update_samples_per_sec_label()
        self.set_ui_for_mode(self.collection_mode)

        # --- Load KNN Model & Scaler ---
        self.load_knn_model()


    def init_ui(self):
        self.setWindowTitle('EEG Activity Collector & Classifier')
        self.setGeometry(100, 100, 1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        top_control_layout = QHBoxLayout()

        mode_panel = QGroupBox("Collection Mode")
        mode_layout = QVBoxLayout()
        mode_btn_layout = QHBoxLayout()
        self.mode1_btn = QPushButton("Mode 1 (Idle)")
        self.mode1_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.SINGLE_ACTIVITY, ActivityState.IDLE))
        mode_btn_layout.addWidget(self.mode1_btn)
        self.mode2_btn = QPushButton("Mode 2 (Writing)")
        self.mode2_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.SINGLE_ACTIVITY, ActivityState.WRITING))
        mode_btn_layout.addWidget(self.mode2_btn)
        self.mode3_btn = QPushButton("Mode 3 (Other)")
        self.mode3_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.SINGLE_ACTIVITY, ActivityState.OTHER))
        mode_btn_layout.addWidget(self.mode3_btn)
        mode_btn_layout.addStretch(1)
        mode_layout.addLayout(mode_btn_layout)
        self.mode4_btn = QPushButton("Mode 4 (Combined Sequence)")
        self.mode4_btn.clicked.connect(lambda: self.set_collection_mode(CollectionMode.COMBINED_SEQUENCE))
        mode_layout.addWidget(self.mode4_btn)
        mode_panel.setLayout(mode_layout)
        top_control_layout.addWidget(mode_panel)

        control_panel = QGroupBox("Recording Control")
        control_layout = QVBoxLayout()
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Target Samples:"))
        self.sample_input = QLineEdit("1000")
        self.sample_input.setValidator(QtGui.QIntValidator(100, 100000))
        sample_layout.addWidget(self.sample_input)
        self.samples_per_sec_label = QLabel()
        sample_layout.addWidget(self.samples_per_sec_label)
        control_layout.addLayout(sample_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.progress_bar)
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

        classification_panel = QGroupBox("KNN Classification")
        classification_layout = QVBoxLayout()
        self.classify_start_btn = QPushButton("Start Classification")
        self.classify_start_btn.clicked.connect(self.toggle_classification)
        classification_layout.addWidget(self.classify_start_btn)
        self.prediction_label = QLabel("Prediction: N/A")
        self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        classification_layout.addWidget(self.prediction_label)
        classification_panel.setLayout(classification_layout)
        top_control_layout.addWidget(classification_panel)

        layout.addLayout(top_control_layout)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def load_knn_model(self):
        """Attempts to load the pre-trained KNN model and scaler."""
        # --- Memuat Model KNN ---
        try:
            self.knn_model = joblib.load(self.model_path)
            self.update_status(f"KNN model '{self.model_path}' loaded successfully.")
            self.classify_start_btn.setEnabled(True) # Enable classification button if model loads
        except FileNotFoundError:
            self.update_status(f"Error: KNN model '{self.model_path}' not found. Classification disabled.", is_error=True)
            self.classify_start_btn.setEnabled(False)
            QMessageBox.critical(self, "Model Load Error",
                                 f"KNN model '{self.model_path}' not found.\n"
                                 "Pastikan file .pkl model Anda berada di direktori yang sama dengan script.")
            return # Keluar dari fungsi jika model tidak ditemukan

        # --- Memuat Scaler (Jika Digunakan) ---
        try:
            # Penting: Hanya coba muat scaler jika Anda benar-benar menggunakannya saat melatih model
            self.scaler_model = joblib.load(self.scaler_path)
            self.update_status(f"Scaler model '{self.scaler_path}' loaded successfully.", is_error=False)
        except FileNotFoundError:
            self.scaler_model = None # Set ke None jika scaler tidak ditemukan
            self.update_status(f"Peringatan: Scaler model '{self.scaler_path}' tidak ditemukan. "
                               "Data tidak akan di-scaling sebelum prediksi KNN. "
                               "Pastikan ini sesuai dengan cara model dilatih.", is_error=True)
            QMessageBox.warning(self, "Scaler Load Warning",
                                f"Scaler model '{self.scaler_path}' tidak ditemukan.\n"
                                "Jika model KNN Anda dilatih dengan data yang di-scaling, prediksi mungkin tidak akurat tanpa scaler yang sesuai.")
        except Exception as e:
            self.scaler_model = None
            self.update_status(f"Error loading Scaler model: {str(e)}. Data tidak akan di-scaling.", is_error=True)
            QMessageBox.critical(self, "Scaler Load Error",
                                 f"Terjadi kesalahan saat memuat scaler model:\n{str(e)}\n"
                                 "Prediksi mungkin tidak akurat.")


    def toggle_classification(self):
        """Toggles the real-time classification on/off."""
        if not self.knn_model:
            QMessageBox.warning(self, "Model Tidak Dimuat", "Model KNN belum dimuat. Tidak dapat memulai klasifikasi.")
            return

        if self.is_classifying:
            self.is_classifying = False
            self.classify_start_btn.setText("Mulai Klasifikasi")
            self.prediction_label.setText("Prediksi: N/A")
            self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50;")
            self.update_status("Klasifikasi real-time dihentikan.")
        else:
            self.is_classifying = True
            self.classify_start_btn.setText("Hentikan Klasifikasi")
            self.update_status("Klasifikasi real-time dimulai...")


    def set_ui_for_mode(self, mode):
        self.is_recording = False
        self.start_btn.setText("Mulai Perekaman")
        self.progress_bar.setValue(0)
        self.recorded_samples = 0
        self.target_samples = 0
        self.data_buffer.clear()
        self.combined_session_active = False
        self.mode_4_current_step = 0
        self.combined_csv_filename = None

        if mode == CollectionMode.SINGLE_ACTIVITY:
            self.activity_selection_group.hide()
            self.start_btn.setEnabled(True)
            self.reset_combined_btn.hide()
            self.update_status(f"Mode: {self.current_activity.name}. Siap merekam.")
        elif mode == CollectionMode.COMBINED_SEQUENCE:
            self.activity_selection_group.show()
            self.start_btn.setEnabled(False)
            self.reset_combined_btn.show()
            self.update_status(f"Mode: Gabungan. Pilih aktivitas pertama untuk merekam.")


    def set_collection_mode(self, mode: CollectionMode, initial_activity: ActivityState = None):
        if self.is_recording:
            QMessageBox.warning(self, "Perekaman Aktif", "Mohon hentikan perekaman saat ini sebelum mengganti mode.")
            return
        if self.is_classifying:
            QMessageBox.warning(self, "Klasifikasi Aktif", "Mohon hentikan klasifikasi real-time sebelum mengganti mode.")
            return

        self.collection_mode = mode
        if initial_activity:
            self.current_activity = initial_activity
            self.set_ui_for_mode(self.collection_mode)
        else:
            self.set_ui_for_mode(self.collection_mode)
            self.combined_session_active = True
            self.idle_btn.setEnabled(True)
            self.writing_btn.setEnabled(True)
            self.other_btn.setEnabled(True)


    def set_activity_and_enable_start(self, activity: ActivityState):
        if self.combined_session_active:
            self.current_activity = activity
            self.start_btn.setEnabled(True)
            for btn in [self.idle_btn, self.writing_btn, self.other_btn]:
                if btn.text().upper() != activity.name:
                    btn.setEnabled(False)
                else:
                    btn.setEnabled(True)
            self.start_btn.setText(f"Mulai Perekaman {activity.name}")
            self.update_status(f"Mode: Gabungan. Siap merekam {activity.name}. Tekan Mulai Perekaman.")
        else:
            self.current_activity = activity
            self.update_status(f"Aktivitas: {self.current_activity.name}. Siap merekam.")


    def update_samples_per_sec_label(self):
        self.samples_per_sec_label.setText(f"(≈ {self.sampling_rate} samples/detik)")

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
            self.num_points = self.seconds_to_samples(4)

            self.plot_widget.setBackground('k')
            self.plot_widget.setYRange(-500, 500)
            self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
            self.plot_widget.getPlotItem().getAxis('bottom').setPen(pg.mkPen(color='w'))
            self.plot_widget.getPlotItem().getAxis('left').setPen(pg.mkPen(color='w'))
            self.plot_widget.setLabel('bottom', "Sampel")
            self.plot_widget.setLabel('left', "Amplitudo (uV)")
            self.plot_widget.setTitle("Data EEG Real-time")

            self.plots = []
            for i, ch_idx in enumerate(self.exg_channels):
                color = pg.ColorMap(np.linspace(0, 1, len(self.exg_channels)),
                                    pg.ColorMap.getPalette('spectrum')).map(i)
                pen = pg.mkPen(color=color, width=2)
                plot = self.plot_widget.plot(pen=pen, name=f'Saluran {ch_idx}')
                self.plots.append(plot)

            self.update_status("Board diinisialisasi. Siap merekam/mengklasifikasi.")
        except Exception as e:
            self.update_status(f"Error menginisialisasi board: {str(e)}. Mohon periksa setup MindRove.", is_error=True)
            self.start_btn.setEnabled(False)
            self.classify_start_btn.setEnabled(False)
            QMessageBox.critical(self, "Error Board",
                                 f"Gagal menginisialisasi board MindRove:\n{str(e)}\n"
                                 "Pastikan board terhubung dan driver terinstal.")


    def toggle_recording(self):
        if not self.is_recording:
            try:
                if not self.board_shim or not self.board_shim.is_prepared():
                    QMessageBox.warning(self, "Board Tidak Siap", "Board tidak disiapkan atau terhubung. Tidak dapat memulai perekaman.")
                    self.update_status("Error: Board tidak siap.")
                    return

                if self.is_classifying:
                    QMessageBox.warning(self, "Klasifikasi Aktif", "Mohon hentikan klasifikasi real-time sebelum memulai perekaman.")
                    return

                self.target_samples = int(self.sample_input.text())
                if self.target_samples < 100:
                    raise ValueError("Minimum 100 sampel.")

                self.recorded_samples = 0
                self.data_buffer = deque(maxlen=self.target_samples + self.sampling_rate * 2)
                self.is_recording = True

                base_dir = r'C:\Users\Dell\Downloads\data'
                os.makedirs(base_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_mode = 'w'

                if self.collection_mode == CollectionMode.SINGLE_ACTIVITY:
                    self.csv_filename = os.path.join(base_dir, f"writing_activity_{self.current_activity.name}_{timestamp}.csv")
                elif self.collection_mode == CollectionMode.COMBINED_SEQUENCE:
                    if self.combined_csv_filename is None:
                        self.combined_csv_filename = os.path.join(base_dir, f"writing_activity_GABUNGAN_{timestamp}.csv")
                        self.csv_filename = self.combined_csv_filename
                    else:
                        self.csv_filename = self.combined_csv_filename
                        file_mode = 'a'

                if file_mode == 'w':
                    with open(self.csv_filename, file_mode, newline='') as f:
                        writer = csv.writer(f)
                        header = ['timestamp', 'activity_label'] + [f'ch_{ch}' for ch in self.exg_channels]
                        writer.writerow(header)

                self.start_btn.setText("Hentikan Perekaman")
                self.update_status(f"Perekaman {self.current_activity.name} dimulai ke {os.path.basename(self.csv_filename)}...")
                self.last_write_time = time.time()

            except ValueError as e:
                self.update_status(f"Error: {str(e)}", is_error=True)
                self.is_recording = False
            except Exception as e:
                self.update_status(f"Terjadi kesalahan tak terduga selama setup perekaman: {str(e)}", is_error=True)
                self.is_recording = False
        else:
            self.stop_recording()

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self.write_to_csv()

        if self.collection_mode == CollectionMode.SINGLE_ACTIVITY:
            self.start_btn.setText("Mulai Perekaman")
            self.progress_bar.setValue(100)
            self.update_status(f"Perekaman selesai. Tersimpan {self.recorded_samples} sampel ke {os.path.basename(self.csv_filename)}")
            self.set_ui_for_mode(self.collection_mode)
            QMessageBox.information(self, "Perekaman Selesai",
                                    f"Pengumpulan data selesai untuk '{self.current_activity.name}'.\n"
                                    f"Tersimpan ke: {os.path.basename(self.csv_filename)}")
        elif self.collection_mode == CollectionMode.COMBINED_SEQUENCE:
            self.mode_4_current_step += 1
            if self.mode_4_current_step < len(self.mode_4_sequence):
                next_activity = self.mode_4_sequence[self.mode_4_current_step]
                self.current_activity = next_activity
                self.start_btn.setText(f"Mulai Perekaman {next_activity.name}")
                self.start_btn.setEnabled(True)
                self.progress_bar.setValue(0)
                self.recorded_samples = 0
                self.data_buffer.clear()
                for btn in [self.idle_btn, self.writing_btn, self.other_btn]:
                    if btn.text().upper() != next_activity.name:
                        btn.setEnabled(False)
                    else:
                        btn.setEnabled(True)
                self.update_status(f"{self.mode_4_sequence[self.mode_4_current_step-1].name} selesai. Siap untuk {next_activity.name}. Tersimpan ke {os.path.basename(self.combined_csv_filename)}")
                QMessageBox.information(self, "Aktivitas Selesai",
                                        f"Data '{self.mode_4_sequence[self.mode_4_current_step-1].name}' terkumpul.\n"
                                        f"Klik tombol '{next_activity.name}' lalu 'Mulai Perekaman' untuk aktivitas berikutnya.")
            else:
                self.start_btn.setText("Mulai Perekaman")
                self.progress_bar.setValue(100)
                self.update_status(f"Perekaman gabungan selesai. Semua aktivitas tersimpan ke {os.path.basename(self.combined_csv_filename)}")
                QMessageBox.information(self, "Perekaman Gabungan Selesai",
                                        f"Semua aktivitas gabungan direkam dan disimpan ke:\n{os.path.basename(self.combined_csv_filename)}")
                self.combined_session_active = False
                self.combined_csv_filename = None
                self.set_ui_for_mode(CollectionMode.SINGLE_ACTIVITY)
                self.mode1_btn.setEnabled(True)
                self.mode2_btn.setEnabled(True)
                self.mode3_btn.setEnabled(True)
                self.mode4_btn.setEnabled(True)
        self.update_status()

    def reset_combined_mode(self):
        if self.is_recording:
            self.stop_recording()
            QMessageBox.information(self, "Perekaman Dihentikan", "Perekaman segmen saat ini dihentikan. Mode gabungan akan direset.")

        self.combined_session_active = False
        self.mode_4_current_step = 0
        self.combined_csv_filename = None
        self.data_buffer.clear()
        self.recorded_samples = 0
        self.progress_bar.setValue(0)

        self.set_ui_for_mode(CollectionMode.COMBINED_SEQUENCE)
        self.idle_btn.setEnabled(True)
        self.writing_btn.setEnabled(True)
        self.other_btn.setEnabled(True)
        self.start_btn.setText("Mulai Perekaman")
        self.start_btn.setEnabled(False)

        self.update_status("Mode gabungan telah direset. Pilih aktivitas pertama untuk memulai urutan perekaman baru.")
        QMessageBox.information(self, "Mode Gabungan Direset",
                                "Mode gabungan telah direset. Anda sekarang dapat memulai urutan perekaman gabungan baru.")

    def update(self):
        if not self.board_shim or not self.exg_channels:
            return

        data = self.board_shim.get_current_board_data(self.num_points)
        if data.shape[1] == 0:
            return

        processed_data_for_plot = []
        latest_sample_features = []

        for i, ch_idx in enumerate(self.exg_channels):
            channel_data = data[ch_idx].copy()

            DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(channel_data, self.sampling_rate, 2.0, 60.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(channel_data, self.sampling_rate, 48.0, 52.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)

            self.plots[i].setData(channel_data)
            processed_data_for_plot.append(channel_data)

            if channel_data.size > 0:
                latest_sample_features.append(channel_data[-1])

        # --- Klasifikasi KNN ---
        if self.is_classifying and self.knn_model and len(latest_sample_features) == len(self.exg_channels):
            feature_vector = np.array(latest_sample_features).reshape(1, -1)

            try:
                # --- APPLY SCALER DI SINI JIKA SCALER DIGUNAKAN SAAT PELATIHAN ---
                if self.scaler_model:
                    feature_vector_scaled = self.scaler_model.transform(feature_vector)
                else:
                    feature_vector_scaled = feature_vector # Gunakan data mentah jika tidak ada scaler

                prediction_value = self.knn_model.predict(feature_vector_scaled)[0]
                predicted_activity = ActivityState(prediction_value).name
                self.prediction_label.setText(f"Prediksi: {predicted_activity}")

                if predicted_activity == ActivityState.WRITING.name:
                    self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFD700;")
                elif predicted_activity == ActivityState.IDLE.name:
                    self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00BFFF;")
                else:
                    self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FF6347;")

            except Exception as e:
                self.prediction_label.setText("Error Prediksi!")
                self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: red;")
                self.update_status(f"Error selama prediksi KNN: {str(e)}", is_error=True)

        # --- Perekaman Data ---
        if self.is_recording:
            current_timestamp = time.time()
            num_new_samples_in_data = data.shape[1]

            for i in range(num_new_samples_in_data):
                if self.recorded_samples >= self.target_samples:
                    self.stop_recording()
                    break

                row_data = [processed_channel[i] for processed_channel in processed_data_for_plot]
                csv_row = [current_timestamp, self.current_activity.value] + row_data
                self.data_buffer.append(csv_row)
                self.recorded_samples += 1

            if self.target_samples > 0:
                progress = int((self.recorded_samples / self.target_samples) * 100)
                self.progress_bar.setValue(min(progress, 100))

            if time.time() - getattr(self, 'last_write_time', 0) > 1.0:
                self.write_to_csv()
                self.last_write_time = time.time()

    def write_to_csv(self):
        if not hasattr(self, 'csv_filename') or not self.data_buffer:
            return

        try:
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                while self.data_buffer:
                    writer.writerow(self.data_buffer.popleft())
        except Exception as e:
            self.update_status(f"Error menyimpan data: {str(e)}", is_error=True)

    def update_status(self, message=None, is_error=False):
        if message:
            self.status_bar.showMessage(message)
        else:
            mode_str = "Aktivitas Tunggal" if self.collection_mode == CollectionMode.SINGLE_ACTIVITY else "Urutan Gabungan"
            activity_str = self.current_activity.name if not self.combined_session_active else \
                           f"{self.mode_4_sequence[self.mode_4_current_step].name} (Berikutnya)" if self.mode_4_current_step < len(self.mode_4_sequence) else "Selesai"

            status_parts = [f"Mode: {mode_str}"]
            if self.is_recording:
                status_parts.append(f"Merekam: {activity_str} {self.recorded_samples}/{self.target_samples} sampel")
            elif self.is_classifying:
                status_parts.append("Mengklasifikasi: ON")
            else:
                status_parts.append(f"Siap. Aktivitas: {activity_str}")

            if self.collection_mode == CollectionMode.COMBINED_SEQUENCE and self.combined_session_active:
                status_parts.append(f"Langkah Gabungan: {self.mode_4_current_step + 1}/{len(self.mode_4_sequence)}")

            self.status_bar.showMessage(" | ".join(status_parts))

        if is_error:
            self.status_bar.setStyleSheet("QStatusBar::item {border: none;} QLabel {color: red;}")
        else:
            self.status_bar.setStyleSheet("")

    def closeEvent(self, event):
        if self.is_recording:
            self.stop_recording()
            QMessageBox.information(self, "Perekaman Dihentikan", "Perekaman telah dihentikan dan data disimpan karena aplikasi ditutup.")

        if self.board_shim and self.board_shim.is_prepared():
            try:
                self.board_shim.stop_stream()
                self.board_shim.release_session()
                self.update_status("Sesi Board dilepaskan.")
            except Exception as e:
                self.update_status(f"Error melepaskan sesi board: {str(e)}", is_error=True)

        event.accept()

def main():
    app = QApplication(sys.argv)
    collector = DataCollector()
    collector.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
