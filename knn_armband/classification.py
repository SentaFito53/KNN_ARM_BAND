import joblib
import numpy as np
from utils import ActivityState # Import ActivityState enum

class EEGClassifier:
    """
    Handles loading KNN models and performing real-time classification of EEG data.
    """
    def __init__(self, model_path: str = 'model_knn.pkl', scaler_path: str = 'scaler_knn.pkl'):
        self.knn_model = None
        self.scaler_model = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.is_ready = False # Flag to indicate if models are successfully loaded

    def load_models(self) -> tuple[bool, str]:
        """
        Loads the pre-trained KNN model and scaler from specified paths.

        Returns:
            tuple[bool, str]: A tuple where the first element is True if both models
                              are loaded successfully, False otherwise.
                              The second element is a status message.
        """
        model_loaded = False
        scaler_loaded = True # Assume true initially, set false if scaler not found

        # Load KNN model
        try:
            self.knn_model = joblib.load(self.model_path)
            model_loaded = True
            model_status = f"KNN model '{self.model_path}' loaded successfully."
        except FileNotFoundError:
            self.knn_model = None
            model_status = f"Error: KNN model '{self.model_path}' not found. Classification disabled."
        except Exception as e:
            self.knn_model = None
            model_status = f"Error loading KNN model: {str(e)}. Classification disabled."

        # Load scaler model
        try:
            self.scaler_model = joblib.load(self.scaler_path)
            scaler_status = f"Scaler model '{self.scaler_path}' loaded successfully."
        except FileNotFoundError:
            self.scaler_model = None
            scaler_loaded = False
            scaler_status = f"Warning: Scaler model '{self.scaler_path}' not found. Data will not be scaled before KNN prediction."
        except Exception as e:
            self.scaler_model = None
            scaler_loaded = False
            scaler_status = f"Error loading Scaler model: {str(e)}. Data will not be scaled."

        # Update overall readiness
        self.is_ready = model_loaded
        
        # Combine status messages
        combined_status = f"{model_status}\n{scaler_status}"
        return self.is_ready, combined_status

    def predict(self, feature_vector: np.ndarray) -> tuple[ActivityState | None, str | None]:
        """
        Performs a real-time prediction using the loaded KNN model.

        Args:
            feature_vector (np.ndarray): A 1D numpy array representing the features
                                         for a single sample.

        Returns:
            tuple[ActivityState | None, str | None]: A tuple containing:
                - ActivityState: The predicted activity state if successful, else None.
                - str: The predicted activity name string if successful, else None.
        """
        if not self.is_ready:
            return None, None # Model not loaded, cannot classify

        # Reshape for single sample prediction
        feature_vector = feature_vector.reshape(1, -1)

        try:
            # Scale features if a scaler is available
            if self.scaler_model:
                feature_vector_scaled = self.scaler_model.transform(feature_vector)
            else:
                feature_vector_scaled = feature_vector

            # Predict the activity
            prediction_value = self.knn_model.predict(feature_vector_scaled)[0]
            predicted_activity = ActivityState(prediction_value)
            return predicted_activity, predicted_activity.name
        except Exception as e:
            print(f"Prediction error: {e}") # For debugging
            return None, "Prediction Error!"

