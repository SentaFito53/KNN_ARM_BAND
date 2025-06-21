# 🧠 Writing Activity Classifier using K-NN

This project provides a Python-based GUI for **data collection**, **labeling**, and **classification** of writing activities using the **K-Nearest Neighbors (K-NN)** algorithm.

## 🚀 Installation Guide

1. **Clone the repository**

```bash
git clone "https://github.com/SentaFito53/KNN_ARM_BAND.git"
cd KNN_ARM_BAND/knn_armband
```

2. **Install the required dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the main application**

```bash
python3 main.py
```

## 🖥️ Graphical User Interface

The GUI includes features for collecting and labeling sensor data, and for real-time classification using the trained K-NN model.

![GUI Screenshot](https://github.com/SentaFito53/KNN_ARM_BAND/blob/main/assets/gui_example.png)


## ⚠️ Notes

* Make sure the trained **model** and **scaler** files are available in the project directory (e.g., `model_knn.pkl`, `scaler.pkl`).
* To improve classification accuracy, increase the **dataset size** by collecting more labeled data samples.
* Train your custom dataset using collab [TRAIN KNN.ipynb](https://github.com/SentaFito53/KNN_ARM_BAND/blob/main/TRAIN_KNN.ipynb)


## 📦 Key Dependencies

* PyQt5
* pyqtgraph
* numpy
* joblib
* mindrove armband

> All required libraries will be installed automatically using the `requirements.txt` file.

---

Developed by [SentaFito53](https://github.com/SentaFito53) — feel free to use, modify, and extend for research and educational purposes.
