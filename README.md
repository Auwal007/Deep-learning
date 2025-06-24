# 🧠 Multi-Layer Perceptrons (MLP) with Scikit-Learn and Keras

This repository contains a comprehensive walkthrough of building **Multi-Layer Perceptrons (MLPs)** for both **regression** and **classification** tasks using **Scikit-Learn** and **Keras**. The goal is to demonstrate step-by-step how to implement, train, and evaluate MLPs on real-world datasets such as the **California Housing Dataset**.

---

## 📁 Notebook Overview

The `MLP.ipynb` notebook is structured into two main sections:

### 🔹 1. Regression with MLPs (Scikit-Learn)
- Uses **Scikit-Learn's `MLPRegressor`** to model housing prices.
- Dataset: `fetch_california_housing` (features about California districts and median house values).
- Pipeline includes:
  - **StandardScaler** for feature normalization.
  - **MLPRegressor** with 3 hidden layers of 50 neurons each.
- Model evaluation:
  - **RMSE** (Root Mean Squared Error) on validation set.

### 🔹 2. Classification with MLPs
- Transition from Scikit-Learn to **Keras** for more control and deeper architectures.
- Likely includes:
  - Building classification models using **Keras Sequential API**.
  - Evaluation using accuracy and confusion matrices (based on typical practices).

---

## 🛠️ Libraries Used

- `sklearn.datasets` – for loading the dataset
- `sklearn.neural_network` – for the MLPRegressor
- `sklearn.model_selection` – for training/validation/test splits
- `sklearn.pipeline` – for constructing preprocessing pipelines
- `sklearn.preprocessing` – for feature scaling
- `sklearn.metrics` – for evaluating performance (e.g., `root_mean_squared_error`)
- `keras`, `tensorflow` – expected in later sections (for more advanced MLPs)

---

## 📊 Output Metrics

The regression model is evaluated with:
- `Root Mean Squared Error (RMSE)` — a standard metric for regression.

Classification models are expected to be evaluated with:
- `Accuracy`
- `Confusion Matrix`
- Possibly other metrics like `Precision`, `Recall`, or `F1 Score`.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Auwal007/Deep-learning.git
   cd mlp-regression-classification

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Launch the notebook:

    ```bash
    jupyter notebook MLP.ipynb
    ```
---

## 📌 Key Takeaways

- Scikit-Learn makes it easy to quickly build MLPs for regression tasks.

- For more flexibility (e.g., custom layers, activation functions, loss functions), Keras is a better choice.

- Feature scaling is essential when working with neural networks to ensure convergence.

- Using a pipeline structure with Scikit-Learn improves code modularity and reproducibility.

## 📂 File Structure
```bash
├── MLP.ipynb              # Jupyter Notebook with regression and classification examples
├── README.md              # Project overview and explanation (this file)
├── requirements.txt       # (Optional) List of dependencies
├── my_model.keras         # Saved Keras model file
├── my_checkpoints.weights.h5  # Model weights checkpoint
└── images/                # Folder containing images used in MLP.ipynb
```
---

## 📚 References
- Scikit-Learn Documentation: https://scikit-learn.org/

- TensorFlow Keras Documentation: https://keras.io/

- California Housing Dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

## ✍️ Author

[Muhammad ADAM
](https://x.com/M0hammadAI)

Feel free to reach out or fork the project for improvements or experiments!
