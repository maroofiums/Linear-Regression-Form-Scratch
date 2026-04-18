# 📦 Linear Regression From Scratch

A beginner-friendly implementation of **Linear Regression using Gradient Descent**, built completely from scratch using **Python**, **NumPy**, and **Matplotlib**.

This project focuses on understanding how machine learning works internally instead of relying on libraries like Scikit-Learn.

---

## 🚀 Features

* Linear Regression from scratch
* Gradient Descent optimization
* Cost/Loss tracking
* Predict new values
* Plot regression line
* Clean object-oriented code

---

## 📂 Project Structure

```bash
📦Linear-Regression-Form-Scratch
 ┣ 📜main.py
 ┣ 📜README.md
 ┗ 📜RegressionPlot.png
```

---

## 🧠 Mathematical Model

Prediction Formula:

```text
y_pred = w * x + b
```

Where:

* `w` = slope (weight)
* `b` = intercept (bias)

Cost Function:

```text
Cost = (1 / 2m) * sum((y_pred - y)^2)
```

Gradient Descent Updates:

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

---

## ▶️ Installation

```bash
git clone https://github.com/yourusername/Linear-Regression-Form-Scratch.git
cd Linear-Regression-Form-Scratch
pip install numpy matplotlib
```

---

## ▶️ Usage

```bash
python main.py
```

---

## 📊 Example Dataset

```python
X = [1,2,3,4,5]
y = [5,7,9,11,13]
```

Expected learned equation:

```text
y = 2x + 3
```

---

## 📈 Example Output

```bash
Epoch: 0 - Cost: ...
Epoch: 100 - Cost: ...
...
Prediction for x = 6: 15
```

---

## 📉 Visualization

Regression line after training:



![Image](RegressionPlot.png)





---

## 🎯 Learning Goals

This project helps you understand:

* How machine learning models train
* How Gradient Descent minimizes loss
* How regression finds best-fit lines
* Core NumPy operations
* Building ML models without frameworks

---

## 🔥 Future Improvements

* Multiple Linear Regression
* R² Score
* Feature Scaling
* Train/Test Split
* Model Serialization
* Compare with Scikit-Learn

---
