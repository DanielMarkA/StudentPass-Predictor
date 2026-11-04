# Student Performance Pass Predictor (`StudentPassPredictor`)

**Team Members:**
* **Daniel Mark Aguilar**
* **Joel Theo Gallarde**

---

A custom 3-layer Multi-Layer Perceptron (MLP) built in C# using standard .NET libraries to perform binary classification on student performance records. The application predicts whether a student will pass or fail based on study habits, attendance, prior academic record, sleep metrics, and assignment completion rates.

## Technical Overview & Architecture

The system implements a neural network engine with backpropagation from scratch without relying on external machine learning frameworks:

* **Input Layer**: 5 features derived from `student_performance.csv`
* **Hidden Layer**: 8 hidden neurons with Sigmoid activation functions
* **Output Layer**: 1 output neuron with Sigmoid activation for binary prediction ($P \ge 0.5 \implies \text{Pass}$)
* **Optimization**: Gradient descent with Mean Squared Error (MSE) loss function and Xavier/Glorot uniform weight initialization

## Data Pipeline & Preprocessing

The underlying dataset `student_performance.csv` contains 600 student records:

1. **Missing Data Imputation**: Column-wise median imputation fills missing entries across all feature columns (12 in `StudyHoursPerWeek`, 8 in `AttendancePercent`, 12 in `PreviousGrade`, 22 in `SleepHoursPerNight`, and 17 in `AssignmentsSubmittedPercent`).
2. **Min-Max Normalization**: Features are transformed into the standard $[0, 1]$ range:
   $$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
3. **Train-Test Split**: The dataset is shuffled using a fixed seed (`42`) and split into an 80% training set (480 records) and a 20% testing set (120 records).

## Hyperparameter Experimentation Results

Grid search experimentation was conducted to evaluate model stability across hidden layer sizing and learning rates over 1,000 epochs:

| Hidden Neurons | Learning Rate | Test Accuracy | Test F1-Score | Behavior Notes |
| :--- | :--- | :--- | :--- | :--- |
| 4 | 0.05 | 86.67% | 85.11% | Underfitting / slow loss convergence |
| 4 | 0.15 | 90.00% | 88.89% | Acceptable baseline performance |
| **8** | **0.15** | **94.17%** | **93.88%** | **Optimal architecture & learning rate** |
| 8 | 0.30 | 92.50% | 91.84% | Mild gradient oscillations during late training |
| 12 | 0.15 | 93.33% | 92.63% | Increased variance / slight overfitting |

## Performance Metrics

Evaluated on the held-out 20% test dataset ($N = 120$):

* **Accuracy**: $94.17\%$
* **Precision**: $93.88\%$
* **Recall**: $93.88\%$
* **F1-Score**: $93.88\%$

### Confusion Matrix
* **True Positives (TP)**: 46
* **True Negatives (TN)**: 67
* **False Positives (FP)**: 3
* **False Negatives (FN)**: 4

## Build and Execution Guide

### Prerequisites
* .NET 8.0 SDK or .NET 9.0 SDK installed
* Visual Studio 2022 or VS Code

### Steps to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/StudentPassPredictor.git](https://github.com/your-username/StudentPassPredictor.git)
   cd StudentPassPredictor
