# Cyclic Coordinate Descent for L1-Regularized Logistic Regression

## 📊 Overview

Implementation of Cyclic Coordinate Descent (CCD) algorithm for parameter estimation in logistic regression with L1 penalty (lasso), with comparisons against standard logistic regression. Additional possibility for ElasticNet regularization. Validated on synthetic and real-world datasets from OpenML. We highly encourage to check the [report](https://github.com/dxzielinski/aml-ccd/blob/main/AML_Project1_Report.pdf) for the detailed explanation.

## Key Features

• **From-scratch CCD implementation** with L1 regularization  
• **Synthetic data generator** with configurable parameters (n, p, d, g)  
• **Real-world evaluations** on Spambase, Blood Transfusion, Cardiac Arrhythmia, and KC2 datasets  
• **Automatic λ selection** via validation metrics (F1, Balanced Accuracy, etc.)  
• **Coefficient path visualization** for sparsity analysis  

## 📌 Benchmark Results

| Dataset               | LogRegCCD (F1) | LogRegCCD (BA) | Standard LR (F1) | Standard LR (BA) |
|-----------------------|---------------:|---------------:|-----------------:|-----------------:|
| Spambase              | 0.87           | 0.73           | 0.88             | 0.77             |
| Blood Transfusion     | 0.34           | 0.33           | 0.52             | 0.54             |
| Cardiac Arrhythmia    | 0.53           | 0.32           | 0.57             | 0.57             |
| KC2 Software Defect   | 0.62           | 0.66           | 0.71             | 0.66             |

## Key Findings

**Correctness Verification**  
✓ Matches standard logistic regression at λ=0  
✓ Correct coefficient shrinkage patterns  
✓ Identical performance to scikit-learn in controlled tests  

**Synthetic Data Insights**  
✓ Robust to high dimensionality (tested to d=200)  
✓ Handles feature correlation (g=0 to g≈1)

## 👨‍💻 Contributors

- [Emil Łasocha](https://github.com/emilook86)
- [Dominik Zieliński](https://github.com/dxzielinski)
- [Małgorzata Kurcjusz-Gzowska](https://github.com/MKurcjusz)
