# Early Detection of Breast Cancer – A Machine Learning Approach to Classification and Anomaly Detection

## 📌 Project Overview
This project applies various machine learning algorithms to accurately detect and classify breast cancer tumors as benign or malignant using diagnostic features. It demonstrates the end-to-end ML workflow — from data preprocessing to evaluation and bias-variance analysis.

## 📁 Dataset
- **Source**: BreastCancer_Screening.csv
- **Size**: 569 rows × 32 columns
- **Target Variable**: `Diagnosis` — (`M` = Malignant, `B` = Benign)
- **Features**: Radius, Texture, Area, Smoothness, etc. (10 original features and their mean, worst, and SE values)

## 🧠 Models Implemented
- Logistic Regression
- Decision Tree (Tuned)
- Random Forest (Tuned)
- XGBoost (Tuned)
- Neural Network (Keras/TensorFlow)
- Manual Logistic Regression (from scratch)
- K-Means Clustering
- Isolation Forest (Anomaly Detection)

## 📊 Performance Summary

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.982    | 1.000     | 0.952  | 0.976    |
| Decision Tree      | 0.929    | 0.904     | 0.904  | 0.904    |
| Random Forest      | 0.973    | 1.000     | 0.928  | 0.962    |
| XGBoost (Tuned)    | 0.956    | 1.000     | 0.881  | 0.936    |
| Neural Network     | 0.982    | 0.976     | 0.976  | 0.976    |

## 📈 Evaluation Insights
- **Bias-Variance Tradeoff** analyzed via learning curves and cross-validation.
- **Neural Network** showed strong generalization with minimal overfitting (validated by training vs validation plots).
- **Decision Tree** exhibited high variance despite tuning.
- **XGBoost** and **Random Forest** performed robustly after tuning.

## 🧪 Unsupervised Learning
- **K-Means** was used to explore natural groupings using selected features.
- **Isolation Forest** successfully flagged potential malignancies as anomalies, cross-validated with actual labels.

## 💡 Key Takeaways
- Feature scaling and model tuning significantly improved predictive performance.
- Ensemble models (Random Forest & XGBoost) consistently outperformed basic classifiers.
- Neural networks, when regularized, offer high accuracy and generalization.

## 📂 File Structure

```
├── Final Project.ipynb             # Main notebook
├── BreastCancer_Screening.csv      # Dataset
├── README.md                       # Project overview and documentation
```

## 🛠️ Tools & Libraries
- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost
- TensorFlow / Keras

## 👨‍💻 Author
Edidiong Ibokete  
[GitHub Profile](https://github.com/Eddy-bok)

