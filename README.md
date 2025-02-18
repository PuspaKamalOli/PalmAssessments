# Titanic Survival Prediction

## Overview
This project implements and compares three supervised machine learning models to predict the survival of Titanic passengers. The models used are:
- **Logistic Regression**
- **Random Forest**
- **Neural Network (TensorFlow/Keras)**

The dataset used for training and evaluation is `titanic_train.csv`.

## Project Structure
```
PalmAssessments/
│── assessments/             # Virtual environment (optional)
│── titanic_train.csv        # Dataset file
│── LogisticRegression.ipynb # Logistic Regression implementation
│── RandomForest.ipynb       # Random Forest implementation
│── NeuralNetwork.ipynb      # Neural Network implementation (TensorFlow/Keras)
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
```

## Dependencies
Before running the notebooks, ensure you have the required dependencies installed. The `requirements.txt` file includes the following:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`
- `tensorflow`

### Installation
To set up the environment, follow these steps:
1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv assessments
   ```
2. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source assessments/bin/activate
     ```
   - On Windows:
     ```bash
     assessments\Scripts\activate
     ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
Each model is implemented in a separate Jupyter Notebook. To run them:
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open and execute any of the following notebooks:
   - `LogisticRegression.ipynb`
   - `RandomForest.ipynb`
   - `NeuralNetwork.ipynb`

## Model Performance Summary
The models were evaluated using accuracy, precision, recall, and F1-score. Below is a summary of their performance:

| Model                | Accuracy | Precision (Class 0) | Precision (Class 1) | Recall (Class 0) | Recall (Class 1) | F1-Score (Class 0) | F1-Score (Class 1) |
|----------------------|----------|---------------------|---------------------|------------------|------------------|--------------------|--------------------|
| **Logistic Regression** | 69%      | 0.67                | 0.74                | 0.87             | 0.46             | 0.75               | 0.57               |
| **Random Forest**       | 70%      | 0.70                | 0.71                | 0.82             | 0.56             | 0.75               | 0.63               |
| **Neural Network**      | 71%      | 0.68                | 0.79                | 0.90             | 0.47             | 0.77               | 0.59               |

## Conclusion
- The **Neural Network model achieved the highest accuracy (71%)** and the best precision for predicting survivors.
- The **Random Forest model provided a more balanced precision-recall trade-off**, making it a strong candidate for general classification.
- The **Logistic Regression model**, while simple and interpretable, had the lowest recall for survivors, making it less effective in this case.
- However , among three models RFC(bagging) is preferred which excels in preventing overfitting specially when we have very few data(length of data is 891) whereas for neural network we oftenly require a lot of data or it may lead to overfitting problem. 

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



