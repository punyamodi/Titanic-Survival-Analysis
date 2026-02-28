# Titanic Survival Prediction: Advanced Machine Learning Model

![Titanic](https://img.shields.io/badge/Machine-Learning-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/Library-XGBoost-red)

An improved, robust approach to predicting passenger survival on the Titanic. This project moves beyond simple baseline models to implement advanced feature engineering and hyperparameter tuning for higher predictive accuracy.

## 🚀 Key Improvements

Our model significantly improves upon basic implementations through:

- **Advanced Feature Engineering**:
  - **Title Extraction**: Parsing social titles (Mr, Mrs, Miss, Master, Rare) from names to infer social status.
  - **Missing Value Imputation**: Strategically filling missing ages using median values grouped by Title and Pclass.
  - **Categorical Feature Binning**: Handling sparse categories and binary flags for cabinet availability.
  - **Family Relationship Features**: Engineering `FamilySize` and `IsAlone` variables.
- **Robust Modeling Pipeline**:
  - Compared multiple algorithms including **Random Forest**, **XGBoost**, and **LightGBM**.
  - Utilized **GridSearchCV** for hyperparameter optimization.
  - Implemented **5-Fold Cross-Validation** for more reliable performance estimates.
- **Production-Ready Codebase**:
  - Refactored from a single Jupyter notebook to a modular Python project structure.
  - Separated data preprocessing, model training, and evaluation logic.

## 📁 Project Structure

```text
.
├── data/                  # Titanic Dataset (Train, Test, and Improved Submission)
├── models/                # Serialized trained models (.joblib)
├── notebooks/             # Exploratory Data Analysis notebooks
├── src/                   # Source code
│   ├── preprocessor.py    # Data cleaning and feature engineering
│   └── __init__.py
├── main.py                # Main entry point for training and evaluation
├── requirements.txt       # Project dependencies
└── README.md
```

## 🛠️ Getting Started

### Prerequisites

Clone the repository and install dependencies:

```bash
git clone https://github.com/punyamodi/ML-Project---Titanic-Survivability.git
cd ML-Project---Titanic-Survivability
pip install -r requirements.txt
```

### Usage

To train the model and generate a new submission file:

```bash
python main.py
```

Results and a serialized model will be saved in the `data/` and `models/` directories, respectively.

## 📊 Performance & Results

The current best model (Random Forest) achieves:

- **Validation Accuracy**: ~83.5%
- **F1-Score**: High performance on both survival and non-survival classes.

| Feature Impact | Categorization |
| -------------- | :------------: |
| Sex            |      High      |
| Pclass         |      High      |
| Title          |     Medium     |
| Fare           |     Medium     |

## 🤝 Contributing

Feel free to open an issue or submit a pull request if you have ideas for further improvements!

## 📜 License

Created by [Punya Modi](https://github.com/punyamodi).
