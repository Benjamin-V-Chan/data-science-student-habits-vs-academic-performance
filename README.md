# data-science-student-habits-vs-academic-performance

## Project Overview
This project investigates how lifestyle habits such as sleep, screen time, study hours, diet, and physical activity affect academic performance. Using the "Student Habits vs Academic Performance" dataset, we build a full data science pipeline including data ingestion, preprocessing, feature engineering, modeling, and evaluation to understand which habits best predict exam scores.

## Folder Structure
```
project-root/
├── data/
│   ├── raw/                            # Original dataset
│   └── processed/                      # Cleaned and feature-engineered datasets
├── outputs/
│   ├── figures/                        # Graphs and visualizations
│   ├── models/                         # Trained model artifacts
│   └── reports/                        # JSON files with performance metrics
├── scripts/                            # Python scripts for each pipeline stage
│   ├── 01_ingest_data.py
│   ├── 02_preprocess_data.py
│   ├── 03_exploratory_analysis.py
│   ├── 04_feature_engineering.py
│   ├── 05_model_training.py
│   └── 06_model_evaluation.py
└── requirements.txt                   # Python dependencies
```

## Usage
1. Setup the Project:

Clone the repository.  
Ensure you have Python installed.  
Install required dependencies using the requirements.txt file:
```bash
pip install -r requirements.txt
```

2. Ingest the raw dataset:
```bash
python scripts/01_ingest_data.py
```

3. Preprocess the dataset:
```bash
python scripts/02_preprocess_data.py
```

4. Run exploratory data analysis (EDA):
```bash
python scripts/03_exploratory_analysis.py
```

5. Create engineered features:
```bash
python scripts/04_feature_engineering.py
```

6. Train and evaluate models:
```bash
python scripts/05_model_training.py
```

7. Generate evaluation visuals:
```bash
python scripts/06_model_evaluation.py
```

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

## Acknowledgments
**dataset name:** Student Habits vs Academic Performance  
**dataset author:** Jayanta Nath  
**dataset source:** https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance