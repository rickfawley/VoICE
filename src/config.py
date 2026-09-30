# src/config.py

from pathlib import Path 

MY_START_POINT = 1

BASE_COLOURS = [
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
    "#FFFF33",
    "#A65628",
    "#F781BF",
]

MODEL_LEVEL_PUBLICATION_MODES = [
    "Unweighted k-means",
    "Ranked k-means",
    "Ranked + weighted SHARK",
]

DATASET_INFORMATION = {

    "Iris": {
        "default_k": 3,
        "ground_truth": "species",
        "feature_exclusions": [],
        "categorical_encoding": "none",
        "missing_strategy": "error",
        "url": "https://archive.ics.uci.edu/dataset/53/iris",
        "description": "Classic flower measurement dataset.",
        "local_path": Path("data/iris/iris.csv"),
        "sep": ",",
        "include_in_experiments" : True,
    },

    "Wine": {
        "default_k": 3,
        "ground_truth": "class",
        "feature_exclusions": [],
        "categorical_encoding": "none",
        "missing_strategy": "error",
        "url": "https://archive.ics.uci.edu/ml/datasets/wine",
        "description": "Chemical analysis of wines.",
        "local_path": Path("data/wine/wine.csv"),
        "sep": ",",
        "include_in_experiments" : True,
    },

    "Palmer Penguins": {
        "default_k": 3,
        "ground_truth": "species",
        "feature_exclusions": [
            "island",
            "sex",
            "year",
        ],
        "categorical_encoding": "none",
        "missing_strategy": "drop_rows",
        "local_path": Path("data/penguins/penguins.csv"),
        "sep": ",",
        "url": "https://archive.ics.uci.edu/dataset/690/palmer+penguins",
        "include_in_experiments": True,
    },

    "Breast Cancer": {
        "default_k": 2,
        "ground_truth": "diagnosis",
        "feature_exclusions": [
            "id",
        ],
        "categorical_encoding": "none",
        "missing_strategy": "error",
        "local_path": Path("data/breast_cancer/breast_cancer.csv"),
        "sep": ",",
        "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "include_in_experiments" : True,
    },

    "Wholesale Customers": {
        "include_in_experiments": True,
        "default_k": 3,
        "ground_truth": "Region",
        "feature_exclusions": [
            "Channel",
            "Region",
        ],
        "categorical_encoding": "none",
        "missing_strategy": "error",
        "local_path": Path("data/wholesale/wholesale.csv"),
        "sep": ",",
        "url": "https://archive.ics.uci.edu/dataset/292/wholesale+customers",
    },

    "Diabetes": {
        "default_k": 3,
        "ground_truth": "Diabetes_012",
        "feature_exclusions": [
            "BMI",
        ],
        "categorical_encoding": "none",
        "missing_strategy": "error",
        "local_path": Path("data/diabetes/diabetes.csv"),
        "sep": ",",
        "url": "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators",
        "include_in_experiments": True,
    },

    "Obesity": {
        "default_k": 7,
        "ground_truth": "NObeyesdad",
        "feature_exclusions": [
            "Weight",
        ],
        "categorical_encoding": "onehot",
        "missing_strategy": "error",
        "local_path": Path("data/obesity/obesity.csv"),
        "sep": ",",
        "url": "https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition",
        "include_in_experiments": True,
    },

    "German Credit": {
        "default_k": 2,
        "ground_truth": "default",
        "feature_exclusions": [],
        "categorical_encoding": "onehot",
        "missing_strategy": "error",
        "url": "https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)",
        "description": "Financial and demographic attributes used to assess credit risk.",
        "local_path": Path("data/german_credit/german_credit.csv"),
        "sep": ",",
        "include_in_experiments" : True,
    },

    "Heart Failure": {
        "default_k": 2,
        "ground_truth": "DEATH_EVENT",
        "feature_exclusions": [],
        "categorical_encoding": "none",
        "missing_strategy": "error",
        "url": "https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records",
        "description": "linical health indicators associated with heart failure patient outcomes.",
        "local_path": Path("data/heart_failure/heart_failure_clinical_records.csv"),
        "sep": ",",
        "include_in_experiments" : True,
    },

    "Student Performance": {
        "default_k": 5,
        "ground_truth": "derived_pseudo_classifier",
        "feature_exclusions": [
            "G1",
            "G2",
            "G3",
        ],
        "categorical_encoding": "onehot",
        "missing_strategy": "error",
        "url": "https://archive.ics.uci.edu/ml/datasets/Student+Performance",
        "description": "xAcademic, social, and demographic factors related to student achievement.",
        "local_path": Path("data/student_performance/student-por.csv"),
        "sep": ",",
        "include_in_experiments" : True,
    },
}

