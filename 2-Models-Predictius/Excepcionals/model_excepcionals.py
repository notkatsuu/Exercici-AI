#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model de predicció d'alumnes excepcionals
=========================================
Aquest script implementa un model de classificació per predir si un alumne és excepcional
(G3>=18) utilitzant els mateixos passos que el model d'aprovats.
Els paràmetres del model es llegeixen des d'un fitxer JSON.
"""
import argparse
import json
import logging
import os
from typing import Dict, List, Tuple, Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configuració del logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "params_excepcionals.json")


def load_data(file_path: str) -> pd.DataFrame:
    logger.info(f"Carregant dades des de: {file_path}")
    return pd.read_csv(file_path)


def load_parameters(params_path: str) -> Dict[str, Any]:
    default_params = {
        "classifier__n_estimators": 100,
        "classifier__learning_rate": 0.1,
        "classifier__max_depth": 5,
        "classifier__min_samples_leaf": 5,
        "classifier__subsample": 0.8,
        "threshold": 17
    }
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
            logger.info(f"Paràmetres carregats des de: {params_path}")
            return params
    except Exception:
        logger.warning(f"No s'ha pogut carregar {params_path}, usant paràmetres per defecte.")
        return default_params


def analyze_class_distribution(data: pd.DataFrame, threshold: int) -> None:
    logger.info(f"Analitzant distribució d'alumnes excepcionals (G3>={threshold})...")
    data['excepcional'] = (data['G3'] >= threshold).astype(int)
    counts = data['excepcional'].value_counts()
    total = len(data)
    logger.info(f"Total alumnes: {total}")
    logger.info(f"Excepcionals: {counts.get(1,0)} ({counts.get(1,0)/total*100:.2f}%)")
    logger.info(f"No excepcionals: {counts.get(0,0)} ({counts.get(0,0)/total*100:.2f}%)")


def prepare_data(data: pd.DataFrame, threshold: int) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    data['excepcional'] = (data['G3'] >= threshold).astype(int)
    X = data.drop(columns=['excepcional','G3'])
    y = data['excepcional']
    categorical = X.select_dtypes(include=['object','category']).columns.tolist()
    return X, y, categorical


def create_pipeline(categorical: List[str], X: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in X.columns if c not in categorical]
    num_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[('num', num_transform, numeric), ('cat', cat_transform, categorical)],
        remainder='passthrough'
    )


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, categorical: List[str], params: Dict[str, Any], output_dir: str):
    preproc = create_pipeline(categorical, X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preproc),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
    for p,v in params.items():
        if p.startswith('classifier__'):
            n = p.split('__')[1]
            setattr(pipeline.named_steps['classifier'], n, v)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    # log metrics
    logger.info(f"Accuracy: {accuracy_score(y_test,y_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_test,y_pred):.4f}")
    logger.info(f"Recall: {recall_score(y_test,y_pred):.4f}")
    logger.info(f"F1: {f1_score(y_test,y_pred):.4f}")
    # save model
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(output_dir,'decision_tree_excepcionals.joblib'))


def main():
    parser = argparse.ArgumentParser(description="Model per alumnes excepcionals (G3>=18)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", default='../../portuguese_hs_students.csv')
    parser.add_argument("--output-dir", default=os.path.dirname(__file__))
    parser.add_argument("--params", default=DEFAULT_PARAMS_PATH)
    # Model type argument is kept for backward compatibility but will always use gradient_boosting
    parser.add_argument("--model-type", type=str, default='gradient_boosting',
                        help="Tipus de model (sempre serà gradient boosting)")
    args = parser.parse_args()
    params = load_parameters(args.params)
    # Ensure threshold is integer
    raw_threshold = params.get('threshold', 19)
    try:
        orig_threshold = int(raw_threshold)
    except (TypeError, ValueError):
        logger.warning(f"Invalid threshold value '{raw_threshold}', defaulting to 19")
        orig_threshold = 19
    if orig_threshold > 18:
        logger.warning(f"Threshold {orig_threshold} too high; capping to 18")
    threshold = min(orig_threshold, 18)
    data = load_data(args.data)
    analyze_class_distribution(data, threshold)
    X, y, cat = prepare_data(data, threshold)
    train_and_evaluate(X, y, cat, params, args.output_dir)

if __name__ == "__main__":
    main()
