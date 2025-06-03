#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model de predicció d'aprovats
=============================
Aquest script implementa un model de classificació per predir si un alumne aprovarà
(G3>=10) utilitzant diversos algoritmes de machine learning.
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import SMOTE for handling imbalanced classes
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline



# Constants
RANDOM_STATE = 42
DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "params_aprovat.json")


def load_data(file_path: str) -> pd.DataFrame:
    """Carrega les dades des d'un fitxer CSV."""
    return pd.read_csv(file_path)


def load_parameters(params_path: str) -> Dict[str, Any]:
    """Carrega els paràmetres del model des d'un fitxer JSON."""
    with open(params_path, 'r') as f:
        params = json.load(f)
        return params


def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepara les dades per a l'entrenament del model."""
    # Crear la variable objectiu: aprovat (1) / suspès (0)
    data['aprovat'] = (data['G3'] >= 10).astype(int)

    # No utilitzar G3 com a característica
    X = data.drop(columns=['aprovat', 'G3'])
    y = data['aprovat']

    # Identificar característiques categòriques
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    return X, y, categorical_features


def prepare_pipeline(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Crea un pipeline de preprocessament per a les dades."""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    return preprocessor


def train_classification_model(X: pd.DataFrame, y: pd.Series,
                               categorical_features: List[str],
                               param_file: str = DEFAULT_PARAMS_PATH) -> Tuple[ImbPipeline, Dict[str, Any]]:
    """Entrena el model de classificació per predir si un alumne aprovarà."""
    # Identificar les característiques numèriques
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # Crear el preprocessador
    preprocessor = prepare_pipeline(numerical_features, categorical_features)
    
    # Carregar paràmetres del model
    model_params = load_parameters(param_file)
    classifier_params = {k.replace('classifier__', ''): v
                        for k, v in model_params.items() if k.startswith('classifier__')}
    
    # Crear pipeline amb SMOTE per gestionar el desbalanceig
    model = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', GradientBoostingClassifier(random_state=RANDOM_STATE, **classifier_params))
    ])
    
    model.fit(X, y)

    return model, model_params

def main():
    """Funció principal per entrenar i avaluar el model de predicció d'aprovats."""
    parser = argparse.ArgumentParser(
        description="Entrena un model de predicció d'aprovats per a estudiants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", type=str, default='../../portuguese_hs_students.csv',
                        help="Ruta al fitxer CSV amb les dades")
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__),
                        help="Directori on desar el model i resultats")
    parser.add_argument("--params", type=str, default=DEFAULT_PARAMS_PATH,
                        help="Ruta al fitxer JSON amb els paràmetres del model")

    args = parser.parse_args()

    # Assegurar-se que el directori de sortida existeix
    os.makedirs(args.output_dir, exist_ok=True)

    # Configura un gestor de logs addicional per desar els logs en un fitxer
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "training_log.txt"), mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        # Carregar dades
    data = load_data(args.data)
        # Preparar dades
    X, y, categorical_features = prepare_data(data)

        # Dividir dades en conjunts d'entrenament i test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
        # Entrenar model
    model, _ = train_classification_model(
        X_train, y_train, categorical_features, args.params
    )
        # Desar el model
    model_path = os.path.join(args.output_dir, "model_aprovat.joblib")
    joblib.dump(model, model_path)

if __name__ == "__main__":
    main()
