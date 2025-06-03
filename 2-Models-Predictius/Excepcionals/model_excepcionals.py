#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model de predicció d'alumnes excepcionals
=========================================
Aquest script implementa un model de classificació per predir si un alumne és excepcional
(G3>=18) utilitzant algoritmes de machine learning i tractament per a dades desbalancejades.
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
    """Carrega les dades des d'un fitxer CSV."""
    return pd.read_csv(file_path)


def load_parameters(params_path: str) -> Dict[str, Any]:
    """Carrega els paràmetres del model des d'un fitxer JSON."""
    with open(params_path, 'r') as f:
        params = json.load(f)
        return params


def prepare_data(data: pd.DataFrame, threshold: int) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepara les dades per a l'entrenament del model d'alumnes excepcionals.
    
    Args:
        data: DataFrame amb les dades
        threshold: Llindar a partir del qual es considera un alumne excepcional
        
    Returns:
        X: Característiques
        y: Variable objectiu (excepcional/no excepcional)
        categorical_features: Llista de característiques categòriques
    """
    # Crear la variable objectiu: excepcional (1) / no excepcional (0)
    data['excepcional'] = (data['G3'] >= threshold).astype(int)
    
    # No utilitzar G3 com a característica
    X = data.drop(columns=['excepcional', 'G3'])
    y = data['excepcional']
    
    # Identificar característiques categòriques
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
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
    """
    Entrena el model de classificació per predir alumnes excepcionals.
    
    Args:
        X: DataFrame amb les característiques
        y: Series amb la variable objectiu
        categorical_features: Llista de característiques categòriques
        param_file: Ruta al fitxer de paràmetres
        
    Returns:
        model: Model entrenat
        model_params: Paràmetres utilitzats
    """
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
    
    logger.info("Entrenant el model amb els paràmetres especificats...")
    model.fit(X, y)
    
    return model, model_params


def main():
    """Funció principal per entrenar i avaluar el model d'alumnes excepcionals."""
    parser = argparse.ArgumentParser(
        description="Entrena un model de predicció d'alumnes excepcionals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", type=str, default='../../portuguese_hs_students.csv',
                        help="Ruta al fitxer CSV amb les dades")
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__),
                        help="Directori on desar el model i resultats")
    parser.add_argument("--params", type=str, default=DEFAULT_PARAMS_PATH,
                        help="Ruta al fitxer JSON amb els paràmetres del model")
    parser.add_argument("--model-type", type=str, default='gradient_boosting',
                        help="Tipus de model (sempre serà gradient boosting)")

    args = parser.parse_args()

    # Assegurar-se que el directori de sortida existeix
    os.makedirs(args.output_dir, exist_ok=True)

    # Configura un gestor de logs addicional per desar els logs en un fitxer
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "training_log.txt"), mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    try:
        logger.info("Iniciant el procés d'entrenament del model d'alumnes excepcionals...")
        
        # Carregar dades i paràmetres
        data = load_data(args.data)
        params = load_parameters(args.params)
        
        # Determinar el llindar per alumnes excepcionals
        raw_threshold = params.get('threshold', 18)
        try:
            threshold = int(raw_threshold)
            threshold = min(threshold, 18)  # Assegurar-se que no sigui superior a 18
        except (TypeError, ValueError):
            logger.warning(f"Valor de llindar invàlid '{raw_threshold}', utilitzant 18 per defecte")
            threshold = 18
        
        logger.info(f"Utilitzant llindar de {threshold} per determinar alumnes excepcionals")
        
        # Preparar dades
        X, y, categorical_features = prepare_data(data, threshold)
        
        # Mostrar distribució de classes
        class_counts = y.value_counts()
        total = len(y)
        logger.info(f"Total d'alumnes: {total}")
        logger.info(f"Alumnes excepcionals: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/total*100:.2f}%)")
        logger.info(f"Alumnes no excepcionals: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/total*100:.2f}%)")
        
        # Dividir dades en conjunts d'entrenament i test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        # Entrenar model
        model, _ = train_classification_model(
            X_train, y_train, categorical_features, args.params
        )
        
        # Avaluar el model
        y_pred = model.predict(X_test)
        
        # Calcular mètriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Rendiment del model:")
        logger.info(f"Exactitud (accuracy): {accuracy:.4f}")
        logger.info(f"Precisió (precision): {precision:.4f}")
        logger.info(f"Sensibilitat (recall): {recall:.4f}")
        logger.info(f"Puntuació F1 (F1 score): {f1:.4f}")
        
        # Desar el model
        model_path = os.path.join(args.output_dir, "model_excepcional.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model desat a {model_path}")
        
    except Exception as e:
        logger.exception(f"Error durant l'execució: {e}")
        with open(os.path.join(args.output_dir, "error_log.txt"), "w") as f:
            f.write(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
