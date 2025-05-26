#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model de predicció d'absències
==============================
Aquest script implementa un model de regressió per predir el nombre d'absències
utilitzant diversos algoritmes més avançats per millorar la sensibilitat.
Els paràmetres del model es llegeixen des d'un fitxer JSON.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Configuració del logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "params_absencies.json")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega les dades des d'un fitxer CSV.

    Args:
        file_path: Ruta al fitxer CSV

    Returns:
        DataFrame amb les dades carregades
    """
    logger.info(f"Carregant dades des de: {file_path}")
    return pd.read_csv(file_path)

def load_parameters(params_path: str) -> Dict[str, Any]:
    """
    Carrega els paràmetres del model des d'un fitxer JSON.
    Si el fitxer no existeix, retorna paràmetres per defecte.

    Args:
        params_path: Ruta al fitxer JSON amb els paràmetres

    Returns:
        Diccionari amb els paràmetres del model
    """
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
            logger.info(f"Paràmetres carregats des de: {params_path}")
            return params
    except FileNotFoundError:
        logger.warning(f"No s'ha trobat el fitxer de paràmetres: {params_path}")
        logger.info("S'utilitzaran paràmetres per defecte")
        # Paràmetres per defecte
        return {
            "regressor__n_estimators": 100,
            "regressor__learning_rate": 0.05,
            "regressor__max_depth": 6,
            "regressor__min_samples_leaf": 5,
            "regressor__subsample": 0.8
        }
    except json.JSONDecodeError:
        logger.error(f"Error de format al fitxer de paràmetres: {params_path}")
        logger.info("S'utilitzaran paràmetres per defecte")
        return {
            "regressor__n_estimators": 100,
            "regressor__learning_rate": 0.05,
            "regressor__max_depth": 6,
            "regressor__min_samples_leaf": 5,
            "regressor__subsample": 0.8
        }

def analyze_absences_distribution(data: pd.DataFrame) -> None:
    """
    Analitza la distribució de la variable absències per entendre millor les dades.

    Args:
        data: DataFrame amb les dades dels estudiants
    """
    absences = data['absences']

    logger.info(f"Estadístiques descriptives de les absències:")
    logger.info(f"Min: {absences.min()}")
    logger.info(f"Max: {absences.max()}")
    logger.info(f"Mitjana: {absences.mean():.2f}")
    logger.info(f"Mediana: {absences.median()}")
    logger.info(f"Desviació estàndard: {absences.std():.2f}")

    # Distribució de valors
    value_counts = absences.value_counts().sort_index()
    logger.info(f"Distribució de valors d'absències:")
    for value, count in value_counts.items():
        logger.info(f"  {value}: {count} estudiants")

def prepare_pipeline(numerical_features: List[str],
                     categorical_features: List[str]) -> ColumnTransformer:
    """
    Prepara el preprocessament de dades amb ColumnTransformer.

    Args:
        numerical_features: Llista de noms de columnes numèriques
        categorical_features: Llista de noms de columnes categòriques

    Returns:
        Pipeline de preprocessament configurat
    """
    logger.info("Preparant pipeline de preprocessament avançat")

    # Preprocessament per a característiques numèriques amb escalat
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  # Afegim escalat per millorar rendiment
    ])

    # Preprocessament per a característiques categòriques amb one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combina tots els preprocessadors en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def train_regression_model(X: pd.DataFrame, y: pd.Series,
                          preprocessor: ColumnTransformer,
                          model_type: str = 'gradient_boosting',
                          param_file: str = DEFAULT_PARAMS_PATH) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Entrena un model de regressió amb validació creuada i cerca en quadrícula.
    Els paràmetres es llegeixen d'un fitxer JSON.

    Args:
        X: Característiques d'entrada
        y: Variable objectiu (absències)
        preprocessor: Pipeline de preprocessament
        model_type: Tipus de model a utilitzar ('decision_tree', 'random_forest', o 'gradient_boosting')
        param_file: Fitxer de paràmetres per al model

    Returns:
        Tuple de (model entrenat, millors hiperparàmetres)
    """
    logger.info(f"Entrenant model de regressió {model_type} per predir absències")

    # Carreguem els paràmetres des del fitxer o utilitzem els per defecte
    model_params = load_parameters(param_file)
    logger.info(f"S'utilitzaran els següents paràmetres: {model_params}")

    # Selecció del model segons el tipus especificat
    if model_type == 'decision_tree':
        regressor = DecisionTreeRegressor(random_state=RANDOM_STATE)
    elif model_type == 'random_forest':
        regressor = RandomForestRegressor(random_state=RANDOM_STATE)
    elif model_type == 'gradient_boosting':
        # Creem el model amb els paràmetres carregats
        # Primer filtrem només els paràmetres que comencen per 'regressor__'
        gb_params = {k.replace('regressor__', ''): v for k, v in model_params.items()
                    if k.startswith('regressor__')}
        regressor = GradientBoostingRegressor(random_state=RANDOM_STATE, **gb_params)
    else:
        raise ValueError(f"Tipus de model no suportat: {model_type}")

    # Crea el pipeline complet amb selecció de característiques
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))),
        ('regressor', regressor)
    ])

    # Entrenament del model directament amb els paràmetres carregats
    logger.info("Entrenant el model amb els paràmetres especificats...")
    model.fit(X, y)

    # Retornem el model i els paràmetres utilitzats
    return model, model_params

def evaluate_regression_model(model: Pipeline, X_test: pd.DataFrame,
                             y_test: pd.Series) -> Dict[str, float]:
    """
    Avalua un model de regressió i torna les mètriques.

    Args:
        model: Model entrenat
        X_test: Dades de prova
        y_test: Etiquetes reals

    Returns:
        Diccionari amb les mètriques d'avaluació
    """
    logger.info("Avaluant model de regressió")

    # Prediccions
    y_pred = model.predict(X_test)

    # Càlcul de mètriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculem percentatge d'errors absoluts per rang
    abs_errors = np.abs(y_test - y_pred)
    error_under_1 = np.mean(abs_errors < 1) * 100
    error_under_2 = np.mean(abs_errors < 2) * 100
    error_under_5 = np.mean(abs_errors < 5) * 100

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'error_under_1': error_under_1,
        'error_under_2': error_under_2,
        'error_under_5': error_under_5
    }

    logger.info(f"Mètriques de regressió:")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"Prediccions amb error < 1: {error_under_1:.2f}%")
    logger.info(f"Prediccions amb error < 2: {error_under_2:.2f}%")
    logger.info(f"Prediccions amb error < 5: {error_under_5:.2f}%")

    return metrics

def save_model_and_params(model: Pipeline, params: Dict[str, Any],
                         model_path: str, params_path: str) -> None:
    """
    Guarda el model i els seus millors paràmetres en fitxers.

    Args:
        model: Model entrenat a guardar
        params: Millors paràmetres del model
        model_path: Ruta per guardar el model
        params_path: Ruta per guardar els paràmetres
    """
    # Guarda el model
    joblib.dump(model, model_path)
    logger.info(f"Model guardat a: {model_path}")

    # Guarda els paràmetres
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Paràmetres guardats a: {params_path}")

def main() -> None:
    """Funció principal que coordina tot el flux de treball."""

    # Configuració dels arguments de línia d'ordres
    parser = argparse.ArgumentParser(
        description='Model de predicció d\'absències'
    )
    parser.add_argument('--data', type=str, default='../../portuguese_hs_students.csv',
                        help='Ruta al fitxer CSV amb les dades')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directori on es guardaran els resultats')
    parser.add_argument('--model-type', type=str,
                        choices=['decision_tree', 'random_forest', 'gradient_boosting'],
                        default='gradient_boosting',
                        help='Tipus de model a entrenar')
    parser.add_argument('--params', type=str, default=DEFAULT_PARAMS_PATH,
                        help='Ruta al fitxer JSON amb paràmetres personalitzats')

    args = parser.parse_args()

    # Carrega les dades
    data = load_data(args.data)
    logger.info(f"Dades carregades: {data.shape[0]} files, {data.shape[1]} columnes")

    # Analitzem la distribució de la variable absències
    analyze_absences_distribution(data)

    # Identificació de característiques numèriques i categòriques
    # Exclou 'absences' que serà la variable objectiu
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'absences']

    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    logger.info(f"Característiques numèriques: {len(numerical_cols)}")
    logger.info(f"Característiques categòriques: {len(categorical_cols)}")

    # Prepara el preprocessament
    preprocessor = prepare_pipeline(numerical_cols, categorical_cols)

    # Prepara les dades
    X = data.drop(columns=['absences'])
    y = data['absences']

    # Dividim les dades en conjunts d'entrenament i test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    logger.info(f"Dades dividides: {X_train.shape[0]} mostres d'entrenament, {X_test.shape[0]} mostres de test")

    # Entrena el model amb els paràmetres del fitxer
    model, params = train_regression_model(
        X_train, y_train, preprocessor, model_type=args.model_type, param_file=args.params
    )

    # Avalua el model
    metrics = evaluate_regression_model(model, X_test, y_test)

    # Guarda el model i els paràmetres
    model_path = os.path.join(args.output_dir, "decision_tree_absences.joblib")
    params_path = os.path.join(args.output_dir, "params_absencies.json")
    save_model_and_params(model, params, model_path, params_path)

    # Mostra un resum final
    logger.info("El model d'absències ha estat entrenat i avaluat correctament.")
    print("\n" + "="*50)
    print("Model de predicció d'absències")
    print("="*50)
    print(f"Model i resultats guardats a: {args.output_dir}")
    print("\nUsa la següent comanda per veure les opcions:")
    print(f"python {os.path.basename(__file__)} --help")
    print("\nFitxers generats:")
    print(f"- {os.path.basename(model_path)} - Model de regressió per a absències")
    print(f"- {os.path.basename(params_path)} - Paràmetres del model")
    print("="*50)


if __name__ == "__main__":
    main()
