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
from typing import Dict, List, Tuple, Any

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
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"El fitxer de dades no s'ha trobat a: {file_path}")
        raise
    except Exception as e:
        logger.error(f"S'ha produït un error carregant les dades: {e}")
        raise


def load_parameters(params_path: str) -> Dict[str, Any]:
    """
    Carrega els paràmetres del model des d'un fitxer JSON.
    Si el fitxer no existeix, retorna paràmetres per defecte.

    Args:
        params_path: Ruta al fitxer JSON amb els paràmetres

    Returns:
        Diccionari amb els paràmetres del model
    """
    default_params = {
        "regressor__n_estimators": 100,
        "regressor__learning_rate": 0.05,
        "regressor__max_depth": 6,
        "regressor__min_samples_leaf": 5,
        "regressor__subsample": 0.8
    }
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
            logger.info(f"Paràmetres carregats des de: {params_path}")
            return params
    except FileNotFoundError:
        logger.warning(f"No s'ha trobat el fitxer de paràmetres: {params_path}. S'utilitzaran paràmetres per defecte.")
        return default_params
    except json.JSONDecodeError:
        logger.error(f"Error de format al fitxer de paràmetres: {params_path}. S'utilitzaran paràmetres per defecte.")
        return default_params
    except Exception as e:
        logger.error(f"S'ha produït un error carregant els paràmetres: {e}. S'utilitzaran paràmetres per defecte.")
        return default_params


def analyze_absences_distribution(data: pd.DataFrame) -> None:
    """
    Analitza la distribució de la variable absències per entendre millor les dades.

    Args:
        data: DataFrame amb les dades dels estudiants
    """
    if 'absences' not in data.columns:
        logger.error("La columna 'absences' no es troba a les dades.")
        return

    absences = data['absences']

    logger.info("Estadístiques descriptives de les absències:")
    logger.info(f"  Min: {absences.min()}")
    logger.info(f"  Max: {absences.max()}")
    logger.info(f"  Mitjana: {absences.mean():.2f}")
    logger.info(f"  Mediana: {absences.median()}")
    logger.info(f"  Desviació estàndard: {absences.std():.2f}")

    # Distribució de valors
    value_counts = absences.value_counts().sort_index()
    logger.info("Distribució de valors d'absències:")
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
        remainder='passthrough'  # Keep other columns if any, though not expected here
    )

    return preprocessor


def train_regression_model(X: pd.DataFrame, y: pd.Series,
                           preprocessor: ColumnTransformer,
                           model_type: str = 'gradient_boosting',
                           param_file: str = DEFAULT_PARAMS_PATH) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Entrena un model de regressió.
    Els paràmetres es llegeixen d'un fitxer JSON.

    Args:
        X: Característiques d'entrada
        y: Variable objectiu (absències)
        preprocessor: Pipeline de preprocessament
        model_type: Tipus de model a utilitzar ('decision_tree', 'random_forest', o 'gradient_boosting')
        param_file: Fitxer de paràmetres per al model

    Returns:
        Tuple de (model entrenat, paràmetres utilitzats)
    """
    logger.info(f"Entrenant model de regressió {model_type} per predir absències")

    model_params = load_parameters(param_file)
    logger.info(f"S'utilitzaran els següents paràmetres: {model_params}")

    regressor_params = {k.replace('regressor__', ''): v
                        for k, v in model_params.items() if k.startswith('regressor__')}

    if model_type == 'decision_tree':
        regressor = DecisionTreeRegressor(random_state=RANDOM_STATE, **regressor_params)
    elif model_type == 'random_forest':
        regressor = RandomForestRegressor(random_state=RANDOM_STATE, **regressor_params)
    elif model_type == 'gradient_boosting':
        regressor = GradientBoostingRegressor(random_state=RANDOM_STATE, **regressor_params)
    else:
        logger.error(f"Tipus de model no suportat: {model_type}")
        raise ValueError(f"Tipus de model no suportat: {model_type}")

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(
            estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            threshold=-np.inf, # Keep all features by default, can be tuned
            max_features=None # Can be set to a number or fraction
        )),
        ('regressor', regressor)
    ])

    logger.info("Entrenant el model amb els paràmetres especificats...")
    model.fit(X, y)

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

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    abs_errors = np.abs(y_test - y_pred)
    error_metrics = {
        'error_under_1_pct': np.mean(abs_errors < 1) * 100,
        'error_under_2_pct': np.mean(abs_errors < 2) * 100,
        'error_under_5_pct': np.mean(abs_errors < 5) * 100
    }

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        **error_metrics  # Merge error percentage metrics
    }

    logger.info("Mètriques de regressió:")
    for key, value in metrics.items():
        if "pct" in key:
            logger.info(f"  {key.replace('_pct', '').replace('_', ' ').capitalize()}: {value:.2f}%")
        else:
            logger.info(f"  {key.upper()}: {value:.4f}")

    return metrics


def save_model_and_params(model: Pipeline, params: Dict[str, Any],
                          output_dir: str, model_filename: str = "decision_tree_absences.joblib",
                          params_filename: str = "params_absencies.json") -> None:
    """
    Guarda el model i els seus paràmetres en fitxers.

    Args:
        model: Model entrenat a guardar
        params: Paràmetres del model
        output_dir: Directori on es guardaran els fitxers
        model_filename: Nom del fitxer del model
        params_filename: Nom del fitxer dels paràmetres
    """
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(model, model_path)
    logger.info(f"Model guardat a: {model_path}")

    params_path = os.path.join(output_dir, params_filename)
    try:
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4) # Improved formatting with indent=4
        logger.info(f"Paràmetres guardats a: {params_path}")
    except Exception as e:
        logger.error(f"No s'han pogut guardar els paràmetres a {params_path}: {e}")


def define_feature_types(data: pd.DataFrame, target_column: str = 'absences') -> Tuple[List[str], List[str]]:
    """
    Identifica columnes numèriques i categòriques, excloent la columna objectiu.
    """
    if target_column not in data.columns:
        logger.error(f"La columna objectiu '{target_column}' no existeix a les dades.")
        raise ValueError(f"La columna objectiu '{target_column}' no existeix a les dades.")

    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != target_column]

    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    # Ensure target is not in categorical if it was accidentally object type
    categorical_cols = [col for col in categorical_cols if col != target_column]


    logger.info(f"Característiques numèriques identificades: {numerical_cols}")
    logger.info(f"Característiques categòriques identificades: {categorical_cols}")
    return numerical_cols, categorical_cols


def main() -> None:
    """Funció principal que coordina tot el flux de treball."""

    parser = argparse.ArgumentParser(
        description="Model de predicció d'absències amb entrenament i avaluació.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument('--data', type=str, default='../../portuguese_hs_students.csv',
                        help='Ruta al fitxer CSV amb les dades dels estudiants.')
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.path.dirname(__file__), 'output_model'),
                        help="Directori on es guardaran el model entrenat i els paràmetres.")
    parser.add_argument('--model-type', type=str,
                        choices=['decision_tree', 'random_forest', 'gradient_boosting'],
                        default='gradient_boosting',
                        help='Tipus de regressor a entrenar.')
    parser.add_argument('--params', type=str, default=DEFAULT_PARAMS_PATH,
                        help="Ruta al fitxer JSON amb paràmetres personalitzats per al model. "
                             "Si no es troba, s'utilitzaran els paràmetres per defecte.")

    args = parser.parse_args()

    try:
        data = load_data(args.data)
        logger.info(f"Dades carregades: {data.shape[0]} files, {data.shape[1]} columnes")

        analyze_absences_distribution(data)

        numerical_cols, categorical_cols = define_feature_types(data, target_column='absences')

        if not numerical_cols and not categorical_cols:
            logger.error("No s'han trobat característiques per entrenar el model. Revisa les dades.")
            return

        preprocessor = prepare_pipeline(numerical_cols, categorical_cols)

        X = data.drop(columns=['absences'])
        y = data['absences']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=pd.cut(y, bins=5, labels=False, include_lowest=True) if y.nunique() > 5 else None
        )
        # Note: Stratification is tricky for regression. Using simple binning for demonstration.
        # For highly skewed data or specific regression needs, custom stratification might be better.
        # If target has few unique values, stratification might fail or be meaningless.

        logger.info(f"Dades dividides: {X_train.shape[0]} mostres d'entrenament, {X_test.shape[0]} mostres de test")

        model, used_params = train_regression_model(
            X_train, y_train, preprocessor, model_type=args.model_type, param_file=args.params
        )

        metrics = evaluate_regression_model(model, X_test, y_test)

        save_model_and_params(model, used_params, args.output_dir)

        logger.info("El model d'absències ha estat entrenat, avaluat i guardat correctament.")
        print("\n" + "="*60)
        print(" Resum de l'Execució: Model de Predicció d'Absències")
        print("="*60)
        print(f"  Tipus de Model: {args.model_type}")
        print(f"  Dades d'Entrada: {os.path.abspath(args.data)}")
        print(f"  Model i Paràmetres Guardats a: {os.path.abspath(args.output_dir)}")
        print("\n  Mètriques d'Avaluació (conjunt de test):")
        for key, value in metrics.items():
            if "pct" in key:
                print(f"    - {key.replace('_pct', '').replace('_', ' ').capitalize()}: {value:.2f}%")
            else:
                print(f"    - {key.upper()}: {value:.4f}")
        print("="*60)

    except FileNotFoundError:
        logger.error(f"Error: El fitxer de dades no s'ha trobat a la ruta especificada: {args.data}")
    except ValueError as ve:
        logger.error(f"Error de valor durant l'execució: {ve}")
    except Exception as e:
        logger.error(f"S'ha produït un error inesperat durant l'execució: {e}", exc_info=True)
        # exc_info=True in logger.error will print traceback for unexpected errors.


if __name__ == "__main__":
    main()
