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
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"El fitxer de dades no s'ha trobat a: {file_path}")
        raise
    except Exception as e:
        logger.error(f"S'ha produït un error carregant les dades: {e}")
        raise


def load_parameters(params_path: str) -> Dict[str, Any]:
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



def prepare_pipeline(numerical_features: List[str],
                     categorical_features: List[str]) -> ColumnTransformer:

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

 
    model_params = load_parameters(param_file)
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



def save_model_and_params(model: Pipeline, params: Dict[str, Any],
                          output_dir: str, model_filename: str = "decision_tree_absences.joblib",
                          params_filename: str = "params_absencies.json") -> None:

    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(model, model_path)
    logger.info(f"Model guardat a: {model_path}")

    params_path = os.path.join(output_dir, params_filename)
    try:
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4) # Improved formatting with indent=4
    except Exception as e:
        logger.error(f"No s'han pogut guardar els paràmetres a {params_path}: {e}")


def define_feature_types(data: pd.DataFrame, target_column: str = 'absences') -> Tuple[List[str], List[str]]:
    if target_column not in data.columns:
        logger.error(f"La columna objectiu '{target_column}' no existeix a les dades.")
        raise ValueError(f"La columna objectiu '{target_column}' no existeix a les dades.")

    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != target_column]

    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
 
    categorical_cols = [col for col in categorical_cols if col != target_column]

    return numerical_cols, categorical_cols


def main() -> None:
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
    
        model, used_params = train_regression_model(
            X_train, y_train, preprocessor, model_type=args.model_type, param_file=args.params
        )

        save_model_and_params(model, used_params, args.output_dir)


    except FileNotFoundError:
        logger.error(f"Error: El fitxer de dades no s'ha trobat a la ruta especificada: {args.data}")
    except ValueError as ve:
        logger.error(f"Error de valor durant l'execució: {ve}")
    except Exception as e:
        logger.error(f"S'ha produït un error inesperat durant l'execució: {e}", exc_info=True)
        # exc_info=True in logger.error will print traceback for unexpected errors.


if __name__ == "__main__":
    main()
