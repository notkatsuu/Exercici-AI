"""
# MODEL DE PREDICCIÓ D'ABSÈNCIES
# ------------------------------
# Aquest script entrena un model de regressió per predir el nombre d'absències dels estudiants.
# 
# LLIBRERIES PRINCIPALS:
# - sklearn: Per a preprocessament de dades i models de machine learning
# - pandas/numpy: Per a manipulació i anàlisi de dades
# - joblib: Per desar el model entrenat
# 
# ESTRUCTURA (PSEUDOCODI):
# 1. Càrrega de dades dels estudiants des d'un CSV
# 2. Preprocessament:
#    - Separació de característiques numèriques i categòriques
#    - Imputació de valors nuls
#    - Normalització de variables numèriques
#    - Codificació one-hot de variables categòriques
# 3. Selecció de característiques utilitzant Random Forest
# 4. Entrenament d'un model GradientBoostingRegressor amb hiperparàmetres carregats d'un JSON
# 5. Desament del model entrenat i paràmetres utilitzats
#
# EXECUCIÓ: python model_absencies.py --data [ruta_dades] --output-dir [directori_sortida] --params [ruta_params]
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

# Configuració del logger per facilitar el debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "params_absencies.json")

# -----------------------
# FUNCIONS DE SUPORT
# -----------------------

# Carrega les dades des d'un fitxer CSV
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

# Carrega els paràmetres del model des d'un fitxer JSON
def load_parameters(params_path: str) -> Dict[str, Any]:
    with open(params_path, 'r') as f:
        return json.load(f)

# Prepara la pipeline de preprocesament per a variables numèriques i categòriques
def prepare_pipeline(numerical_features: List[str],
                     categorical_features: List[str]) -> ColumnTransformer:
    # Preprocessat de variables numèriques
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),   # Omple valors nuls amb la mediana
        ('scaler', StandardScaler())                     # Escala estàndard (media=0, desviació=1)
    ])

    # Preprocessat de variables categòriques
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Omple valors nuls amb el valor més freqüent
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Codificació one-hot
    ])

    # Combina transformadors per columnes segons el tipus de dada
    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Manté qualsevol columna no especificada (en cas que hi sigui)
    )

# Entrena un model de regressió amb Gradient Boosting, aplicant selecció de característiques prèvia
def train_regression_model(X: pd.DataFrame, y: pd.Series,
                           preprocessor: ColumnTransformer,
                           param_file: str = DEFAULT_PARAMS_PATH) -> Tuple[Pipeline, Dict[str, Any]]:

    # Carrega els paràmetres del fitxer
    model_params = load_parameters(param_file)

    # Extreu els paràmetres que són específics del regressor
    regressor_params = {
        k.replace('regressor__', ''): v
        for k, v in model_params.items()
        if k.startswith('regressor__')
    }

    # Inicialitza el model final: Gradient Boosting
    regressor = GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        **regressor_params
    )

    # Crea la pipeline completa:
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),

        # Selecció de característiques amb Random Forest
        # S'utilitza per seleccionar les variables més importants abans d'entrenar el model final.
        ('feature_selection', SelectFromModel(
            estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            threshold=-np.inf,   # Manté totes les característiques per defecte
            max_features=None    # Es pot ajustar per limitar el nombre de features
        )),

        ('regressor', regressor)
    ])

    # Entrenament de la pipeline
    model.fit(X, y)
    return model, model_params

# Guarda el model entrenat i els paràmetres en fitxers
def save_model_and_params(model: Pipeline, params: Dict[str, Any],
                          output_dir: str,
                          model_filename: str = "model_absences.joblib",
                          params_filename: str = "params_absencies.json") -> None:

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(model, model_path)

    params_path = os.path.join(output_dir, params_filename)
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

# Defineix quines variables són numèriques i quines categòriques
def define_feature_types(data: pd.DataFrame, target_column: str = 'absences') -> Tuple[List[str], List[str]]:
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != target_column]

    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_column]

    return numerical_cols, categorical_cols

# -----------------------
# FUNCIÓ PRINCIPAL
# -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model de predicció d'absències amb entrenament i avaluació.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Arguments de línia de comandes
    parser.add_argument('--data', type=str, default='../../portuguese_hs_students.csv',
                        help='Ruta al fitxer CSV amb les dades dels estudiants.')

    parser.add_argument('--output-dir', type=str, default=os.path.dirname(__file__),
                        help="Directori on es guardaran el model entrenat i els paràmetres.")

    parser.add_argument('--model-type', type=str,
                        choices=['gradient_boosting'], default='gradient_boosting',
                        help='Tipus de regressor a entrenar (actualment només Gradient Boosting).')

    parser.add_argument('--params', type=str, default=DEFAULT_PARAMS_PATH,
                        help="Ruta al fitxer JSON amb paràmetres per al model.")

    args = parser.parse_args()

    # Carrega de dades
    data = load_data(args.data)

    # Determina tipus de columnes
    numerical_cols, categorical_cols = define_feature_types(data, target_column='absences')

    # Preprocessat
    preprocessor = prepare_pipeline(numerical_cols, categorical_cols)

    # Separació de features i variable objectiu
    X = data.drop(columns=['absences'])
    y = data['absences']

    # Divisió train/test estratificada per garantir distribució de y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
        stratify=pd.cut(y, bins=5, labels=False, include_lowest=True) if y.nunique() > 5 else None
    )

    # Entrenament del model
    model, used_params = train_regression_model(X_train, y_train, preprocessor, param_file=args.params)

    # Guarda el model i els paràmetres
    save_model_and_params(model, used_params, args.output_dir)

if __name__ == "__main__":
    main()
