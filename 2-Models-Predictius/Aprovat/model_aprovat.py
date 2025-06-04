"""
# MODEL DE PREDICCIÓ D'APROVATS
# ----------------------------
# Aquest script entrena un model de classificació binària per predir si un estudiant aprovarà (G3 >= 10).
#
# LLIBRERIES PRINCIPALS:
# - sklearn: Per a preprocessament de dades i models de machine learning
# - imblearn: Per a gestionar el desequilibri de classes amb tècniques de sobremostreig (SMOTE)
# - pandas: Per a manipulació i anàlisi de dades
# - joblib: Per desar el model entrenat
#
# ESTRUCTURA (PSEUDOCODI):
# 1. Càrrega de dades dels estudiants des d'un CSV
# 2. Creació de la variable objectiu binària (aprovat/suspès)
# 3. Preprocessament:
#    - Separació de característiques numèriques i categòriques
#    - Imputació de valors nuls 
#    - Escalat de variables numèriques
#    - Codificació one-hot de variables categòriques
# 4. Sobremostreig amb SMOTE per equilibrar les classes
# 5. Entrenament d'un model GradientBoostingClassifier amb hiperparàmetres carregats d'un JSON
# 6. Desament del model entrenat
#
# EXECUCIÓ: python model_aprovat.py --data [ruta_dades] --output-dir [directori_sortida] --params [ruta_params]
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

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -----------------------
# CONSTANTS
# -----------------------

RANDOM_STATE = 42
DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "params_aprovat.json")


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

# Prepara les dades: separa les característiques i la variable objectiu
def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    # Es crea la variable objectiu binària: 1 si G3 >= 10, 0 si no
    data['aprovat'] = (data['G3'] >= 10).astype(int)

    # S'elimina G3 perquè conté la mateixa informació que la variable objectiu
    X = data.drop(columns=['aprovat', 'G3'])
    y = data['aprovat']

    # S'identifiquen les columnes categòriques
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    return X, y, categorical_features

# Prepara el pipeline de preprocessament per a columnes numèriques i categòriques
def prepare_pipeline(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    # Pipeline per a característiques numèriques
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Omple valors nuls amb la mediana
        ('scaler', StandardScaler())                    # Escala normalitzada
    ])

    # Pipeline per a característiques categòriques
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Omple amb valor més freqüent
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Codificació one-hot
    ])

    # Combina transformadors segons el tipus de dada
    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Deixa passar columnes no tractades (en cas que n’hi hagi)
    )

# Entrena el model de classificació utilitzant Gradient Boosting i SMOTE
def train_classification_model(X: pd.DataFrame, y: pd.Series,
                               categorical_features: List[str],
                               param_file: str = DEFAULT_PARAMS_PATH) -> Tuple[ImbPipeline, Dict[str, Any]]:

    # Identificació de columnes numèriques per exclusió de les categòriques
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # Preprocessat
    preprocessor = prepare_pipeline(numerical_features, categorical_features)

    # Carrega els paràmetres des del fitxer JSON
    model_params = load_parameters(param_file)
    classifier_params = {
        k.replace('classifier__', ''): v
        for k, v in model_params.items()
        if k.startswith('classifier__')
    }

    # Pipeline complet amb preprocessat, SMOTE i classificació
    model = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),  # Sobremostreig per a la classe minoritària
        ('classifier', GradientBoostingClassifier(random_state=RANDOM_STATE, **classifier_params))
    ])

    # Entrenament
    model.fit(X, y)
    return model, model_params

# -----------------------
# FUNCIÓ PRINCIPAL
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Entrena un model de predicció d'aprovats per a estudiants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Arguments CLI
    parser.add_argument("--data", type=str, default='../../portuguese_hs_students.csv',
                        help="Ruta al fitxer CSV amb les dades")

    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__),
                        help="Directori on desar el model i resultats")

    parser.add_argument("--params", type=str, default=DEFAULT_PARAMS_PATH,
                        help="Ruta al fitxer JSON amb els paràmetres del model")

    args = parser.parse_args()

    # Assegura que el directori de sortida existeix
    os.makedirs(args.output_dir, exist_ok=True)

    # Configura un arxiu de log dins el directori de sortida
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "training_log.txt"), mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # -----------------------
    # EXECUCIÓ DE L'SCRIPT
    # -----------------------

    # Carrega de dades
    data = load_data(args.data)

    # Preparació de les dades
    X, y, categorical_features = prepare_data(data)

    # Divisió del conjunt en entrenament/test (no estratificat perquè la variable és binària i SMOTE compensa)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Entrenament del model
    model, _ = train_classification_model(X_train, y_train, categorical_features, args.params)

    # Desament del model entrenat
    model_path = os.path.join(args.output_dir, "model_aprovat.joblib")
    joblib.dump(model, model_path)

if __name__ == "__main__":
    main()
