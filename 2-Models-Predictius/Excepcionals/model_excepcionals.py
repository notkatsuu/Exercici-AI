"""
# MODEL DE PREDICCIÓ D'ALUMNES EXCEPCIONALS
# -----------------------------------------
# Aquest script entrena un model de classificació binària per identificar alumnes excepcionals 
# (estudiants amb notes molt altes, G3 >= threshold, normalment 18).
#
# LLIBRERIES PRINCIPALS:
# - sklearn: Per a preprocessament de dades, models de classificació i mètriques d'avaluació
# - imblearn: Per a gestionar el desequilibri de classes amb tècniques de sobremostreig (SMOTE)
# - pandas: Per a manipulació i anàlisi de dades
# - joblib: Per desar el model entrenat
#
# ESTRUCTURA (PSEUDOCODI):
# 1. Càrrega de dades dels estudiants des d'un CSV
# 2. Creació de variable objectiu binària (excepcional/no-excepcional) segons el llindar configurat
# 3. Preprocessament:
#    - Separació de característiques numèriques i categòriques
#    - Imputació de valors nuls 
#    - Escalat de variables numèriques
#    - Codificació one-hot de variables categòriques
# 4. Sobremostreig amb SMOTE per abordar el fort desequilibri de classes (pocs alumnes excepcionals)
# 5. Entrenament d'un model GradientBoostingClassifier amb hiperparàmetres carregats d'un JSON
# 6. Avaluació del model amb mètriques (exactitud, precisió, sensibilitat, F1)
# 7. Desament del model entrenat
#
# EXECUCIÓ: python model_excepcionals.py --data [ruta_dades] --output-dir [directori_sortida] --params [ruta_params]
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

# -----------------------
# CONFIGURACIÓ GENERAL
# -----------------------

# Configuració del logger per registrar informació de l'execució
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants globals
RANDOM_STATE = 42
DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "params_excepcionals.json")

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

# Prepara les dades per a la classificació d'alumnes excepcionals
def prepare_data(data: pd.DataFrame, threshold: int) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    # Es defineix la variable objectiu binària: 1 si G3 >= threshold, 0 si no
    data['excepcional'] = (data['G3'] >= threshold).astype(int)

    # S'elimina G3 per evitar redundància amb la variable objectiu
    X = data.drop(columns=['excepcional', 'G3'])
    y = data['excepcional']

    # S'identifiquen les columnes categòriques
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    return X, y, categorical_features

# Prepara el pipeline de preprocessament per a les dades
def prepare_pipeline(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    # Tractament de columnes numèriques
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Tractament de columnes categòriques
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # ColumnTransformer que aplica el processament corresponent a cada tipus de columna
    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Manté columnes no especificades
    )

# Entrena el model de classificació utilitzant Gradient Boosting i SMOTE
def train_classification_model(X: pd.DataFrame, y: pd.Series,
                               categorical_features: List[str],
                               param_file: str = DEFAULT_PARAMS_PATH) -> Tuple[ImbPipeline, Dict[str, Any]]:

    # Determina columnes numèriques per exclusió de les categòriques
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # Crea el pipeline de preprocessament
    preprocessor = prepare_pipeline(numerical_features, categorical_features)

    # Carrega els paràmetres del model des del fitxer
    model_params = load_parameters(param_file)
    classifier_params = {
        k.replace('classifier__', ''): v
        for k, v in model_params.items() if k.startswith('classifier__')
    }

    # Pipeline complet: preprocessat + sobremostreig + model de classificació
    model = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', GradientBoostingClassifier(random_state=RANDOM_STATE, **classifier_params))
    ])

    # Entrenament
    logger.info("Entrenant el model amb els paràmetres especificats...")
    model.fit(X, y)
    return model, model_params

# -----------------------
# FUNCIÓ PRINCIPAL
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Entrena un model de predicció d'alumnes excepcionals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Arguments CLI
    parser.add_argument("--data", type=str, default='../../portuguese_hs_students.csv',
                        help="Ruta al fitxer CSV amb les dades")
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__),
                        help="Directori on desar el model i resultats")
    parser.add_argument("--params", type=str, default=DEFAULT_PARAMS_PATH,
                        help="Ruta al fitxer JSON amb els paràmetres del model")
    parser.add_argument("--model-type", type=str, default='gradient_boosting',
                        help="Tipus de model (actualment només s'accepta gradient boosting)")

    args = parser.parse_args()

    # Crea el directori de sortida si no existeix
    os.makedirs(args.output_dir, exist_ok=True)

    # Logger també en fitxer
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "training_log.txt"), mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    try:
        logger.info("Iniciant el procés d'entrenament del model d'alumnes excepcionals...")

        # Carreguem dades i paràmetres
        data = load_data(args.data)
        params = load_parameters(args.params)

        # Llindar per definir què és un alumne excepcional
        raw_threshold = params.get('threshold', 18)
        try:
            threshold = int(raw_threshold)
            threshold = min(threshold, 18)  # No té sentit superar 18
        except (TypeError, ValueError):
            logger.warning(f"Valor de llindar invàlid '{raw_threshold}', utilitzant 18 per defecte")
            threshold = 18

        logger.info(f"Utilitzant llindar de {threshold} per determinar alumnes excepcionals")

        # Prepara les dades
        X, y, categorical_features = prepare_data(data, threshold)

        # Mostra la distribució de classes
        class_counts = y.value_counts()
        total = len(y)
        logger.info(f"Total d'alumnes: {total}")
        logger.info(f"Alumnes excepcionals: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/total*100:.2f}%)")
        logger.info(f"Alumnes no excepcionals: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/total*100:.2f}%)")

        # Separació train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # Entrenament del model
        model, _ = train_classification_model(
            X_train, y_train, categorical_features, args.params
        )

        # Prediccions sobre el conjunt de test
        y_pred = model.predict(X_test)

        # Mètriques de rendiment
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"Rendiment del model:")
        logger.info(f"Exactitud (accuracy): {accuracy:.4f}")
        logger.info(f"Precisió (precision): {precision:.4f}")
        logger.info(f"Sensibilitat (recall): {recall:.4f}")
        logger.info(f"Puntuació F1 (F1 score): {f1:.4f}")

        # Desa el model entrenat
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
