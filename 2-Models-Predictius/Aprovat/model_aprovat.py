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
import numpy as np
import pandas as pd
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Import SMOTE for handling imbalanced classes
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
DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "params_aprovat.json")


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
        "classifier__n_estimators": 100,
        "classifier__learning_rate": 0.1,
        "classifier__max_depth": 5,
        "classifier__min_samples_leaf": 5,
        "classifier__subsample": 0.8
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


def analyze_class_distribution(data: pd.DataFrame) -> None:
    """
    Analitza la distribució de la variable target (aprovats/suspesos) per entendre millor les dades.

    Args:
        data: DataFrame amb les dades a analitzar
    """
    logger.info("Analitzant distribució de classes (aprovats/suspesos)...")

    # Crear la variable target (aprovat si G3 >= 10)
    data['aprovat'] = (data['G3'] >= 10).astype(int)

    # Mostrar distribució
    class_counts = data['aprovat'].value_counts()
    total = len(data)

    logger.info(f"Total d'estudiants: {total}")
    logger.info(f"Estudiants aprovats: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/total*100:.2f}%)")
    logger.info(f"Estudiants suspesos: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/total*100:.2f}%)")


def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepara les dades per a l'entrenament del model.

    Args:
        data: DataFrame amb les dades a preparar

    Returns:
        X: DataFrame amb les característiques
        y: Series amb la variable objectiu (aprovat/suspès)
        categorical_features: Llista de característiques categòriques
    """
    logger.info("Preparant dades per a l'entrenament del model...")

    # Crear la variable objectiu: aprovat (1) / suspès (0)
    data['aprovat'] = (data['G3'] >= 10).astype(int)

    # No utilitzar G3 com a característica perquè és la base del que volem predir
    X = data.drop(columns=['aprovat', 'G3'])
    y = data['aprovat']

    # Identificar característiques categòriques
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    logger.info(f"Dades preparades: {X.shape[0]} files, {X.shape[1]} columnes")
    logger.info(f"Característiques categòriques: {categorical_features}")

    return X, y, categorical_features


def create_preprocessing_pipeline(categorical_features: List[str], X: pd.DataFrame) -> ColumnTransformer:
    """
    Crea un pipeline de preprocessament per a les dades.

    Args:
        categorical_features: Llista de característiques categòriques
        X: DataFrame amb les dades a processar

    Returns:
        ColumnTransformer que processa característiques numèriques i categòriques
    """
    logger.info("Creant pipeline de preprocessament...")

    # Identificar les característiques numèriques (les que no són categòriques)
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # Preprocessament per a característiques numèriques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessament per a característiques categòriques
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combinació de preprocessadors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    return preprocessor


def train_model(X: pd.DataFrame, y: pd.Series, categorical_features: List[str], params: Dict[str, Any],
               output_dir: str) -> Pipeline:
    """
    Entrena el model de classificació per predir si un alumne aprovarà.

    Args:
        X: DataFrame amb les característiques
        y: Series amb la variable objectiu (aprovat/suspès)
        categorical_features: Llista de característiques categòriques
        params: Diccionari amb els paràmetres del model
        output_dir: Directori on desar el model entregat

    Returns:
        Model entrenat
    """
    logger.info("Entrenant model de classificació...")

    # Preprocessament
    preprocessor = create_preprocessing_pipeline(categorical_features, X)

    # Dividir dades en conjunts d'entrenament i test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    logger.info(f"Conjunt d'entrenament: {X_train.shape[0]} mostres, Conjunt de test: {X_test.shape[0]} mostres")

    # Comprovar balanceig de classes
    train_class_dist = np.bincount(y_train)
    test_class_dist = np.bincount(y_test)

    logger.info(f"Distribució de classes en entrenament: {train_class_dist}")
    logger.info(f"Distribució de classes en test: {test_class_dist}")

    # Crear pipeline amb SMOTE per gestionar el desbalanceig
    smote_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])

    # Establir paràmetres del model
    for param, value in params.items():
        if param.startswith("classifier__"):
            param_parts = param.split("__")
            if len(param_parts) == 2 and param_parts[0] == "classifier":
                param_name = param_parts[1]
                try:
                    setattr(smote_pipeline.named_steps['classifier'], param_name, value)
                    logger.info(f"Paràmetre {param_name} establert a {value}")
                except Exception as e:
                    logger.error(f"No s'ha pogut establir el paràmetre {param_name}: {e}")

    # Entrenar el model
    logger.info("Iniciant entrenament del model...")
    smote_pipeline.fit(X_train, y_train)
    logger.info("Model entrenat correctament")

    # Avaluar el model
    y_pred = smote_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Rendiment del model:")
    logger.info(f"Exactitud (accuracy): {accuracy:.4f}")
    logger.info(f"Precisió (precision): {precision:.4f}")
    logger.info(f"Sensibilitat (recall): {recall:.4f}")
    logger.info(f"Puntuació F1 (F1 score): {f1:.4f}")

    # Calcular la matriu de confusió
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    logger.info(f"Matriu de confusió:")
    logger.info(f"Verdaders positius (TP): {tp}")
    logger.info(f"Falsos positius (FP): {fp}")
    logger.info(f"Verdaders negatius (TN): {tn}")
    logger.info(f"Falsos negatius (FN): {fn}")

    # Desar el model
    model_path = os.path.join(output_dir, "model_aprovat.joblib")
    joblib.dump(smote_pipeline, model_path)
    logger.info(f"Model desat a {model_path}")

    return smote_pipeline


def generate_precision_recall_curve(model, X_test, y_test, output_dir):
    """
    Genera la corba de precisió-record per al model.

    Args:
        model: Model entrenat
        X_test: Dades de test
        y_test: Etiquetes de test
        output_dir: Directori on desar el gràfic
    """
    import matplotlib.pyplot as plt

    try:
        # Obtenir probabilitats de predicció per a la classe positiva (1)
        y_score = model.predict_proba(X_test)[:, 1]

        # Calcular precisió i record per a diferents llindars
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)

        # Calcular l'àrea sota la corba PR
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'Corba PR (àrea = {pr_auc:.2f})')
        plt.xlabel('Record (sensibilitat)')
        plt.ylabel('Precisió')
        plt.title('Corba de Precisió-Record')
        plt.legend(loc="best")
        plt.grid(True)

        # Desar el gràfic
        output_path = os.path.join(output_dir, "pr_curves.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Corba de precisió-record desada a {output_path}")

    except Exception as e:
        logger.error(f"Error generant la corba de precisió-record: {e}")


def main():
    """
    Funció principal per entrenar i avaluar el model de predicció d'aprovats.
    """
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
    # Model type argument is kept for backward compatibility but will always use gradient_boosting
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
        logger.info("Iniciant el procés d'entrenament del model de predicció d'aprovats...")
        logger.info(f"Fitxer de dades: {args.data}")
        logger.info(f"Directori de sortida: {args.output_dir}")
        logger.info(f"Fitxer de paràmetres: {args.params}")

        # Carregar dades
        data = load_data(args.data)

        # Analitzar distribució de classes
        analyze_class_distribution(data)

        # Preparar dades
        X, y, categorical_features = prepare_data(data)

        # Carregar paràmetres del model
        params = load_parameters(args.params)

        # Entrenar model
        model = train_model(X, y, categorical_features, params, args.output_dir)

        # Dividir dades en conjunts d'entrenament i test per a l'avaluació final
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        # Generar corba de precisió-record
        try:
            generate_precision_recall_curve(model, X_test, y_test, args.output_dir)
        except Exception as e:
            logger.warning(f"No s'ha pogut generar la corba de precisió-record: {e}")

        logger.info("Procés d'entrenament completat correctament.")

        print("\n" + "="*60)
        print(" Resum de l'Execució: Model de Predicció d'Aprovats")
        print("="*60)
        print(f"  Dades d'Entrada: {os.path.abspath(args.data)}")
        print(f"  Model i Paràmetres Guardats a: {os.path.abspath(args.output_dir)}")
        print("\n  Avaluació del model al conjunt de test:")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"    - Exactitud (accuracy): {accuracy:.4f}")
        print(f"    - Precisió (precision): {precision:.4f}")
        print(f"    - Sensibilitat (recall): {recall:.4f}")
        print(f"    - Puntuació F1: {f1:.4f}")
        print("="*60)

    except Exception as e:
        logger.exception(f"Error durant l'execució: {e}")
        raise


if __name__ == "__main__":
    main()
