#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAC3 - Models de predicció amb arbres de decisió
================================================
Aquest script implementa tres models predictius basats en arbres de decisió:
1. Regressió: predir el nombre d'absències
2. Classificació binària: predir si l'alumne aprovarà (G3 ≥ 10)
3. Classificació binària desequilibrada: predir si l'alumne serà excepcional (G3 ≥ 19)
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any
import warnings  # Added import for warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, r2_score, mean_squared_error,
                             precision_recall_curve, auc)
from sklearn.model_selection import (GridSearchCV, KFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline  # Keep scikit-learn's Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Imports for SMOTE
from imblearn.over_sampling import SMOTE, RandomOverSampler  # Added RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline  # Alias for imbalanced-learn's Pipeline

# Suppress the specific UserWarning from _ranking.py related to precision_recall_curve
warnings.filterwarnings(
    "ignore",
    message="No positive class found in y_true, recall is set to one for all thresholds.",
    category=UserWarning,
    module="sklearn.metrics._ranking"
)

# Configuració del logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42


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
    logger.info("Preparant pipeline de preprocessament")
    
    # Preprocessament per a característiques numèriques
    numerical_transformer = SimpleImputer(strategy='mean')
    
    # Preprocessament per a característiques categòriques
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combina tots els preprocessadors en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor


def train_regression_model(X: pd.DataFrame, y: pd.Series, 
                          preprocessor: ColumnTransformer) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Entrena un model de regressió amb validació creuada i cerca en quadrícula.
    
    Args:
        X: Característiques d'entrada
        y: Variable objectiu (absències)
        preprocessor: Pipeline de preprocessament
    
    Returns:
        Tuple de (model entrenat, millors hiperparàmetres)
    """
    logger.info("Entrenant model de regressió per predir absències")
    
    # Crea el pipeline complet
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=RANDOM_STATE))
    ])
    
    # Paràmetres per a la cerca en quadrícula
    param_grid = {
        'regressor__max_depth': [None, 3, 5, 7, 9],
        'regressor__min_samples_leaf': [1, 5, 10]
    }
    
    # Configuració de la validació creuada
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Cerca en quadrícula amb validació creuada
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=1
    )
    
    # Entrenament del model
    grid_search.fit(X, y)
    
    logger.info(f"Millors paràmetres per regressió: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def train_classification_model(X: pd.DataFrame, y: pd.Series, 
                              preprocessor: ColumnTransformer,
                              class_weight: Union[Dict, str, None] = None) -> Tuple[Union[Pipeline, ImbPipeline], Dict[str, Any]]:
    """
    Entrena un model de classificació amb validació creuada i cerca en quadrícula.
    Pot utilitzar RandomOverSampler si class_weight és "balanced".
    
    Args:
        X: Característiques d'entrada
        y: Variable objectiu binària
        preprocessor: Pipeline de preprocessament
        class_weight: Pesos de les classes (None o "balanced")
    
    Returns:
        Tuple de (model entrenat, millors hiperparàmetres)
    """
    label = "aprovat" if class_weight is None else "excepcional"
    logger.info(f"Entrenant model de classificació per predir {label}")
    
    if class_weight == "balanced":  # Per al model "excepcional"
        logger.info("Aplicant RandomOverSampler per al model desequilibrat.")  # Changed log message
        # Crea el pipeline amb RandomOverSampler, preprocessador i classificador
        model = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('ros', RandomOverSampler(random_state=RANDOM_STATE)),  # Changed SMOTE to RandomOverSampler
            ('classifier', DecisionTreeClassifier(
                random_state=RANDOM_STATE, 
                class_weight=None # Set to None as ROS handles balancing
            ))
        ])
    else:  # Per al model "aprovat" (sense SMOTE)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(
                random_state=RANDOM_STATE, class_weight=class_weight))
        ])
    
    # Paràmetres per a la cerca en quadrícula
    param_grid = {
        'classifier__max_depth': [None, 3, 5, 7, 9],
        'classifier__min_samples_leaf': [1, 5, 10]
    }
    
    # Configuració de la validació creuada
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Cerca en quadrícula amb validació creuada
    # Per a dades extremadament desequilibrades, utilitzem average_precision
    scoring = 'average_precision' if class_weight == "balanced" else 'f1'
    
    # Use n_jobs=1 for the exceptional model to ensure warning filters are respected
    n_jobs_for_gs = 1 if class_weight == "balanced" else -1
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs_for_gs, verbose=0
    )
    
    # Entrenament del model
    grid_search.fit(X, y)
    
    logger.info(f"Millors paràmetres per {label}: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, grid_search.best_params_


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
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'rmse': rmse,
        'r2': r2
    }
    
    logger.info(f"Mètriques de regressió: RMSE={rmse:.4f}, R²={r2:.4f}")
    
    return metrics


def evaluate_classification_model(model: Pipeline, X_test: pd.DataFrame, 
                                y_test: pd.Series, 
                                imbalanced: bool = False) -> Dict[str, Union[float, np.ndarray]]:
    """
    Avalua un model de classificació i torna les mètriques.
    
    Args:
        model: Model entrenat
        X_test: Dades de prova
        y_test: Etiquetes reals
        imbalanced: Indica si és un problema desequilibrat
    
    Returns:
        Diccionari amb les mètriques d'avaluació
    """
    label = "alumne excepcional" if imbalanced else "alumne aprovat"
    logger.info(f"Avaluant model de classificació per {label}")
    
    # Prediccions
    y_pred = model.predict(X_test)
    
    # Probabilitats (només per a problemes desequilibrats)
    if imbalanced:
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Càlcul de mètriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix
    }
    
    # Mètriques addicionals per a problemes desequilibrats
    if imbalanced:
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        metrics['pr_auc'] = pr_auc
        metrics['precision_curve'] = precision_curve
        metrics['recall_curve'] = recall_curve
        
        logger.info(f"Mètriques de classificació desequilibrada: Accuracy={accuracy:.4f}, "
                   f"Precision={precision:.4f}, Recall={recall:.4f}, PR-AUC={pr_auc:.4f}")
        
        # Imprimim la distribució de classes per verificar el desequilibri
        class_counts = np.bincount(y_test)
        logger.info(f"Distribució de classes en el conjunt de test: {class_counts}")
        if len(class_counts) > 1:
            logger.info(f"Percentatge de la classe minoritària: {100*class_counts[1]/sum(class_counts):.2f}%")
        else:
            logger.info("La classe minoritària no està present en el conjunt de test.")
    else:
        logger.info(f"Mètriques de classificació: Accuracy={accuracy:.4f}, "
                   f"Precision={precision:.4f}, Recall={recall:.4f}")

    logger.info(f"Matriu de confusió:\n{conf_matrix}")

    return metrics


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray,
                 auc_score: float, save_path: str) -> None:
    """
    Genera i guarda un gràfic de la corba PR.

    Args:
        precision: Valors de precisió
        recall: Valors de recall
        auc_score: Àrea sota la corba PR
        save_path: Ruta per guardar la imatge
    """
    logger.info("Generant gràfic de la corba PR")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (area = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Corba de Precision-Recall per a la detecció d\'alumnes excepcionals')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(save_path)
    logger.info(f"Gràfic PR guardat a: {save_path}")


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


def save_metrics_summary(metrics_dict: Dict[str, Dict[str, Any]],
                        csv_path: str) -> None:
    """
    Guarda un resum de totes les mètriques en un fitxer CSV.

    Args:
        metrics_dict: Diccionari amb les mètriques de tots els models
        csv_path: Ruta per guardar el fitxer CSV
    """
    # Prepara les dades per al DataFrame
    metrics_summary = {}

    # Regressió
    metrics_summary['absences_rmse'] = metrics_dict['regression']['rmse']
    metrics_summary['absences_r2'] = metrics_dict['regression']['r2']

    # Classificació (aprovat)
    metrics_summary['pass_accuracy'] = metrics_dict['classification']['accuracy']
    metrics_summary['pass_precision'] = metrics_dict['classification']['precision']
    metrics_summary['pass_recall'] = metrics_dict['classification']['recall']

    # Classificació (excepcional)
    metrics_summary['exceptional_accuracy'] = metrics_dict['imbalanced']['accuracy']
    metrics_summary['exceptional_precision'] = metrics_dict['imbalanced']['precision']
    metrics_summary['exceptional_recall'] = metrics_dict['imbalanced']['recall']
    metrics_summary['exceptional_pr_auc'] = metrics_dict['imbalanced']['pr_auc']

    # Converteix a DataFrame i guarda
    pd.DataFrame([metrics_summary]).to_csv(csv_path, index=False)
    logger.info(f"Resum de mètriques guardat a: {csv_path}")


def main() -> None:
    """Funció principal que coordina tot el flux de treball."""

    # Configuració dels arguments de línia d'ordres
    parser = argparse.ArgumentParser(
        description='PAC3 - Models predictius amb arbres de decisió'
    )
    parser.add_argument('--data', type=str, default='../portuguese_hs_students.csv', # Adjusted default path
                        help='Ruta al fitxer CSV amb les dades')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directori on es guardaran els resultats')

    args = parser.parse_args()

    # Carrega les dades
    data = load_data(args.data)

    logger.info(f"Dades carregades: {data.shape[0]} files, {data.shape[1]} columnes")

    # Identificació de característiques numèriques i categòriques
    # Exclou columnes objectiu: absences, G3 s'utilitzaran com a targets
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['absences', 'G3']]

    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    logger.info(f"Característiques numèriques: {len(numerical_cols)}")
    logger.info(f"Característiques categòriques: {len(categorical_cols)}")

    # Prepara el preprocessament
    preprocessor = prepare_pipeline(numerical_cols, categorical_cols)

    # Resultats per guardar
    all_metrics = {}

    # =============== MODEL DE REGRESSIÓ ===============
    logger.info("Iniciant entrenament del model de regressió (absències)")

    # Prepara les dades
    X_regression = data.drop(columns=['absences'])
    y_regression = data['absences']

    # No dividim les dades en entrenament/prova - usem validació creuada

    # Entrena el model
    regression_model, regression_params = train_regression_model(
        X_regression, y_regression, preprocessor
    )

    # Avalua el model (usant les mateixes dades com a test en aquest cas)
    regression_metrics = evaluate_regression_model(
        regression_model, X_regression, y_regression
    )

    all_metrics['regression'] = regression_metrics

    # Guarda el model i els paràmetres
    save_model_and_params(
        regression_model, regression_params,
        os.path.join(args.output_dir, "decision_tree_absences.joblib"),
        os.path.join(args.output_dir, "params_absences.json")
    )

    # =============== MODEL DE CLASSIFICACIÓ: APROVAT ===============
    logger.info("Iniciant entrenament del model de classificació (aprovat)")

    # Prepara les dades
    X_classification = data.drop(columns=['G3'])
    # Crea la variable objectiu: aprovat (G3 >= 10)
    y_classification = (data['G3'] >= 10).astype(int)

    # Divideix les dades en entrenament i prova
    X_train, X_test, y_train, y_test = train_test_split(
        X_classification, y_classification,
        test_size=0.2, random_state=RANDOM_STATE,
        stratify=y_classification
    )

    # Entrena el model
    classification_model, classification_params = train_classification_model(
        X_train, y_train, preprocessor
    )

    # Avalua el model
    classification_metrics = evaluate_classification_model(
        classification_model, X_test, y_test
    )

    all_metrics['classification'] = classification_metrics

    # Guarda el model i els paràmetres
    save_model_and_params(
        classification_model, classification_params,
        os.path.join(args.output_dir, "decision_tree_pass.joblib"),
        os.path.join(args.output_dir, "params_pass.json")
    )

    # =============== MODEL DE CLASSIFICACIÓ: EXCEPCIONAL ===============
    logger.info("Iniciant entrenament del model de classificació (excepcional)")

    # Prepara les dades
    X_imbalanced = data.drop(columns=['G3'])
    # Crea la variable objectiu: excepcional (G3 >= 17) segons la nova definició
    excepcional_threshold = 17  # Llindar modificat
    y_imbalanced = (data['G3'] >= excepcional_threshold).astype(int)

    # Examina la distribució de la classe "excepcional"
    class_distribution = np.bincount(y_imbalanced)
    logger.info(f"Distribució de la classe 'excepcional' (G3 >= {excepcional_threshold}): "
              f"{class_distribution}")
    # Check if minority class exists before trying to access class_distribution[1]
    if len(class_distribution) > 1:
        logger.info(f"Percentatge d'alumnes excepcionals: {100*class_distribution[1]/sum(class_distribution):.2f}%")
    else:
        logger.info("La classe excepcional no té instàncies amb el nou llindar.")

    # Divideix les dades en entrenament i prova
    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imbalanced, y_imbalanced,
        test_size=0.2, random_state=RANDOM_STATE,
        stratify=y_imbalanced
    )

    # Entrena el model amb pesos de classes equilibrades
    imbalanced_model, imbalanced_params = train_classification_model(
        X_train_imb, y_train_imb, preprocessor, class_weight="balanced"
    )

    # Avalua el model
    imbalanced_metrics = evaluate_classification_model(
        imbalanced_model, X_test_imb, y_test_imb, imbalanced=True
    )

    all_metrics['imbalanced'] = imbalanced_metrics

    # Guarda el model i els paràmetres
    save_model_and_params(
        imbalanced_model, imbalanced_params,
        os.path.join(args.output_dir, "decision_tree_exceptional.joblib"),
        os.path.join(args.output_dir, "params_exceptional.json")
    )

    # Genera i guarda el gràfic PR
    plot_pr_curve(
        imbalanced_metrics['precision_curve'],
        imbalanced_metrics['recall_curve'],
        imbalanced_metrics['pr_auc'],
        os.path.join(args.output_dir, "pr_curves.png")
    )

    # Guarda el resum de mètriques
    save_metrics_summary(
        all_metrics,
        os.path.join(args.output_dir, "metrics_summary.csv")
    )

    # Mostra un resum final
    logger.info("Tots els models han estat entrenats i avaluats correctament.")
    print("\n" + "="*50)
    print("PAC3 - Models predictius amb arbres de decisió")
    print("="*50)
    print(f"Models i resultats guardats a: {args.output_dir}")
    print("\nUsa la següent comanda per veure les opcions:")
    print("python pac3_models.py --help")
    print("\nFitxers generats:")
    print("- decision_tree_absences.joblib - Model de regressió per a absències")
    print("- decision_tree_pass.joblib - Model de classificació per aprovat/suspès")
    print("- decision_tree_exceptional.joblib - Model de classificació per alumnes excepcionals")
    print("- metrics_summary.csv - Resum de totes les mètriques d'avaluació")
    print("- pr_curves.png - Corba PR per a la detecció d'alumnes excepcionals")
    print("="*50)


"""
LIMITACIONS I MILLORES FUTURES:

1. Millora en la selecció de característiques: No s'ha realitzat una anàlisi exploratòria
   profunda per determinar quines variables són més rellevants per a cada problema.

2. Tractament de valors extrems: No s'han tractat outliers que podrien afectar
   negativament al rendiment dels models.

3. Optimització d'hiperparàmetres: Es podria ampliar la graella de paràmetres
   i utilitzar tècniques més avançades com RandomizedSearchCV o Bayesian optimization.

4. Models alternatius: Es podria comparar amb altres algorismes com Random Forest,
   Gradient Boosting o SVM per veure si ofereixen millor rendiment.

5. Escalat de característiques: Per a alguns models podria ser beneficiós escalar
   les dades numèriques (tot i que els arbres de decisió no són tan sensibles).

6. Balanceig de dades: Per al problema desequilibrat, es podrien aplicar tècniques
   com SMOTE, undersampling o altres alternatives al class_weight.

7. Validació més robusta: Es podria implementar nested cross-validation per obtenir
   estimacions més fiables del rendiment del model.

8. Visualització més detallada: Es podrien incloure gràfiques de la importància de les
   característiques i visualitzacions dels arbres de decisió per fer-los més interpretables.
"""


if __name__ == "__main__":
    main()

