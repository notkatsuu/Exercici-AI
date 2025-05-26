#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model de predicció d'estudiants excepcionals
============================================
Aquest script implementa un model de classificació binària per predir
si un alumne serà excepcional (G3 ≥ 17) tractant el problema del desequilibri de classes.
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Tuple, Union, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                            recall_score, f1_score, roc_curve, auc,
                            precision_recall_curve, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Imports per balancejar les dades
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
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

def analyze_target_distribution(data: pd.DataFrame, target_column: str = 'G3',
                               threshold: int = 17) -> None:
    """
    Analitza la distribució de la variable objectiu per entendre millor les dades.

    Args:
        data: DataFrame amb les dades dels estudiants
        target_column: Columna objectiu a analitzar
        threshold: Llindar per considerar un alumne excepcional
    """
    target = data[target_column]
    binary_target = (target >= threshold).astype(int)  # Excepcional: G3 >= threshold

    logger.info(f"Estadístiques descriptives de {target_column}:")
    logger.info(f"Min: {target.min()}")
    logger.info(f"Max: {target.max()}")
    logger.info(f"Mitjana: {target.mean():.2f}")
    logger.info(f"Mediana: {target.median()}")

    # Distribució de valors binaris (excepcional/no excepcional)
    value_counts = binary_target.value_counts()
    total = len(binary_target)
    logger.info(f"Distribució d'alumnes excepcionals (G3 >= {threshold}):")
    logger.info(f"  No excepcionals (0): {value_counts.get(0, 0)} estudiants ({100 * value_counts.get(0, 0) / total:.2f}%)")
    logger.info(f"  Excepcionals (1): {value_counts.get(1, 0)} estudiants ({100 * value_counts.get(1, 0) / total:.2f}%)")

    # Distribució detallada de G3
    value_counts_detailed = target.value_counts().sort_index()
    logger.info(f"Distribució detallada de {target_column}:")
    for value, count in value_counts_detailed.items():
        logger.info(f"  {value}: {count} estudiants ({100 * count / total:.2f}%)")

    # Identificar les variables més correlacionades amb G3
    correlations = data.corr()[target_column].sort_values(ascending=False)
    logger.info(f"Variables més correlacionades amb {target_column}:")
    for col, corr in correlations.items():
        if col != target_column and abs(corr) > 0.3:
            logger.info(f"  {col}: {corr:.4f}")

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

def train_classification_model(X: pd.DataFrame, y: pd.Series,
                              preprocessor: ColumnTransformer,
                              model_type: str = 'random_forest',
                              oversampling: str = 'smote') -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Entrena un model de classificació amb balanceig de classes i validació creuada.

    Args:
        X: Característiques d'entrada
        y: Variable objectiu binària (excepcional/no excepcional)
        preprocessor: Pipeline de preprocessament
        model_type: Tipus de model a utilitzar ('random_forest' o 'gradient_boosting')
        oversampling: Mètode de sobremostreig ('smote', 'adasyn', 'borderline_smote', o None)

    Returns:
        Tuple de (model entrenat, millors hiperparàmetres)
    """
    logger.info(f"Entrenant model de classificació {model_type} per predir estudiants excepcionals")
    logger.info(f"Utilitzant mètode de sobremostreig: {oversampling}")

    # Selecció del model segons el tipus especificat
    if model_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'gradient_boosting':
        classifier = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 5, 8],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__min_samples_leaf': [1, 3, 5]
        }
    else:
        raise ValueError(f"Tipus de model no suportat: {model_type}")

    # Selecció del mètode de sobremostreig
    if oversampling == 'smote':
        oversampler = SMOTE(random_state=RANDOM_STATE, sampling_strategy='auto')
    elif oversampling == 'adasyn':
        oversampler = ADASYN(random_state=RANDOM_STATE, sampling_strategy='auto')
    elif oversampling == 'borderline_smote':
        oversampler = BorderlineSMOTE(random_state=RANDOM_STATE, sampling_strategy='auto')
    elif oversampling is None:
        oversampler = None
    else:
        raise ValueError(f"Mètode de sobremostreig no suportat: {oversampling}")

    # Crea el pipeline complet amb selecció de característiques i balanceig (si s'especifica)
    steps = []
    steps.append(('preprocessor', preprocessor))

    # Utilitzem RFECV per seleccionar les millors característiques
    steps.append(('feature_selection', RFECV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        step=1,
        cv=StratifiedKFold(5),
        scoring='f1',
        min_features_to_select=5
    )))

    # Afegim sobremostreig si s'ha especificat
    if oversampler is not None:
        steps.append(('oversampling', oversampler))

    steps.append(('classifier', classifier))

    # Creem el pipeline
    if oversampler is not None:
        model = ImbPipeline(steps=steps)
    else:
        model = Pipeline(steps=steps)

    # Configuració de la validació creuada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Cerca en quadrícula amb validació creuada
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
    )

    # Entrenament del model
    grid_search.fit(X, y)

    logger.info(f"Millors paràmetres per classificació: {grid_search.best_params_}")
    logger.info(f"Millor puntuació F1 en validació creuada: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_classification_model(model: Union[Pipeline, ImbPipeline], X_test: pd.DataFrame,
                                y_test: pd.Series) -> Dict[str, Union[float, np.ndarray]]:
    """
    Avalua un model de classificació i torna les mètriques.

    Args:
        model: Model entrenat
        X_test: Dades de prova
        y_test: Etiquetes reals

    Returns:
        Diccionari amb les mètriques d'avaluació
    """
    logger.info("Avaluant model de classificació")

    # Prediccions
    y_pred = model.predict(X_test)

    # Probabilitats per ROC i PR curves
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        y_proba = None
        has_proba = False

    # Càlcul de mètriques
    accuracy = accuracy_score(y_test, y_pred)
    # Utilitzem zero_division=0 per evitar warnings si no hi ha positius
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

    # Càlcul de corbes ROC i PR si hi ha probabilitats disponibles
    if has_proba:
        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        metrics['roc_auc'] = roc_auc
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr

        # PR curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
        metrics['pr_auc'] = pr_auc
        metrics['precision_curve'] = precision_curve
        metrics['recall_curve'] = recall_curve

    logger.info(f"Mètriques de classificació per estudiants excepcionals:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    if has_proba:
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"PR AUC: {pr_auc:.4f}")

    logger.info(f"Matriu de confusió:")
    logger.info(f"{conf_matrix}")

    return metrics

def plot_feature_importance(model: Union[Pipeline, ImbPipeline], column_names: List[str],
                           output_path: str) -> None:
    """
    Genera i guarda un gràfic d'importància de característiques.

    Args:
        model: Model entrenat
        column_names: Noms de les columnes originals
        output_path: Ruta on guardar el gràfic
    """
    try:
        # Intentem extreure el model final
        classifier = model.named_steps['classifier']

        if hasattr(classifier, 'feature_importances_'):
            feature_importances = classifier.feature_importances_

            # Obtenim les característiques seleccionades
            selected_indices = model.named_steps['feature_selection'].get_support()

            # Processador de dades
            preprocessor = model.named_steps['preprocessor']

            # Obtenim els noms de les característiques després de preprocessament
            feature_names = preprocessor.get_feature_names_out()

            # Filtrem només les característiques seleccionades
            selected_features = [feature_names[i] for i, selected in enumerate(selected_indices) if selected]

            # Només usem importàncies de característiques seleccionades
            sorted_indices = np.argsort(feature_importances)
            sorted_features = [selected_features[i] for i in sorted_indices[-15:]]  # Top 15
            sorted_importances = [feature_importances[i] for i in sorted_indices[-15:]]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
            plt.yticks(range(len(sorted_importances)), [str(f) for f in sorted_features])
            plt.title('Importància de les Característiques en la Predicció d\'Estudiants Excepcionals')
            plt.xlabel('Importància')
            plt.tight_layout()
            plt.savefig(output_path)
            logger.info(f"Gràfic d'importància de característiques guardat a: {output_path}")
        else:
            logger.warning("El model no té atribut feature_importances_. No es pot generar el gràfic.")
    except Exception as e:
        logger.error(f"Error generant gràfic d'importància: {e}")

def plot_confusion_matrix(conf_matrix: np.ndarray, output_path: str) -> None:
    """
    Genera i guarda un gràfic de la matriu de confusió.

    Args:
        conf_matrix: Matriu de confusió
        output_path: Ruta on guardar el gràfic
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Excepcional'],
               yticklabels=['Normal', 'Excepcional'])
    plt.xlabel('Predicció')
    plt.ylabel('Real')
    plt.title('Matriu de Confusió - Estudiants Excepcionals')
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Gràfic de matriu de confusió guardat a: {output_path}")

def plot_roc_pr_curves(metrics: Dict[str, Union[float, np.ndarray]], output_path: str) -> None:
    """
    Genera i guarda un gràfic amb les corbes ROC i PR.

    Args:
        metrics: Diccionari amb mètriques incloent dades per corbes ROC i PR
        output_path: Ruta on guardar el gràfic
    """
    if 'fpr' not in metrics or 'tpr' not in metrics:
        logger.warning("No es poden generar corbes ROC/PR perquè no hi ha dades de probabilitats")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Corba ROC
    ax1.plot(metrics['fpr'], metrics['tpr'],
            label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Taxa de Falsos Positius')
    ax1.set_ylabel('Taxa de Veritables Positius')
    ax1.set_title('Corba ROC - Estudiants Excepcionals')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    # Corba PR
    ax2.plot(metrics['recall_curve'], metrics['precision_curve'],
            label=f'PR (AUC = {metrics["pr_auc"]:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Corba Precision-Recall - Estudiants Excepcionals')
    ax2.legend(loc='lower left')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Gràfic de corbes ROC i PR guardat a: {output_path}")

def save_model_and_params(model: Union[Pipeline, ImbPipeline], params: Dict[str, Any],
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
        description='Model de predicció d\'estudiants excepcionals'
    )
    parser.add_argument('--data', type=str, default='../portuguese_hs_students.csv',
                        help='Ruta al fitxer CSV amb les dades')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directori on es guardaran els resultats')
    parser.add_argument('--model-type', type=str,
                        choices=['random_forest', 'gradient_boosting'],
                        default='random_forest',
                        help='Tipus de model a entrenar')
    parser.add_argument('--threshold', type=int, default=17,
                        help='Llindar per considerar un alumne excepcional (G3 >= threshold)')
    parser.add_argument('--oversampling', type=str,
                        choices=['smote', 'adasyn', 'borderline_smote', 'none'],
                        default='smote',
                        help='Mètode de sobremostreig per tractar el desequilibri de classes')

    args = parser.parse_args()

    # Processa l'argument de sobremostreig
    oversampling = None if args.oversampling == 'none' else args.oversampling

    # Carrega les dades
    data = load_data(args.data)
    logger.info(f"Dades carregades: {data.shape[0]} files, {data.shape[1]} columnes")

    # Analitza la distribució de la variable objectiu
    analyze_target_distribution(data, threshold=args.threshold)

    # Identificació de característiques numèriques i categòriques
    # Exclou G3 que s'utilitzarà per crear la variable objectiu
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'G3']

    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    logger.info(f"Característiques numèriques: {len(numerical_cols)}")
    logger.info(f"Característiques categòriques: {len(categorical_cols)}")

    # Prepara el preprocessament
    preprocessor = prepare_pipeline(numerical_cols, categorical_cols)

    # Prepara les dades
    X = data.drop(columns=['G3'])
    # Crea la variable objectiu: excepcional (G3 >= threshold)
    y = (data['G3'] >= args.threshold).astype(int)

    # Divideix les dades en entrenament i prova
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(f"Dades dividides: {X_train.shape[0]} mostres d'entrenament, {X_test.shape[0]} mostres de test")

    # Comprovem la distribució en les dades de test i entrenament
    logger.info(f"Distribució de classes en entrenament: {np.bincount(y_train)}")
    logger.info(f"Distribució de classes en test: {np.bincount(y_test)}")

    # Entrena el model
    model, params = train_classification_model(
        X_train, y_train, preprocessor,
        model_type=args.model_type,
        oversampling=oversampling
    )

    # Avalua el model
    metrics = evaluate_classification_model(model, X_test, y_test)

    # Guarda el model i els paràmetres
    model_path = os.path.join(args.output_dir, "decision_tree_exceptional.joblib")
    params_path = os.path.join(args.output_dir, "params_exceptional.json")
    save_model_and_params(model, params, model_path, params_path)

    # Genera i guarda gràfics
    # Gràfic d'importància de característiques
    feature_importance_path = os.path.join(args.output_dir, "feature_importance_exceptional.png")
    plot_feature_importance(model, list(X.columns), feature_importance_path)

    # Gràfic de matriu de confusió
    confusion_matrix_path = os.path.join(args.output_dir, "confusion_matrix_exceptional.png")
    plot_confusion_matrix(metrics['confusion_matrix'], confusion_matrix_path)

    # Gràfic de corbes ROC i PR
    if 'roc_auc' in metrics:
        curves_path = os.path.join(args.output_dir, "roc_pr_curves_exceptional.png")
        plot_roc_pr_curves(metrics, curves_path)

    # Mostra un resum final
    logger.info("El model d'estudiants excepcionals ha estat entrenat i avaluat correctament.")
    print("\n" + "="*50)
    print("Model de predicció d'estudiants excepcionals")
    print("="*50)
    print(f"Model i resultats guardats a: {args.output_dir}")
    print("\nUsa la següent comanda per veure les opcions:")
    print(f"python {os.path.basename(__file__)} --help")
    print("\nFitxers generats:")
    print(f"- {os.path.basename(model_path)} - Model de classificació per estudiants excepcionals")
    print(f"- {os.path.basename(params_path)} - Paràmetres del model")
    print(f"- {os.path.basename(feature_importance_path)} - Gràfic d'importància de característiques")
    print(f"- {os.path.basename(confusion_matrix_path)} - Gràfic de matriu de confusió")
    if 'roc_auc' in metrics:
        print(f"- {os.path.basename(curves_path)} - Gràfic de corbes ROC i PR")
    print("="*50)


if __name__ == "__main__":
    main()
