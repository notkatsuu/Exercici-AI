import joblib
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox  # Removed filedialog
import json
import os
import sys
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score

# Columnes esperades pel model d'absències (totes menys 'absences' que és el target, però G3 sí s'inclou com a feature)
EXPECTED_COLUMNS_ABSENCES = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
    'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'G1', 'G2', 'G3'
]

# Columnes esperades pel model d'aprovats (totes menys G3 que és el que volem predir)
EXPECTED_COLUMNS_APROVAT = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
    'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'G1', 'G2', 'absences'
]

# Diccionari amb prompts, tipus, opcions vàlides (opcional) i valors per defecte per a cada columna
# Aquest diccionari serveix per a ambdós models, ja que tenen les mateixes columnes
COLUMN_DEFINITIONS = {
    'school': {"prompt": "Escola (GP/MS):", "type": str, "options": ['GP', 'MS'], "default": 'GP'},
    'sex': {"prompt": "Sexe (F/M):", "type": str, "options": ['F', 'M'], "default": 'F'},
    'age': {"prompt": "Edat (15-22):", "type": int, "default": 18},
    'address': {"prompt": "Adreça (U/R):", "type": str, "options": ['U', 'R'], "default": 'U'},
    'famsize': {"prompt": "Mida família (LE3/GT3):", "type": str, "options": ['LE3', 'GT3'], "default": 'GT3'},
    'Pstatus': {"prompt": "Estat pares (T/A):", "type": str, "options": ['T', 'A'], "default": 'A'},
    'Medu': {"prompt": "Educ. mare (0-4):", "type": int, "default": 3},
    'Fedu': {"prompt": "Educ. pare (0-4):", "type": int, "default": 3},
    'Mjob': {"prompt": "Feina mare:", "type": str, "options": ['teacher', 'health', 'services', 'at_home', 'other'], "default": 'other'},
    'Fjob': {"prompt": "Feina pare:", "type": str, "options": ['teacher', 'health', 'services', 'at_home', 'other'], "default": 'other'},
    'reason': {"prompt": "Raó escola:", "type": str, "options": ['home', 'reputation', 'course', 'other'], "default": 'course'},
    'guardian': {"prompt": "Tutor:", "type": str, "options": ['mother', 'father', 'other'], "default": 'mother'},
    'traveltime': {"prompt": "Temps viatge (1-4):", "type": int, "default": 2},
    'studytime': {"prompt": "Temps estudi (1-4):", "type": int, "default": 2},
    'failures': {"prompt": "Suspesos previs:", "type": int, "default": 0},
    'schoolsup': {"prompt": "Sup. escolar (yes/no):", "type": str, "options": ['yes', 'no'], "default": 'no'},
    'famsup': {"prompt": "Sup. familiar (yes/no):", "type": str, "options": ['yes', 'no'], "default": 'yes'},
    'paid': {"prompt": "Classes pagades (yes/no):", "type": str, "options": ['yes', 'no'], "default": 'no'},
    'activities': {"prompt": "Activitats extra (yes/no):", "type": str, "options": ['yes', 'no'], "default": 'yes'},
    'nursery': {"prompt": "Escola bressol (yes/no):", "type": str, "options": ['yes', 'no'], "default": 'yes'},
    'higher': {"prompt": "Estudis sup. (yes/no):", "type": str, "options": ['yes', 'no'], "default": 'yes'},
    'internet': {"prompt": "Internet (yes/no):", "type": str, "options": ['yes', 'no'], "default": 'yes'},
    'romantic': {"prompt": "Relació (yes/no):", "type": str, "options": ['yes', 'no'], "default": 'no'},
    'famrel': {"prompt": "Rel. familiar (1-5):", "type": int, "default": 4},
    'freetime': {"prompt": "Temps lliure (1-5):", "type": int, "default": 3},
    'goout': {"prompt": "Sortides (1-5):", "type": int, "default": 3},
    'Dalc': {"prompt": "Alcohol lab. (1-5):", "type": int, "default": 1},
    'Walc': {"prompt": "Alcohol cap set. (1-5):", "type": int, "default": 1},
    'health': {"prompt": "Salut (1-5):", "type": int, "default": 3},
    'G1': {"prompt": "Nota G1 (0-20):", "type": int, "default": 10},
    'G2': {"prompt": "Nota G2 (0-20):", "type": int, "default": 10},
    'G3': {"prompt": "Nota G3 (0-20):", "type": int, "default": 10},
    'absences': {"prompt": "Absències:", "type": int, "default": 5}
}

# Paràmetres configurables dels models
DEFAULT_MODEL_PARAMS_ABSENCES = {
    "regressor__n_estimators": 300,
    "regressor__learning_rate": 0.05,
    "regressor__max_depth": 8,
    "regressor__min_samples_leaf": 3,
    "regressor__subsample": 0.9
}

DEFAULT_MODEL_PARAMS_APROVAT = {
    "classifier__n_estimators": 100,
    "classifier__learning_rate": 0.1,
    "classifier__max_depth": 5,
    "classifier__min_samples_leaf": 5,
    "classifier__subsample": 0.8
}

class PredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Predicció d'Estudiants")
        master.configure(bg='#2B2B2B')
        master.geometry("850x700") # Una mica més gran per acomodar la secció de paràmetres

        # Style for widgets
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='#2B2B2B', foreground='white')
        style.configure('TLabel', background='#2B2B2B', foreground='white', padding=3, font=('Arial', 9))
        style.configure('TButton', background='#4A4A4A', foreground='white', padding=5, borderwidth=0, font=('Arial', 10, 'bold'))
        style.map('TButton', background=[('active', '#6A6A6A')])
        style.configure('TEntry', fieldbackground='#3C3F41', foreground='white', insertcolor='white', borderwidth=1, relief='flat', padding=3, font=('Arial', 9))
        style.configure('TCombobox', fieldbackground='#3C3F41', foreground='white', selectbackground='#3C3F41', selectforeground='white', arrowcolor='white', padding=3, font=('Arial', 9))
        style.configure('TSpinbox', arrowsize=13, background='#2B2B2B', foreground='white', fieldbackground='#3C3F41', buttonbackground='#4A4A4A', relief='flat', padding=3)
        style.configure('TRadiobutton', background='#2B2B2B', foreground='white', padding=3, font=('Arial', 9))
        master.option_add('*TCombobox*Listbox.background', '#3C3F41')
        master.option_add('*TCombobox*Listbox.foreground', 'white')
        master.option_add('*TCombobox*Listbox.selectBackground', '#4A4A4A')
        master.option_add('*TCombobox*Listbox.selectForeground', 'white')
        master.option_add('*TSpinbox*Listbox.background', '#3C3F41')
        master.option_add('*TSpinbox*Listbox.foreground', '#FFFFFF')
        style.configure('TNotebook', background='#2B2B2B', borderwidth=0)
        style.configure('TNotebook.Tab', background='#3C3F41', foreground='white', padding=[10, 5], font=('Arial', 9, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', '#5C5C5C')], foreground=[('selected', 'white')])
        style.configure('TFrame', background='#2B2B2B')

        # Ruta al fitxer CSV amb les dades
        self.data_path = os.path.abspath("../portuguese_hs_students.csv")

        # Variable per seguir el model actiu
        self.model_type = tk.StringVar(value="absencies")

        # Rutes als fitxers dels models (camins absoluts per evitar ambigüitats)
        self.model_paths = {
            "absencies": os.path.abspath("../2-Models-Predictius/Absencies/decision_tree_absences.joblib"),
            "aprovat": os.path.abspath("../2-Models-Predictius/Aprovat/decision_tree_aprovat.joblib")
        }

        self.params_paths = {
            "absencies": os.path.abspath("../2-Models-Predictius/Absencies/params_absencies.json"),
            "aprovat": os.path.abspath("../2-Models-Predictius/Aprovat/params_aprovat.json")
        }

        # Crear un notebook amb pestanyes
        self.notebook = ttk.Notebook(master)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1)

        # Model selector frame
        self.selector_frame = ttk.Frame(master)
        self.selector_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Selector de model
        ttk.Label(self.selector_frame, text="Selecciona el model:").grid(row=0, column=0, padx=5, pady=5)

        # Botó de ràdio per a absències
        self.radio_absencies = ttk.Radiobutton(self.selector_frame, text="Predicció d'Absències",
                                              variable=self.model_type, value="absencies",
                                              command=self.change_model)
        self.radio_absencies.grid(row=0, column=1, padx=10, pady=5)

        # Botó de ràdio per a aprovats
        self.radio_aprovat = ttk.Radiobutton(self.selector_frame, text="Predicció d'Aprovats",
                                             variable=self.model_type, value="aprovat",
                                             command=self.change_model)
        self.radio_aprovat.grid(row=0, column=2, padx=10, pady=5)

        # Pestanya per a la predicció
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="Predicció")

        # Pestanya per a la configuració del model
        self.config_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text="Configuració del Model")

        # Pestanya per a l'avaluació del model
        self.evaluation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.evaluation_tab, text="Avaluació del Model")

        # Carregar model i paràmetres
        self.load_active_model()

        # Configurar pestanya de predicció
        self.entries = {}
        self.create_input_fields(self.prediction_tab)

        self.predict_button = ttk.Button(self.prediction_tab, text="Predir", command=self.predict)
        self.predict_button.grid(row=1, column=0, pady=15, padx=10, sticky="ew")

        self.result_label = ttk.Label(self.prediction_tab, text="Resultat de la predicció:", font=("Arial", 11, "bold"))
        self.result_label.grid(row=2, column=0, pady=10, padx=10)

        # Configurar pestanya de configuració del model
        self.param_entries = {}
        self.create_param_config_ui(self.config_tab)

        # Configurar pestanya d'avaluació del model
        self.create_evaluation_ui(self.evaluation_tab)

        # Actualitzar el títol i contingut segons el model seleccionat
        self.update_ui_for_model()

    def change_model(self):
        """Canvia el model actiu segons la selecció de l'usuari."""
        self.load_active_model()
        self.update_ui_for_model()

    def update_ui_for_model(self):
        """Actualitza la UI segons el model seleccionat."""
        model_type = self.model_type.get()

        # Actualitzar títol
        if model_type == "absencies":
            self.master.title("Predicció d'Absències")
            self.predict_button.config(text="Predir Absències")
        else:
            self.master.title("Predicció d'Aprovats")
            self.predict_button.config(text="Predir Aprovat")

        # Actualitzar paràmetres al tab de configuració
        self.create_param_config_ui(self.config_tab)

        # Re-crear els camps d'entrada segons les columnes necessàries
        for widget in self.prediction_tab.winfo_children():
            if widget != self.predict_button and widget != self.result_label:
                widget.destroy()

        self.entries = {}
        self.create_input_fields(self.prediction_tab)

    def load_active_model(self):
        """Carrega el model actiu i els seus paràmetres."""
        model_type = self.model_type.get()

        try:
            self.model = joblib.load(self.model_paths[model_type])
            print(f"Model {model_type} carregat des de: {self.model_paths[model_type]}")
        except FileNotFoundError:
            messagebox.showerror("Error", f"No s'ha trobat el fitxer del model a {self.model_paths[model_type]}")
            self.model = None
        except Exception as e:
            messagebox.showerror("Error", f"S'ha produït un error carregant el model: {e}")
            self.model = None

        self.params = self.load_params(model_type)

    def load_params(self, model_type):
        """Carrega els paràmetres del model indicat."""
        default_params = DEFAULT_MODEL_PARAMS_ABSENCES if model_type == "absencies" else DEFAULT_MODEL_PARAMS_APROVAT
        params_path = self.params_paths[model_type]

        try:
            with open(params_path, 'r') as f:
                params = json.load(f)
                print(f"Paràmetres del model carregats des de: {params_path}")
                return params
        except FileNotFoundError:
            print(f"No s'ha trobat el fitxer de paràmetres a {params_path}, s'utilitzaran paràmetres per defecte")
            return default_params
        except json.JSONDecodeError:
            print(f"Error de format al fitxer de paràmetres, s'utilitzaran paràmetres per defecte")
            return default_params
        except Exception as e:
            print(f"S'ha produït un error carregant els paràmetres: {e}")
            return default_params

    def create_input_fields(self, parent):
        """Crea els camps d'entrada segons el model seleccionat."""
        model_type = self.model_type.get()
        expected_columns = EXPECTED_COLUMNS_ABSENCES if model_type == "absencies" else EXPECTED_COLUMNS_APROVAT

        # Main frame for all inputs
        main_frame = ttk.Frame(parent)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        parent.grid_rowconfigure(0, weight=1)

        num_cols_ui = 3 # Number of columns for input fields in the UI
        current_row, current_col_idx = 0, 0

        for col_name in expected_columns:
            definition = COLUMN_DEFINITIONS.get(col_name)
            if not definition:
                continue

            label = ttk.Label(main_frame, text=definition["prompt"])
            label.grid(row=current_row, column=current_col_idx * 2, sticky=tk.W, padx=5, pady=3)

            if "options" in definition:
                entry = ttk.Combobox(main_frame, values=definition["options"], width=17, style='TCombobox')
                entry.set(str(definition["default"]))
            else:
                entry = ttk.Entry(main_frame, width=20, style='TEntry')
                entry.insert(0, str(definition["default"]))

            entry.grid(row=current_row, column=current_col_idx * 2 + 1, sticky=tk.EW, padx=5, pady=3)
            self.entries[col_name] = entry

            current_col_idx += 1
            if current_col_idx >= num_cols_ui:
                current_col_idx = 0
                current_row += 1

        for i in range(num_cols_ui * 2): # Configure column weights for the input frame
            main_frame.grid_columnconfigure(i, weight=1 if i % 2 != 0 else 0)

    def create_param_config_ui(self, parent):
        """Crea la interfície d'usuari per configurar els paràmetres del model."""
        # Neteja els widgets existents
        for widget in parent.winfo_children():
            widget.destroy()

        model_type = self.model_type.get()
        param_prefix = "regressor__" if model_type == "absencies" else "classifier__"

        param_frame = ttk.Frame(parent)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Encapçalament
        if model_type == "absencies":
            title_text = "Configuració dels Paràmetres del Model d'Absències"
            desc_text = "Modifica els paràmetres per millorar la precisió del model de predicció d'absències."
        else:
            title_text = "Configuració dels Paràmetres del Model d'Aprovats"
            desc_text = "Modifica els paràmetres per millorar la precisió del model de predicció d'aprovats."

        ttk.Label(param_frame, text=title_text, font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)
        ttk.Label(param_frame, text=desc_text).grid(row=1, column=0, columnspan=2, pady=(0, 15), sticky=tk.W)

        # Crear camps d'entrada per cada paràmetre
        self.param_entries = {}
        row = 2
        for param, value in self.params.items():
            param_name = param.replace(param_prefix, "")
            ttk.Label(param_frame, text=f"{param_name}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)

            # Determinar el tipus d'entrada basant-se en el valor
            if isinstance(value, int):
                entry = ttk.Spinbox(param_frame, from_=1, to=1000, width=15)
                entry.set(value)
            elif isinstance(value, float):
                entry = ttk.Spinbox(param_frame, from_=0.01, to=1.0, increment=0.01, width=15)
                entry.set(f"{value:.2f}")
            else:
                entry = ttk.Entry(param_frame, width=15)
                entry.insert(0, str(value))

            entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
            self.param_entries[param] = entry
            row += 1

        # Descripció dels paràmetres
        ttk.Label(param_frame, text="Descripció dels paràmetres:", font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, pady=(15, 5), sticky=tk.W)
        row += 1

        if model_type == "absencies":
            descriptions = {
                "n_estimators": "Nombre d'arbres en el bosc. Més arbres = més precisió però més temps d'entrenament.",
                "learning_rate": "Taxa d'aprenentatge. Valors més baixos requereixen més arbres però poden donar millors resultats.",
                "max_depth": "Profunditat màxima de cada arbre. Més profunditat pot causar sobreentrenament.",
                "min_samples_leaf": "Nombre mínim de mostres necessàries en una fulla. Valors més alts eviten sobreentrenament.",
                "subsample": "Fracció de mostres per entrenar cada arbre. Valors < 1.0 poden evitar sobreentrenament."
            }
        else:
            descriptions = {
                "n_estimators": "Nombre d'arbres en el bosc. Més arbres = més precisió però més temps d'entrenament.",
                "learning_rate": "Taxa d'aprenentatge. Valors més baixos requereixen més arbres però poden donar millors resultats.",
                "max_depth": "Profunditat màxima de cada arbre. Més profunditat pot causar sobreentrenament.",
                "min_samples_leaf": "Nombre mínim de mostres necessàries en una fulla. Valors més alts eviten sobreentrenament.",
                "subsample": "Fracció de mostres per entrenar cada arbre. Valors < 1.0 poden evitar sobreentrenament."
            }

        for param, desc in descriptions.items():
            ttk.Label(param_frame, text=f"• {param}: {desc}", wraplength=500, justify=tk.LEFT).grid(
                row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
            row += 1

        # Botons per guardar, recarregar i re-entrenar
        button_frame = ttk.Frame(param_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=15)

        ttk.Button(button_frame, text="Guardar Paràmetres", command=self.save_params).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Recarregar Model", command=self.reload_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Re-entrenar Model", command=self.retrain_model).pack(side=tk.LEFT, padx=5)

    def create_evaluation_ui(self, parent):
        """Crea la interfície d'usuari per a l'avaluació del model."""
        try:
            evaluation_frame = ttk.Frame(parent)
            evaluation_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Encapçalament
            ttk.Label(evaluation_frame, text="Avaluació del Model", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=10)
            ttk.Label(evaluation_frame, text="Analitza la precisió del model amb diferents mètriques.").pack(anchor=tk.W, pady=(0, 10))

            # Botó per executar l'avaluació
            ttk.Button(evaluation_frame, text="Executar Avaluació", command=self.run_model_evaluation).pack(pady=10)

            # Marc per a les mètriques
            self.metrics_frame = ttk.LabelFrame(evaluation_frame, text="Mètriques d'Avaluació")
            self.metrics_frame.pack(fill=tk.X, expand=False, pady=10)

            # Espai per al gràfic
            self.plot_frame = ttk.Frame(evaluation_frame)
            self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        except ImportError as e:
            ttk.Label(parent, text=f"Error: No s'han pogut importar les biblioteques necessàries.\n{e}",
                     font=("Arial", 10)).pack(pady=20)
            ttk.Label(parent, text="Assegureu-vos de tenir instal·lades matplotlib i scikit-learn.",
                     font=("Arial", 10)).pack()

    def run_model_evaluation(self):
        """Executa l'avaluació del model i mostra els resultats."""
        try:
            # Netejar els widgets anteriors
            for widget in self.metrics_frame.winfo_children():
                widget.destroy()
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Carregar les dades
            if not os.path.exists(self.data_path):
                messagebox.showerror("Error", f"No s'ha trobat el fitxer de dades a {self.data_path}")
                return

            model_type = self.model_type.get()
            data = pd.read_csv(self.data_path)

            # Preparar les dades segons el model
            if model_type == "absencies":
                X = data.drop(columns=['absences'])
                y = data['absences']

                # Dividir les dades en conjunts d'entrenament i test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Fer prediccions amb el model carregat
                y_pred = self.model.predict(X_test)

                # Calcular mètriques
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                # Validació creuada
                cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_squared_error')
                cv_rmse = (-cv_scores)**0.5

                # Mostrar mètriques
                metrics_data = [
                    ("Error Absolut Mitjà (MAE)", f"{mae:.4f}"),
                    ("Error Quadràtic Mitjà (MSE)", f"{mse:.4f}"),
                    ("Arrel de l'Error Quadràtic Mitjà (RMSE)", f"{rmse:.4f}"),
                    ("Coeficient de Determinació (R²)", f"{r2:.4f}"),
                    ("RMSE Validació Creuada (5-fold)", f"{cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
                ]

                # Crear gràfic
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                fig.set_facecolor('#2B2B2B')

                # Gràfic 1: Valors reals vs. prediccions
                ax1.scatter(y_test, y_pred, color='#5DA5DA', alpha=0.6)
                ax1.plot([0, max(y_test.max(), y_pred.max())], [0, max(y_test.max(), y_pred.max())], 'r--')
                ax1.set_xlabel('Valors Reals', color='white')
                ax1.set_ylabel('Prediccions', color='white')
                ax1.set_title('Prediccions vs. Valors Reals', color='white')
                ax1.tick_params(colors='white')

                # Gràfic 2: Distribució de l'error
                errors = y_test - y_pred
                ax2.hist(errors, bins=20, color='#60BD68', alpha=0.7)
                ax2.axvline(x=0, color='r', linestyle='--')
                ax2.set_xlabel('Error de Predicció', color='white')
                ax2.set_ylabel('Freqüència', color='white')
                ax2.set_title('Distribució de l\'Error', color='white')
                ax2.tick_params(colors='white')

                # Ajustar l'estil per a fons fosc
                for ax in [ax1, ax2]:
                    ax.set_facecolor('#383838')
                    for spine in ax.spines.values():
                        spine.set_color('white')

                plt.tight_layout()

                # Mostrar les mètriques
                for i, (metric, value) in enumerate(metrics_data):
                    ttk.Label(self.metrics_frame, text=f"{metric}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=3)
                    ttk.Label(self.metrics_frame, text=value, font=("Arial", 9, "bold")).grid(row=i, column=1, sticky=tk.W, padx=5, pady=3)

                # Mostrar el gràfic a Tkinter
                canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            else:  # Model d'aprovats
                # Crear la variable objectiu (aprovat si G3 >= 10)
                data['aprovat'] = (data['G3'] >= 10).astype(int)

                # No utilitzar G3 com a característica perquè és la base del que volem predir
                X = data.drop(columns=['aprovat', 'G3'])
                y = data['aprovat']

                # Dividir les dades en conjunts d'entrenament i test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Fer prediccions amb el model carregat
                y_pred = self.model.predict(X_test)

                # Calcular mètriques de classificació
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Validació creuada
                cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')

                # Mostrar mètriques
                metrics_data = [
                    ("Exactitud (Accuracy)", f"{accuracy:.4f}"),
                    ("Precisió (Precision)", f"{precision:.4f}"),
                    ("Sensibilitat (Recall)", f"{recall:.4f}"),
                    ("Puntuació F1", f"{f1:.4f}"),
                    ("Exactitud Validació Creuada (5-fold)", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                ]

                # Crear frame principal per contenir els gràfics i les explicacions
                main_eval_frame = ttk.Frame(self.plot_frame)
                main_eval_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                # Mostrar les mètriques
                for i, (metric, value) in enumerate(metrics_data):
                    ttk.Label(self.metrics_frame, text=f"{metric}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=3)
                    ttk.Label(self.metrics_frame, text=value, font=("Arial", 9, "bold")).grid(row=i, column=1, sticky=tk.W, padx=5, pady=3)

                # Frame per al primer gràfic (matriu de confusió) i la seva explicació
                confusion_frame = ttk.Frame(main_eval_frame)
                confusion_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

                # Frame per al segon gràfic (corba ROC) i la seva explicació
                roc_frame = ttk.Frame(main_eval_frame)
                roc_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

                main_eval_frame.grid_rowconfigure(0, weight=1)
                main_eval_frame.grid_rowconfigure(1, weight=1)
                main_eval_frame.grid_columnconfigure(0, weight=1)

                # Configuració del frame de la matriu de confusió
                confusion_fig = plt.figure(figsize=(3.5, 3.5))  # Gràfic més petit
                confusion_fig.set_facecolor('#2B2B2B')
                conf_ax = confusion_fig.add_subplot(111)

                # Dibuixar la matriu de confusió
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=conf_ax)
                conf_ax.set_xlabel('Predit', color='white', fontsize=8)  # Text més petit
                conf_ax.set_ylabel('Real', color='white', fontsize=8)  # Text més petit
                conf_ax.set_title('Matriu de Confusió', color='white', fontsize=10)  # Títol més petit
                conf_ax.tick_params(colors='white', labelsize=8)  # Text més petit
                conf_ax.set_facecolor('#383838')

                # Añadir etiquetas a los ejes para clarificar la matriz de confusión
                conf_ax.set_xticks([0.5, 1.5])
                conf_ax.set_xticklabels(['No (0)', 'Sí (1)'])  # Etiquetes més breus
                conf_ax.set_yticks([0.5, 1.5])
                conf_ax.set_yticklabels(['No (0)', 'Sí (1)'])  # Etiquetes més breus

                for spine in conf_ax.spines.values():
                    spine.set_color('white')

                confusion_fig.tight_layout()  # Optimitzar espai

                # Dividir el frame de la matriu de confusió en dues columnes
                confusion_frame.grid_columnconfigure(0, weight=1)  # Gràfic
                confusion_frame.grid_columnconfigure(1, weight=2)  # Explicació

                # Canvas per a la matriu de confusió
                confusion_canvas = FigureCanvasTkAgg(confusion_fig, master=confusion_frame)
                confusion_canvas.draw()
                confusion_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

                # Explicació de la matriu de confusió
                confusion_explanation_frame = ttk.Frame(confusion_frame)
                confusion_explanation_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

                ttk.Label(confusion_explanation_frame, text="Interpretació:",
                         font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 3))

                explanation_text = ("• Sup. esq. (TN): V. Negatius - No aprovat predit correctament\n"
                               "• Sup. dret. (FP): F. Positius - Aprovat predit incorrectament\n"
                               "• Inf. esq. (FN): F. Negatius - No aprovat predit incorrectament\n"
                               "• Inf. dret. (TP): V. Positius - Aprovat predit correctament")

                ttk.Label(confusion_explanation_frame, text=explanation_text, justify=tk.LEFT, font=("Arial", 8)).pack(anchor=tk.W)

                # Configuració del frame de la corba ROC
                if hasattr(self.model, "predict_proba"):
                    roc_fig = plt.figure(figsize=(3.5, 3.5))  # Gràfic més petit
                    roc_fig.set_facecolor('#2B2B2B')
                    roc_ax = roc_fig.add_subplot(111)

                    # Dibuixar la corba ROC
                    y_scores = self.model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_scores)
                    roc_auc = auc(fpr, tpr)

                    roc_ax.plot(fpr, tpr, color='#5DA5DA', lw=2, label=f'AUC = {roc_auc:.2f}')
                    roc_ax.plot([0, 1], [0, 1], 'r--', label='Base (50%)')  # Etiqueta més breu
                    roc_ax.set_xlim([0.0, 1.0])
                    roc_ax.set_ylim([0.0, 1.05])
                    roc_ax.set_xlabel('Taxa FP', color='white', fontsize=8)  # Text més breu i petit
                    roc_ax.set_ylabel('Taxa VP', color='white', fontsize=8)  # Text més breu i petit
                    roc_ax.set_title('Corba ROC', color='white', fontsize=10)  # Títol més petit
                    roc_ax.legend(loc="lower right", fontsize=8)  # Llegenda més petita
                    roc_ax.tick_params(colors='white', labelsize=8)  # Text més petit
                    roc_ax.set_facecolor('#383838')

                    for spine in roc_ax.spines.values():
                        spine.set_color('white')

                    roc_fig.tight_layout()  # Optimitzar espai

                    # Dividir el frame de la corba ROC en dues columnes
                    roc_frame.grid_columnconfigure(0, weight=1)  # Gràfic
                    roc_frame.grid_columnconfigure(1, weight=2)  # Explicació

                    # Canvas per a la corba ROC
                    roc_canvas = FigureCanvasTkAgg(roc_fig, master=roc_frame)
                    roc_canvas.draw()
                    roc_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

                    # Explicació de la corba ROC
                    roc_explanation_frame = ttk.Frame(roc_frame)
                    roc_explanation_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

                    ttk.Label(roc_explanation_frame, text="Interpretació:",
                             font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 3))

                    roc_quality = ("molt bo" if roc_auc > 0.9 else
                                  "bo" if roc_auc > 0.8 else
                                  "acceptable" if roc_auc > 0.7 else
                                  "millorable")

                    roc_text = ("• Mostra rendiment del classificador a diferents llindars\n"
                           "• Eix X: Taxa de Falsos Positius\n"
                           "• Eix Y: Taxa de Verdaders Positius (Recall)\n"
                           "• Línia vermella: classificador aleatori (50%)\n"
                           "• Millor rendiment: corba cap a l'angle superior esquerre\n"
                           f"• AUC = {roc_auc:.2f}: rendiment {roc_quality}")

                    ttk.Label(roc_explanation_frame, text=roc_text, justify=tk.LEFT, font=("Arial", 8)).pack(anchor=tk.W)

                else:
                    ttk.Label(roc_frame, text="No s'ha pogut generar la corba ROC.\nEl model no proporciona probabilitats.").pack(pady=20)

        except Exception as e:
            messagebox.showerror("Error", f"S'ha produït un error durant l'avaluació: {e}")
            print(f"Excepció detallada: {e}")
            import traceback
            traceback.print_exc()

    def save_params(self):
        """Guarda els paràmetres actuals al fitxer de paràmetres."""
        model_type = self.model_type.get()
        params_path = self.params_paths[model_type]

        try:
            updated_params = {}
            param_prefix = "regressor__" if model_type == "absencies" else "classifier__"

            for param, entry in self.param_entries.items():
                value = entry.get()
                # Convertir al tipus correcte
                if "learning_rate" in param or "subsample" in param:
                    value = float(value)
                elif "n_estimators" in param or "max_depth" in param or "min_samples_leaf" in param:
                    value = int(value)
                updated_params[param] = value

            # Guardar al fitxer
            with open(params_path, 'w') as f:
                json.dump(updated_params, f, indent=2)

            self.params = updated_params
            messagebox.showinfo("Èxit", f"Paràmetres del model {model_type} guardats correctament")
        except Exception as e:
            messagebox.showerror("Error", f"No s'han pogut guardar els paràmetres: {e}")

    def reload_model(self):
        """Recarrega el model amb els paràmetres actuals."""
        try:
            model_type = self.model_type.get()
            self.model = joblib.load(self.model_paths[model_type])
            if self.model:
                messagebox.showinfo("Èxit", f"Model {model_type} recarregat correctament")
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut recarregar el model: {e}")

    def retrain_model(self):
        """Re-entrena el model amb els paràmetres actuals."""
        try:
            model_type = self.model_type.get()

            # Primer guardar els nous paràmetres
            self.save_params()

            # Determinar la ruta al script d'entrenament i a les dades
            model_dir = "../2-Models-Predictius"
            if model_type == "absencies":
                model_script_path = os.path.abspath(f"{model_dir}/Absencies/model_absencies.py")
                output_dir = os.path.dirname(self.model_paths["absencies"])
            else:
                model_script_path = os.path.abspath(f"{model_dir}/Aprovat/model_aprovat.py")
                output_dir = os.path.dirname(self.model_paths["aprovat"])

            # Busquem el fitxer csv a la carpeta arrel del projecte
            data_path = self.data_path

            if not os.path.exists(model_script_path):
                messagebox.showerror("Error", f"No s'ha trobat l'script d'entrenament a {model_script_path}")
                return

            if not os.path.exists(data_path):
                messagebox.showerror("Error", f"No s'ha trobat el fitxer de dades a {data_path}")
                return

            # Executar l'script d'entrenament en un procés separat
            command = [sys.executable, model_script_path, "--data", data_path, "--output-dir", output_dir]

            # Variable per controlar si el procés s'ha cancel·lat
            self.process_cancelled = False

            # Mostrar un diàleg amb la barra de progrés
            progress_window = tk.Toplevel(self.master)
            progress_window.title("Entrenant Model")
            progress_window.geometry("400x150")
            progress_window.configure(bg='#2B2B2B')

            progress_window.transient(self.master)
            progress_window.grab_set()

            ttk.Label(progress_window, text=f"S'està entrenant el model {model_type} amb els nous paràmetres...",
                     font=("Arial", 10)).pack(pady=20)

            progress = ttk.Progressbar(progress_window, mode="indeterminate")
            progress.pack(fill=tk.X, padx=20, pady=10)
            progress.start()

            # Funció per cancel·lar el procés
            def cancel_training():
                if hasattr(self, 'process') and self.process:
                    try:
                        self.process_cancelled = True
                        # En Windows usem taskkill per assegurar-nos que s'eliminen els processos fills també
                        if sys.platform == 'win32':
                            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                        else:
                            self.process.terminate()
                    except Exception as e:
                        print(f"Error al cancel·lar el procés: {e}")
                progress_window.destroy()

            # Botó per cancel·lar
            cancel_button = ttk.Button(progress_window, text="Cancel·lar",
                                      command=cancel_training)
            cancel_button.pack(pady=10)

            # Protocol per gestionar el tancament de la finestra
            progress_window.protocol("WM_DELETE_WINDOW", cancel_training)

            # Actualitzar la UI abans d'iniciar el procés
            self.master.update()

            try:
                # Executar el procés en mode no bloquejant
                self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                              bufsize=1, universal_newlines=True)

                # Comprovar periòdicament si el procés ha acabat
                def check_process():
                    if not hasattr(self, 'process') or self.process is None:
                        return

                    if self.process.poll() is None:  # Encara en execució
                        if hasattr(self, 'process_cancelled') and self.process_cancelled:
                            return
                        self.master.after(100, check_process)
                    else:
                        if hasattr(progress_window, 'winfo_exists') and progress_window.winfo_exists():
                            progress_window.destroy()

                        if self.process_cancelled:
                            messagebox.showinfo("Cancel·lat", "Entrenament cancel·lat per l'usuari")
                            return

                        if self.process.returncode == 0:
                            # Obtenir la sortida estàndard per mostrar informació útil
                            stdout, stderr = self.process.communicate()
                            messagebox.showinfo("Èxit", f"Model {model_type} entrenat correctament")

                            # Guardar els logs per a depuració
                            with open(os.path.join(output_dir, "training_log.txt"), "w") as log_file:
                                log_file.write("STDOUT:\n" + stdout + "\n\nSTDERR:\n" + stderr)

                            # Recarregar el model
                            self.reload_model()
                        else:
                            # Llegir l'error
                            stdout, stderr = self.process.communicate()
                            messagebox.showerror("Error", f"Error entrenant el model. Consulta el fitxer de log per més detalls.")

                            # Guardar l'error en un log
                            with open(os.path.join(output_dir, "error_log.txt"), "w") as log_file:
                                log_file.write("STDOUT:\n" + stdout + "\n\nSTDERR:\n" + stderr)

                # Iniciar la comprovació
                self.master.after(100, check_process)

            except Exception as e:
                if hasattr(progress_window, 'winfo_exists') and progress_window.winfo_exists():
                    progress_window.destroy()
                messagebox.showerror("Error", f"No s'ha pogut iniciar l'entrenament: {e}")

        except Exception as e:
            messagebox.showerror("Error", f"Error general: {e}")
            import traceback
            traceback.print_exc()

    def predict(self):
        """Realitza la predicció segons el model seleccionat."""
        model_type = self.model_type.get()

        if not self.model:
            messagebox.showerror("Error", "El model no està carregat.")
            return

        expected_columns = EXPECTED_COLUMNS_ABSENCES if model_type == "absencies" else EXPECTED_COLUMNS_APROVAT
        input_data = {}

        try:
            for col_name, entry_widget in self.entries.items():
                val_str = entry_widget.get()
                definition = COLUMN_DEFINITIONS[col_name]
                expected_type = definition["type"]

                if not val_str and definition["default"] is not None: # Handle empty input by using default
                    val_str = str(definition["default"])

                if expected_type == int:
                    # Basic validation for integer fields
                    val_int = int(val_str)
                    if col_name in ['Medu', 'Fedu'] and not (0 <= val_int <= 4):
                        raise ValueError(f"{col_name}: El valor ha de ser entre 0 i 4.")
                    if col_name in ['traveltime', 'studytime'] and not (1 <= val_int <= 4):
                        raise ValueError(f"{col_name}: El valor ha de ser entre 1 i 4.")
                    if col_name in ['famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health'] and not (1 <= val_int <= 5):
                        raise ValueError(f"{col_name}: El valor ha de ser entre 1 i 5.")
                    if col_name in ['G1', 'G2', 'G3', 'age', 'failures', 'absences'] and val_int < 0:
                        raise ValueError(f"{col_name}: El valor no pot ser negatiu.")
                    if col_name in ['G1', 'G2', 'G3'] and val_int > 20:
                        raise ValueError(f"{col_name}: La nota ha de ser entre 0 i 20.")
                    input_data[col_name] = val_int
                else: # str
                    if "options" in definition and val_str not in definition["options"]:
                         raise ValueError(f"{col_name}: Valor \"{val_str}\" no és una opció vàlida. Opcions: {definition['options']}")
                    input_data[col_name] = val_str

            # Crear un DataFrame amb una sola fila per a la predicció
            df_to_predict = pd.DataFrame([input_data])

            # Assegurar-se que el DataFrame té totes les columnes esperades pel model
            for col in expected_columns:
                if col not in df_to_predict.columns:
                    df_to_predict[col] = COLUMN_DEFINITIONS[col]["default"]

            # Reordenar les columnes segons l'ordre esperat pel model
            df_to_predict = df_to_predict[expected_columns]

            # Debug info
            print(f"Dades per a la predicció ({model_type}):")
            print(df_to_predict)

            # Fer la predicció
            prediction = self.model.predict(df_to_predict)

            if model_type == "absencies":
                result_text = f"El model prediu aproximadament {prediction[0]:.2f} absències escolars al llarg d'un trimestre.\n"
            else:
                # Obtenir la probabilitat d'aprovar (només per al model d'aprovats)
                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(df_to_predict)
                    prob_pass = probabilities[0][1] * 100

                    if prediction[0] == 1:
                        result_text = f"El model prediu que l'estudiant APROVARÀ (probabilitat: {prob_pass:.2f}%).\n"
                    else:
                        result_text = f"El model prediu que l'estudiant NO APROVARÀ (probabilitat d'aprovar: {prob_pass:.2f}%).\n"
                else:
                    if prediction[0] == 1:
                        result_text = "El model prediu que l'estudiant APROVARÀ.\n"
                    else:
                        result_text = "El model prediu que l'estudiant NO APROVARÀ.\n"

            self.result_label.config(text=result_text)

        except ValueError as ve:
            messagebox.showerror("Error de validació", f"Si us plau, revisa els valors introduïts.\n{ve}")
        except Exception as e:
            messagebox.showerror("Error en la predicció", f"S'ha produït un error: {e}")
            print(f"Excepció detallada: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()

