import joblib
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox  # Removed filedialog
import json
import os
import sys
import subprocess

# Columnes esperades pel model d'absències (totes menys 'absences' que és el target, però G3 sí s'inclou com a feature)
EXPECTED_COLUMNS_ABSENCES = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
    'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'G1', 'G2', 'G3'
]

# Diccionari amb prompts, tipus, opcions vàlides (opcional) i valors per defecte per a cada columna
COLUMN_DEFINITIONS_ABSENCES = {
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
    'G3': {"prompt": "Nota G3 (0-20):", "type": int, "default": 10}
}

# Paràmetres configurables del model
DEFAULT_MODEL_PARAMS = {
    "regressor__n_estimators": 300,
    "regressor__learning_rate": 0.05,
    "regressor__max_depth": 8,
    "regressor__min_samples_leaf": 3,
    "regressor__subsample": 0.9
}

class AbsencesPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Predicció d'Absències")
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

        # Rutes als fitxers (camins absoluts per evitar ambigüitats)
        model_dir = os.path.abspath("../2-Models-Predictius/Absencies")
        self.model_path = os.path.join(model_dir, "decision_tree_absences.joblib")
        self.params_path = os.path.join(model_dir, "params_absencies.json")

        # Ruta al fitxer CSV amb les dades
        self.data_path = os.path.abspath("../portuguese_hs_students.csv")

        # Crear un notebook amb pestanyes
        self.notebook = ttk.Notebook(master)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        # Pestanya per a la predicció d'absències
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="Predicció d'Absències")

        # Pestanya per a la configuració del model
        self.config_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text="Configuració del Model")

        # Pestanya per a l'avaluació del model
        self.evaluation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.evaluation_tab, text="Avaluació del Model")

        # Carregar model i paràmetres
        self.model = self.load_model()
        self.params = self.load_params()

        # Configurar pestanya de predicció
        self.entries = {}
        self.create_input_fields(self.prediction_tab)

        self.predict_button = ttk.Button(self.prediction_tab, text="Predir Absències", command=self.predict_absences)
        self.predict_button.grid(row=1, column=0, pady=15, padx=10, sticky="ew")

        self.result_label = ttk.Label(self.prediction_tab, text="Resultat de la predicció:", font=("Arial", 11, "bold"))
        self.result_label.grid(row=2, column=0, pady=10, padx=10)

        # Configurar pestanya de configuració del model
        self.param_entries = {}
        self.create_param_config_ui(self.config_tab)

        # Configurar pestanya d'avaluació del model
        self.create_evaluation_ui(self.evaluation_tab)

    def load_model(self):
        try:
            model = joblib.load(self.model_path)
            print(f"Model d'absències carregat des de: {self.model_path}")
            return model
        except FileNotFoundError:
            messagebox.showerror("Error", f"No s'ha trobat el fitxer del model a {self.model_path}")
            return None
        except Exception as e:
            messagebox.showerror("Error", f"S'ha produït un error carregant el model: {e}")
            return None

    def load_params(self):
        try:
            with open(self.params_path, 'r') as f:
                params = json.load(f)
                print(f"Paràmetres del model carregats des de: {self.params_path}")
                return params
        except FileNotFoundError:
            print(f"No s'ha trobat el fitxer de paràmetres a {self.params_path}, s'utilitzaran paràmetres per defecte")
            return DEFAULT_MODEL_PARAMS
        except json.JSONDecodeError:
            print(f"Error de format al fitxer de paràmetres, s'utilitzaran paràmetres per defecte")
            return DEFAULT_MODEL_PARAMS
        except Exception as e:
            print(f"S'ha produït un error carregant els paràmetres: {e}")
            return DEFAULT_MODEL_PARAMS

    def create_input_fields(self, parent):
        # Main frame for all inputs
        main_frame = ttk.Frame(parent)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        parent.grid_rowconfigure(0, weight=1)

        num_cols_ui = 3 # Number of columns for input fields in the UI
        current_row, current_col_idx = 0, 0

        for col_name in EXPECTED_COLUMNS_ABSENCES:  # Removed unused 'i' from enumerate
            definition = COLUMN_DEFINITIONS_ABSENCES.get(col_name)
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
        param_frame = ttk.Frame(parent)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Encapçalament
        ttk.Label(param_frame, text="Configuració dels Paràmetres del Model", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)
        ttk.Label(param_frame, text="Modifica els paràmetres per millorar la precisió del model.").grid(row=1, column=0, columnspan=2, pady=(0, 15), sticky=tk.W)

        # Crear camps d'entrada per cada paràmetre
        row = 2
        for param, value in self.params.items():
            param_name = param.replace("regressor__", "")
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
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.model_selection import train_test_split, cross_val_score

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
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.model_selection import train_test_split, cross_val_score

            # Netejar els widgets anteriors
            for widget in self.metrics_frame.winfo_children():
                widget.destroy()
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Carregar les dades
            if not os.path.exists(self.data_path):
                messagebox.showerror("Error", f"No s'ha trobat el fitxer de dades a {self.data_path}")
                return

            data = pd.read_csv(self.data_path)

            # Preparar les dades
            X = data.drop(columns=['absences'])
            y = data['absences']

            # Dividir les dades en conjunts d'entrenament i test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Fer prediccions amb el model carregat
            y_pred = self.model.predict(X_test)

            # Calcular mètriques
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            # Calcular RMSE manualment en lloc d'utilitzar squared=False
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

            for i, (metric, value) in enumerate(metrics_data):
                ttk.Label(self.metrics_frame, text=f"{metric}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=3)
                ttk.Label(self.metrics_frame, text=value, font=("Arial", 9, "bold")).grid(row=i, column=1, sticky=tk.W, padx=5, pady=3)

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

            # Mostrar el gràfic a Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"S'ha produït un error durant l'avaluació: {e}")
            print(f"Excepció detallada: {e}")
            import traceback
            traceback.print_exc()

    def save_params(self):
        """Guarda els paràmetres actuals al fitxer de paràmetres."""
        try:
            updated_params = {}
            for param, entry in self.param_entries.items():
                value = entry.get()
                # Convertir al tipus correcte
                if "learning_rate" in param or "subsample" in param:
                    value = float(value)
                elif "n_estimators" in param or "max_depth" in param or "min_samples_leaf" in param:
                    value = int(value)
                updated_params[param] = value

            # Guardar al fitxer
            with open(self.params_path, 'w') as f:
                json.dump(updated_params, f, indent=2)

            self.params = updated_params
            messagebox.showinfo("Èxit", "Paràmetres guardats correctament")
        except Exception as e:
            messagebox.showerror("Error", f"No s'han pogut guardar els paràmetres: {e}")

    def reload_model(self):
        """Recarrega el model amb els paràmetres actuals."""
        try:
            self.model = self.load_model()
            if self.model:
                messagebox.showinfo("Èxit", "Model recarregat correctament")
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut recarregar el model: {e}")

    def retrain_model(self):
        """Re-entrena el model amb els paràmetres actuals."""
        try:
            # Primer guardar els nous paràmetres
            self.save_params()

            # Determinar la ruta al script d'entrenament i a les dades
            model_script_path = os.path.abspath("../2-Models-Predictius/Absencies/model_absencies.py")
            # Busquem el fitxer csv a la carpeta arrel del projecte
            data_path = os.path.abspath("../portuguese_hs_students.csv")

            if not os.path.exists(model_script_path):
                messagebox.showerror("Error", f"No s'ha trobat l'script d'entrenament a {model_script_path}")
                return

            if not os.path.exists(data_path):
                messagebox.showerror("Error", f"No s'ha trobat el fitxer de dades a {data_path}")
                return

            # Executar l'script d'entrenament en un procés separat
            # Utilitzem --output-dir en lloc de --model-dir ja que és l'argument que accepta el script
            command = [sys.executable, model_script_path, "--data", data_path, "--output-dir", os.path.dirname(self.model_path)]

            # Variable per controlar si el procés s'ha cancel·lat
            self.process_cancelled = False

            # Mostrar un diàleg amb la barra de progrés
            progress_window = tk.Toplevel(self.master)
            progress_window.title("Entrenant Model")
            progress_window.geometry("400x150")
            progress_window.configure(bg='#2B2B2B')
            progress_window.transient(self.master)
            progress_window.grab_set()

            ttk.Label(progress_window, text="S'està entrenant el model amb els nous paràmetres...",
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
                            messagebox.showinfo("Èxit", "Model entrenat correctament")

                            # Guardar els logs per a depuració
                            with open(os.path.join(os.path.dirname(self.model_path), "training_log.txt"), "w") as log_file:
                                log_file.write("STDOUT:\n" + stdout + "\n\nSTDERR:\n" + stderr)

                            # Recarregar el model
                            self.model = self.load_model()
                        else:
                            # Llegir l'error
                            stdout, stderr = self.process.communicate()
                            messagebox.showerror("Error", f"Error entrenant el model. Consulta el fitxer de log per més detalls.")

                            # Guardar l'error en un log
                            with open(os.path.join(os.path.dirname(self.model_path), "error_log.txt"), "w") as log_file:
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

    def predict_absences(self):
        if not self.model:
            messagebox.showerror("Error", "El model no està carregat.")
            return

        input_data = {}
        try:
            for col_name, entry_widget in self.entries.items():
                val_str = entry_widget.get()
                definition = COLUMN_DEFINITIONS_ABSENCES[col_name]
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
                    if col_name in ['G1', 'G2', 'G3', 'age', 'failures'] and val_int < 0:
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
            for col in EXPECTED_COLUMNS_ABSENCES:
                if col not in df_to_predict.columns:
                    df_to_predict[col] = COLUMN_DEFINITIONS_ABSENCES[col]["default"]

            # Reordenar les columnes segons l'ordre esperat pel model
            df_to_predict = df_to_predict[EXPECTED_COLUMNS_ABSENCES]

            # Debug info
            print("Dades per a la predicció:")
            print(df_to_predict)

            # Fer la predicció
            prediction = self.model.predict(df_to_predict)

            result_text = f"El model prediu aproximadament {prediction[0]:.2f} absències escolars al llarg d'un trimestre.\n"
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
    app = AbsencesPredictorApp(root)
    root.mainloop()

