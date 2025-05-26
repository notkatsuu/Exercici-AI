import joblib
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

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

class AbsencesPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Predicció d'Absències")
        master.configure(bg='#2B2B2B') # Dark mode background
        master.geometry("850x650") # Adjust window size

        # Style for widgets
        style = ttk.Style()
        style.theme_use('clam') 
        style.configure('.', background='#2B2B2B', foreground='white')
        style.configure('TLabel', background='#2B2B2B', foreground='white', padding=3, font=('Arial', 9))
        style.configure('TButton', background='#4A4A4A', foreground='white', padding=5, borderwidth=0, font=('Arial', 10, 'bold'))
        style.map('TButton', background=[('active', '#6A6A6A')])
        style.configure('TEntry', fieldbackground='#3C3F41', foreground='white', insertcolor='white', borderwidth=1, relief='flat', padding=3, font=('Arial', 9))
        style.configure('TCombobox', fieldbackground='#3C3F41', foreground='white', selectbackground='#3C3F41', selectforeground='white', arrowcolor='white', padding=3, font=('Arial', 9))
        master.option_add('*TCombobox*Listbox.background', '#3C3F41')
        master.option_add('*TCombobox*Listbox.foreground', 'white')
        master.option_add('*TCombobox*Listbox.selectBackground', '#4A4A4A')
        master.option_add('*TCombobox*Listbox.selectForeground', 'white')

        self.model_path = '../2-Models-Predictius/decision_tree_absences.joblib'
        self.model = self.load_model()

        self.entries = {}
        self.create_input_fields()

        self.predict_button = ttk.Button(master, text="Predir Absències", command=self.predict_absences)
        # Place button below the input fields frame
        self.predict_button.grid(row=1, column=0, pady=15, padx=10, sticky="ew")

        self.result_label = ttk.Label(master, text="Resultat de la predicció:", font=("Arial", 11, "bold"))
        self.result_label.grid(row=2, column=0, pady=10, padx=10)
        
        # Configure master grid
        master.grid_rowconfigure(0, weight=0) # Input fields frame
        master.grid_rowconfigure(1, weight=0) # Button
        master.grid_rowconfigure(2, weight=0) # Result
        master.grid_columnconfigure(0, weight=1)

    def load_model(self):
        try:
            model = joblib.load(self.model_path)
            print(f"Model d'absències carregat des de: {self.model_path}")
            return model
        except FileNotFoundError:
            messagebox.showerror("Error", f"No s'ha trobat el fitxer del model a {self.model_path}")
            self.master.quit()
            return None
        except Exception as e:
            messagebox.showerror("Error", f"S'ha produït un error carregant el model: {e}")
            self.master.quit()
            return None

    def create_input_fields(self):
        # Main frame for all inputs
        main_frame = ttk.Frame(self.master, style='TFrame')
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.master.grid_rowconfigure(0, weight=1) # Allow main_frame to expand if needed

        num_cols_ui = 3 # Number of columns for input fields in the UI
        current_row, current_col_idx = 0, 0

        for i, col_name in enumerate(EXPECTED_COLUMNS_ABSENCES):
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
                    input_data[col_name] = [val_int]
                else: # str
                    if "options" in definition and val_str not in definition["options"]:
                         raise ValueError(f"{col_name}: Valor \"{val_str}\" no és una opció vàlida. Opcions: {definition['options']}")
                    input_data[col_name] = [val_str]
            
            df_to_predict = pd.DataFrame(input_data)
            df_to_predict = df_to_predict[EXPECTED_COLUMNS_ABSENCES]

            prediction = self.model.predict(df_to_predict)
            result_text = f"El model prediu aproximadament {prediction[0]:.2f} absències escolars.\n"
            result_text += "(El període exacte cobert per aquesta predicció depèn de la definició original de la columna 'absences'.)"
            self.result_label.config(text=result_text)

        except ValueError as ve:
            messagebox.showerror("Error de validació", f"Si us plau, revisa els valors introduïts.\n{ve}")
        except Exception as e:
            messagebox.showerror("Error en la predicció", f"S'ha produït un error: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = AbsencesPredictorApp(root)
    root.mainloop()
