import csv
import matplotlib.pyplot as plt
import numpy as np
import os

# Definim la correspondència de valors famrel a descripció
famrel_dict = {
    '1': 'Molt dolenta',
    '2': 'Dolenta',
    '3': 'Bona',
    '4': 'Molt bona',
    '5': 'Excel·lent'
}

# Inicialitzem el recompte per cada categoria
counts = {k: 0 for k in famrel_dict.keys()}

total = 0

# Funció per llegir el CSV i validar que existeix
def read_csv_data():
    try:
        # Nova ruta: accés al fitxer CSV des de la carpeta arrel del projecte
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'portuguese_hs_students.csv')
        with open(csv_path, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)  # Converteix a llista per facilitar múltiples lectures
    except FileNotFoundError:
        print(f"Error: No s'ha trobat el fitxer 'portuguese_hs_students.csv'")
        return []
    except Exception as e:
        print(f"Error en llegir el fitxer CSV: {e}")
        return []

# Llegeix les dades una vegada
data = read_csv_data()

# Compta les relacions familiars
for row in data:
    famrel = row['famrel'].strip()
    if famrel in counts:
        counts[famrel] += 1
        total += 1

print("Classificació de la relació familiar dels alumnes:")
for k in sorted(famrel_dict.keys()):
    print(f"{famrel_dict[k]} ({k}): {counts[k]} alumnes")

# Gràfic de barres
labels = [famrel_dict[k] for k in sorted(famrel_dict.keys())]
values = [counts[k] for k in sorted(famrel_dict.keys())]
plt.figure(figsize=(8,5))
plt.bar(labels, values, color='skyblue')
plt.xlabel('Qualitat de la relació familiar')
plt.ylabel('Nombre d\'alumnes')
plt.title('Distribució de la qualitat de la relació familiar')
plt.tight_layout()
plt.savefig('famrel_plot.png')
plt.show()

# --- FUNCIONS AUXILIARS ---
def safe_boxplot(data, labels, xlabel, ylabel, title):
    filtered = [(d, l) for d, l in zip(data, labels) if len(d) > 0]
    if not filtered:
        print(f"No hi ha dades per a: {title}")
        return False
        
    filtered_data, filtered_labels = zip(*filtered)
    plt.figure(figsize=(8,5))
    plt.boxplot(filtered_data, tick_labels=filtered_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_').replace('\'', '')}.png")
    plt.show()
    return True

def safe_barplot(categories, values, xlabel, ylabel, title):
    # Filtra categories amb valors zero
    filtered = [(c, v) for c, v in zip(categories, values) if v > 0]
    if not filtered:
        print(f"No hi ha dades per a: {title}")
        return False
    
    filtered_categories, filtered_values = zip(*filtered)
    plt.figure(figsize=(8,5))
    plt.bar(filtered_categories, filtered_values, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_').replace('\'', '')}.png")
    plt.show()
    return True

# Determinem si la relació familiar general és bona o dolenta
if total > 0:
    mitjana = sum(int(k) * v for k, v in counts.items()) / total
    print(f"\nMitjana de famrel: {mitjana:.2f}")
    if mitjana >= 3:
        print("En general, els alumnes tenen una bona relació amb la família.")
    else:
        print("En general, els alumnes NO tenen una bona relació amb la família.")
else:
    print("No s'han trobat dades.")

# --- FUNCIONS ADDICIONALS ---
def plot_famrel_vs_variable(variable, variable_label, variable_type='numeric'):
    """
    Mostra un boxplot (per variables numèriques) o barres (per categòriques) de famrel vs una altra variable.
    """
    var_key = variable.lower()
    famrel_vals = []
    var_vals = []
    
    # Comença per comprovar si la columna existeix (pot ser majúscula o minúscula)
    columnes = set()
    if data:
        columnes = set(data[0].keys())
        if var_key not in [c.lower() for c in columnes]:
            print(f"Error: No s'ha trobat la columna '{variable}'")
            print(f"Columnes disponibles: {', '.join(columnes)}")
            return
    
    # Troba la clau correcta a les dades (pot ser majúscula o minúscula)
    clau_real = next((c for c in columnes if c.lower() == var_key), None)
    
    if not clau_real:
        print(f"Error: No s'ha pogut trobar la columna '{variable}'")
        return
    
    for row in data:
        famrel = row['famrel'].strip()
        if famrel in famrel_dict:
            try:
                val = row[clau_real].strip()
                famrel_vals.append(famrel_dict[famrel])
                var_vals.append(val)
            except:
                continue
    
    famrel_unique = [famrel_dict[k] for k in sorted(famrel_dict.keys())]
    data_dict = {k: [] for k in famrel_unique}
    
    for f, v in zip(famrel_vals, var_vals):
        if variable_type == 'numeric':
            try:
                data_dict[f].append(float(v))
            except:
                continue
        else:
            data_dict[f].append(v)
    
    # Verifica si hi ha dades vàlides
    if not any(len(data_dict[k]) > 0 for k in famrel_unique):
        print(f"No hi ha dades vàlides per a: {variable_label} segons qualitat de la relació familiar")
        return
    
    if variable_type == 'numeric':
        safe_boxplot([data_dict[k] for k in famrel_unique], famrel_unique, 
                   'Qualitat de la relació familiar', variable_label, 
                   f'{variable_label} segons relació familiar')
    else:
        from collections import Counter
        counts = [Counter(data_dict[k]) for k in famrel_unique]
        categories = sorted(set([v for c in counts for v in c]))
        
        if not categories:
            print(f"No hi ha dades categòriques vàlides per a: {variable_label}")
            return
            
        bar_data = np.array([[c[cat] for cat in categories] for c in counts])
        bottom = np.zeros(len(famrel_unique))
        
        plt.figure(figsize=(8,5))
        for i, cat in enumerate(categories):
            plt.bar(famrel_unique, bar_data[:,i], bottom=bottom, label=cat)
            bottom += bar_data[:,i]
        plt.xlabel('Qualitat de la relació familiar')
        plt.ylabel('Nombre d\'alumnes')
        plt.title(f'{variable_label} segons relació familiar')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{variable_label.lower().replace(' ', '_').replace('\'', '')}_segons_famrel.png")
        plt.show()

# --- GRÀFICS GENERALS INTERESSANTS ---
def plot_age_distribution():
    """Mostra la distribució d'edats dels alumnes"""
    edats = []
    for row in data:
        try:
            edats.append(int(row['age']))
        except:
            continue
    
    if not edats:
        print("No hi ha dades d'edat vàlides")
        return
        
    safe_barplot(sorted(set(edats)), [edats.count(e) for e in sorted(set(edats))], 
               'Edat', 'Nombre d\'alumnes', 'Distribució d\'edat dels alumnes')

def plot_g3_distribution():
    """Mostra la distribució de notes finals (G3)"""
    notes = []
    for row in data:
        try:
            note = row['G3']
            if note and note.strip() and int(note) >= 0:
                notes.append(int(note))
        except:
            continue
    
    if not notes:
        print("No hi ha dades de notes finals vàlides")
        return
        
    safe_barplot(sorted(set(notes)), [notes.count(n) for n in sorted(set(notes))], 
               'Nota final (G3)', 'Nombre d\'alumnes', 'Distribució de la nota final')

def plot_absences_by_sex():
    """Compara les absències segons el sexe"""
    sexes = {'F': [], 'M': []}
    for row in data:
        sex = row['sex']
        try:
            absn = int(row['absences'])
            if sex in sexes:
                sexes[sex].append(absn)
        except:
            continue
    
    safe_boxplot([sexes['F'], sexes['M']], ['Femení', 'Masculí'], 
               'Sexe', 'Absències', 'Absències segons sexe')

def plot_g3_by_schoolsup():
    """Compara les notes finals segons si tenen suport escolar"""
    vals = {'yes': [], 'no': []}
    for row in data:
        try:
            sup = row['schoolsup'].lower()
            note = row['G3']
            if sup in vals and note and note.strip() and int(note) >= 0:
                vals[sup].append(int(note))
        except:
            continue
    
    safe_boxplot([vals['yes'], vals['no']], ['Amb suport', 'Sense suport'], 
               'Suport escolar', 'Nota final (G3)', 'Nota final segons suport escolar')

def plot_studytime_by_internet():
    """Compara el temps d'estudi segons si tenen Internet a casa"""
    vals = {'yes': [], 'no': []}
    for row in data:
        try:
            net = row['internet'].lower()
            st = int(row['studytime'])
            if net in vals:
                vals[net].append(st)
        except:
            continue
    
    safe_boxplot([vals['yes'], vals['no']], ['Amb Internet', 'Sense Internet'], 
               'Internet a casa', 'Temps d\'estudi', 'Temps d\'estudi segons Internet')

def plot_health_by_alcohol():
    """Compara l'estat de salut segons el consum d'alcohol"""
    # Busquem primer les columnes adequades (en majúscules o minúscules)
    dalc_key = next((c for c in data[0].keys() if c.lower() == 'dalc'), 'Dalc')
    walc_key = next((c for c in data[0].keys() if c.lower() == 'walc'), 'Walc')
    
    dalc = {i: [] for i in range(1,6)}
    walc = {i: [] for i in range(1,6)}
    
    for row in data:
        try:
            d = int(row[dalc_key])
            h = int(row['health'])
            if d in dalc:
                dalc[d].append(h)
        except:
            pass
            
        try:
            w = int(row[walc_key])
            h = int(row['health'])
            if w in walc:
                walc[w].append(h)
        except:
            pass
    
    safe_boxplot([dalc[i] for i in range(1,6)], [f'Nivell {i}' for i in range(1,6)], 
               'Consum alcohol (dies laborables)', 'Salut', 'Salut segons alcohol laboral')
    safe_boxplot([walc[i] for i in range(1,6)], [f'Nivell {i}' for i in range(1,6)], 
               'Consum alcohol (cap de setmana)', 'Salut', 'Salut segons alcohol cap de setmana')

def plot_freetime_by_romantic():
    """Compara el temps lliure segons si tenen parella"""
    vals = {'yes': [], 'no': []}
    for row in data:
        try:
            rom = row['romantic'].lower()
            ft = int(row['freetime'])
            if rom in vals:
                vals[rom].append(ft)
        except:
            continue
    
    safe_boxplot([vals['yes'], vals['no']], ['Amb parella', 'Sense parella'], 
               'Té parella', 'Temps lliure', 'Temps lliure segons parella')

def plot_g3_by_parent_status():
    """Compara les notes finals segons l'estat de convivència dels pares"""
    # Troba el nom correcte de la columna (Pstatus o pstatus)
    pstatus_key = next((c for c in data[0].keys() if c.lower() == 'pstatus'), None)
            
    if not pstatus_key:
        print("Error: No s'ha trobat la columna 'Pstatus'")
        return
        
    vals = {'T': [], 'A': []}
    for row in data:
        try:
            pstat = row[pstatus_key]
            note = row['G3']
            if pstat in vals and note and note.strip() and int(note) >= 0:
                vals[pstat].append(int(note))
        except:
            continue
    
    safe_boxplot([vals['T'], vals['A']], ['Conviuen', 'Separats'], 
               'Estat pares', 'Nota final (G3)', 'Nota final segons estat pares')

def plot_g3_by_failures():
    """Compara les notes finals segons el nombre de suspesos anteriors"""
    failures = {i: [] for i in range(5)}  # 0 a 4 suspesos
    for row in data:
        try:
            f = int(row['failures'])
            note = row['G3']
            if f in failures and note and note.strip() and int(note) >= 0:
                failures[f].append(int(note))
        except:
            continue
    
    safe_boxplot([failures[i] for i in range(5)], [f'{i} suspesos' for i in range(5)], 
               'Nombre de suspesos', 'Nota final (G3)', 'Nota final segons suspesos')

def plot_g3_by_parental_education():
    """Compara les notes finals segons el nivell educatiu mitjà dels pares"""
    # Troba els noms correctes de les columnes
    medu_key = next((c for c in data[0].keys() if c.lower() == 'medu'), None)
    fedu_key = next((c for c in data[0].keys() if c.lower() == 'fedu'), None)
    
    if not medu_key or not fedu_key:
        print("Error: No s'han trobat les columnes 'Medu' o 'Fedu'")
        return
    
    education = {i: [] for i in range(5)}  # 0 a 4 nivell educatiu
    for row in data:
        try:
            medu = int(row[medu_key])
            fedu = int(row[fedu_key])
            avg_edu = (medu + fedu) // 2
            note = row['G3']
            if avg_edu in education and note and note.strip() and int(note) >= 0:
                education[avg_edu].append(int(note))
        except:
            continue
    
    safe_boxplot([education[i] for i in range(5)], [f'Nivell {i}' for i in range(5)], 
               'Nivell educatiu pares', 'Nota final (G3)', 'Nota final segons estudis pares')

# Funció principal per executar tots els gràfics
def main():
    print("\nGenerant gràfics generals d'interès...")
    plot_age_distribution()
    plot_g3_distribution()
    plot_absences_by_sex()
    plot_g3_by_schoolsup()
    plot_studytime_by_internet()
    plot_health_by_alcohol()
    plot_freetime_by_romantic()
    plot_g3_by_parent_status()
    plot_g3_by_failures()
    plot_g3_by_parental_education()
    
    print("\nGenerant gràfics relacionats amb la relació familiar...")
    plot_famrel_vs_variable('age', 'Edat')
    plot_famrel_vs_variable('G3', 'Nota final')
    plot_famrel_vs_variable('absences', 'Absències')
    plot_famrel_vs_variable('Medu', 'Estudis de la mare')
    plot_famrel_vs_variable('Fedu', 'Estudis del pare')
    plot_famrel_vs_variable('studytime', 'Temps d\'estudi setmanal')
    plot_famrel_vs_variable('health', 'Estat de salut')
    plot_famrel_vs_variable('freetime', 'Temps lliure')
    plot_famrel_vs_variable('famsup', 'Suport familiar', 'categoric')
    plot_famrel_vs_variable('internet', 'Accés a Internet', 'categoric')
    
    print("\nAnàlisi completada.")

if __name__ == '__main__':
    main()
