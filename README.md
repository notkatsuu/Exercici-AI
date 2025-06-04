# Predicció d'Alumnat: Models Predictius per a Rendiment d'Estudiants

Aquest projecte utilitza tècniques d'anàlisi de dades i aprenentatge automàtic per predir diferents aspectes del rendiment acadèmic d'estudiants de secundària a partir d'un conjunt de dades de Portugal.

## 🎵 Desenvolupat en "Vibe Code" 🎵
Aquest projecte ha estat desenvolupat seguint la filosofia "vibe code" - una aproximació de programació basada en la intuïció i el flux creatiu, prioritzant la claredat i l'expressivitat del codi sobre la rigidesa i l'optimització prematura.

## 📚 Context del Projecte

Aquest projecte utilitza un conjunt de dades d'estudiants de secundària de Portugal que conté informació sobre els seus atributs demogràfics, socials i escolars. L'objectiu és utilitzar aquestes dades per desenvolupar models predictius que puguin:

1. Predir el nombre d'absències dels estudiants
2. Classificar si un estudiant aprovarà o suspendrà (nota >= 10)
3. Identificar estudiants excepcionals (amb notes molt altes, generalment >= 19), tot
i que s'ha utilitzat un llindar de 17 per temes de precisió i recall.

Els models desenvolupats poden ajudar el professorat i els centres educatius a identificar estudiants en risc d'absentisme, fracàs escolar o amb potencial d'excel·lència acadèmica, permetent intervencions proactives i personalitzades.

## 📊 Estructura del Projecte

El projecte està organitzat en tres parts principals:

### 1. Anàlisi d'Alumnat (`1-Analisi-Alumnat/`)
Aquesta part conté scripts i visualitzacions per a l'anàlisi exploratòria de les dades. S'examinen les relacions entre diferents variables i es generen gràfics que ajuden a comprendre millor les dades.

Algunes visualitzacions destacades:
- Distribució de la nota final
- Absències segons relació familiar
- Nota final segons suspesos previs
- Estat de salut segons consum d'alcohol

### 2. Models Predictius (`2-Models-Predictius/`)
Aquesta secció conté els tres models d'aprenentatge automàtic desenvolupats:

#### Model d'Absències (`Absencies/`)
Un model de regressió que prediu el nombre d'absències d'un estudiant basant-se en els seus atributs personals i acadèmics.

```
python model_absencies.py --data [ruta_dades] --output-dir [directori_sortida] --params [ruta_params]
```

#### Model d'Aprovats (`Aprovat/`)
Un model de classificació binària que prediu si un estudiant aprovarà (nota >= 10) o no.

```
python model_aprovat.py --data [ruta_dades] --output-dir [directori_sortida] --params [ruta_params]
```

#### Model d'Estudiants Excepcionals (`Excepcionals/`)
Un model de classificació binària que identifica estudiants amb rendiment excepcional (nota >= 18).

```
python model_excepcionals.py --data [ruta_dades] --output-dir [directori_sortida] --params [ruta_params]
```

### 3. Testing dels Models (`3-Testing-Models/`)
Scripts i eines per avaluar el rendiment dels models entrenats.

## 🛠 Característiques Tècniques

### Llibreries Principals
- **scikit-learn**: Per a preprocessament de dades i models de machine learning
- **pandas/numpy**: Per a manipulació i anàlisi de dades
- **imblearn**: Per a gestionar el desequilibri de classes amb tècniques de sobremostreig (SMOTE)
- **joblib**: Per desar models entrenats

### Tècniques d'Aprenentatge Automàtic
- **Preprocessament**: Imputació de valors nuls, normalització, codificació one-hot
- **Selecció de característiques**: Random Forest per identificar les variables més importants
- **Modelatge**: GradientBoosting per a regressió i classificació
- **Balanceig de classes**: SMOTE per gestionar classes desequilibrades en els problemes de classificació

## 📝 Preprocessament de Dades

Tots els models segueixen un flux de preprocessament similar:
1. **Separació de característiques** numèriques i categòriques
2. **Imputació de valors nuls**: 
   - Mediana per a variables numèriques
   - Valor més freqüent per a variables categòriques
3. **Normalització/Escalat** de variables numèriques
4. **Codificació one-hot** per a variables categòriques

## 🧪 Avaluació dels Models

Els models es validen utilitzant:
- **Divisió train/test** (80%/20%)
- **Mètriques específiques** per a cada tipus de model:
  - Regressió: Error quadràtic mitjà, R²
  - Classificació: Exactitud, precisió, sensibilitat, F1
- **Corbes PR** (Precision-Recall) per als models de classificació

## 🚀 Com Utilitzar el Projecte

1. Clone el repositori
2. Instal·li les dependències:
   ```
   pip install scikit-learn pandas numpy matplotlib seaborn imblearn joblib
   ```
3. Executi els scripts d'anàlisi o models predictius segons les seves necessitats

## 📂 Conjunt de Dades

El dataset utilitzat conté informació d'estudiants de secundària de Portugal, incloent:
- Variables demogràfiques (edat, sexe, etc.)
- Situació familiar (educació dels pares, relació familiar, etc.)
- Activitats escolars i extraescolars
- Temps d'estudi i oci
- Consum d'alcohol
- Notes en diferents períodes i assignatures

## 🔍 Conclusions i Resultats

Els models entrenats mostren capacitat predictiva significativa, especialment:
- El model d'aprovats aconsegueix alta precisió en identificar estudiants en risc de suspendre
- El model d'absències identifica patrons relacionats amb l'absentisme escolar
- El model d'estudiants excepcionals, malgrat el fort desequilibri de classes, identifica correctament els estudiants amb potencial molt alt

## 👥 Autors

[Nom de l'autor/a]

## 📄 Llicència

Aquest projecte està disponible sota [especificar llicència].
