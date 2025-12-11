

# 📘 COMPTE RENDU : ANALYSE DU PROJET DATA SCIENCE (CYBERSÉCURITÉ)

![WhatsApp Image 2025-10-27 à 13 39 11_c6ff40d2](https://github.com/user-attachments/assets/b394e0fd-933c-49ff-a8f4-046bf238ea93)













Chorouk dghoughi
22006691

## 1. Le Contexte Métier et la Mission

### Le Problème (Business Case)
Contrairement au cas médical (cancer), nous sommes ici face à un enjeu de **Cybersécurité Mondiale**. Les entreprises et gouvernements subissent des attaques variées générant des pertes financières massives.
* **Objectif :** Créer un modèle d'IA capable de classifier/prédire la nature de la menace (la Cible comporte ici **72 classes** distinctes, ce qui est beaucoup plus complexe qu'un problème binaire).
* **L'Enjeu critique :** Identifier correctement le type d'attaque ou l'attaquant permet d'activer la bonne stratégie de défense (ex: Firewall vs IA-based detection) et de minimiser les pertes financières et le vol de données.

### Les Données (L'Input)
Le dataset analysé dans le notebook contient **3000 observations** et **10 colonnes**.
* **Features (X) :** Variables mixtes incluant l'année (`Year`), les pertes financières (`Financial Loss`), le nombre d'utilisateurs affectés, etc.
* **Target (y) :** Une variable catégorielle très fragmentée avec **72 classes uniques**, ce qui rend la tâche de classification particulièrement ardue pour un modèle aléatoire.

---

## 2. Le Code Python (Laboratoire)
Le notebook suit la structure standard "Paillasse de laboratoire" :
C'est une excellente initiative. Pour respecter rigoureusement la structure pédagogique du fichier "Correction Projet.md" (style "Paillasse de laboratoire"), j'ai réorganisé ton code.

J'ai conservé toute la logique spécifique à ton dataset de Cybersécurité (gestion des 72 classes, encodage One-Hot, imputation mixte) mais je l'ai habillée avec les commentaires, les étapes numérotées et les affichages "pas à pas" typiques du fichier de correction.

Voici le code transformé :

```python
# ==============================================================================
# 📘 PROJET DATA SCIENCE : CYBERSECURITY THREAT ANALYSIS
# ==============================================================================
# Ce script suit la structure "Paillasse de Laboratoire" du guide de correction.
# Objectif : Nettoyer, Explorer et Modéliser des menaces de cybersécurité.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modules Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration esthétique
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore') # Silence les alertes pour la clarté

print("1. Bibliothèques importées. Prêt à démarrer.\n")

# ------------------------------------------------------------------------------
# 2. CHARGEMENT DES DONNÉES (L'Input)
# ------------------------------------------------------------------------------
print("2. Chargement du dataset...")

# Chargement du fichier
file_path = '/content/drive/MyDrive/Projet DS/Global_Cybersecurity_Threats_2015-2024.csv'
df = pd.read_csv(file_path)

# --- Normalisation de la cible (Spécifique à ce dataset) ---
# Si la colonne cible n'est pas nommée 'target', on la renomme pour standardiser le code
if df.columns[-1] != 'target':
    df.rename(columns={df.columns[-1]: 'target'}, inplace=True)

# Récupération des labels réels pour gérer les 72 classes correctement plus tard
actual_target_labels = sorted(df['target'].unique())
target_names = [str(label) for label in actual_target_labels]

print(f"   >>> Dataset chargé : {df.shape[0]} lignes, {df.columns.size} colonnes.")
print(f"   >>> Complexité du problème : {len(actual_target_labels)} classes uniques à prédire.\n")

# ------------------------------------------------------------------------------
# 3. SIMULATION DE "DONNÉES SALES" (Mise en situation)
# ------------------------------------------------------------------------------
# Le monde réel est sale. On simule des trous de données (NaN) pour tester notre nettoyage.
print("3. Sabotage contrôlé des données (Introduction de NaN)...")

np.random.seed(42) 
df_dirty = df.copy()

# On ne touche pas à la Target, mais on abîme les Features (5% de trous)
features_columns = df.columns[:-1]
for col in features_columns:
    mask = np.random.random(df.shape[0]) < 0.05
    df_dirty.loc[mask, col] = np.nan

nb_missing = df_dirty.isnull().sum().sum()
print(f"   >>> {nb_missing} valeurs manquantes générées artificiellement.\n")

# ------------------------------------------------------------------------------
# 4. NETTOYAGE ET PRÉPARATION (Data Wrangling)
# ------------------------------------------------------------------------------
print("4. Nettoyage des données (Réparation)...")

# Séparation X (Features) et y (Target)
X = df_dirty.drop('target', axis=1)
y = df_dirty['target']

# --- Stratégie Hybride : Numérique vs Catégoriel ---
# Contrairement au cancer (tout numérique), ici nous avons du texte.
numerical_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

# A. Imputation Numérique (Moyenne)
if len(numerical_cols) > 0:
    imputer_num = SimpleImputer(strategy='mean')
    X_num = pd.DataFrame(imputer_num.fit_transform(X[numerical_cols]), 
                         columns=numerical_cols, index=X.index)
else:
    X_num = pd.DataFrame(index=X.index)

# B. Imputation Catégorielle (Mode / Plus fréquent)
if len(categorical_cols) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_cat = pd.DataFrame(imputer_cat.fit_transform(X[categorical_cols]), 
                         columns=categorical_cols, index=X.index)
else:
    X_cat = pd.DataFrame(index=X.index)

# Reconstruction du dataset propre
X_clean = pd.concat([X_num, X_cat], axis=1)
# On remet les colonnes dans l'ordre d'origine
X_clean = X_clean[X.columns]

print(f"   >>> Nettoyage terminé. Valeurs manquantes restantes : {X_clean.isnull().sum().sum()}\n")

# ------------------------------------------------------------------------------
# 5. ANALYSE EXPLORATOIRE (EDA)
# ------------------------------------------------------------------------------
print("5. Inspection des données (EDA)...")

# A. Statistiques descriptives
print("   --- Statistiques (Variables Numériques) ---")
if len(numerical_cols) > 0:
    print(X_clean[numerical_cols].describe().T.head())
else:
    print("   (Pas de variables numériques)")

# B. Visualisation de distribution
plt.figure(figsize=(10, 5))
if len(numerical_cols) > 0:
    col_plot = numerical_cols[0]
    sns.histplot(data=df, x=col_plot, hue='target', element="step", common_norm=False)
    plt.title(f"Distribution : {col_plot} (Premier Feature Numérique)")
elif len(categorical_cols) > 0:
    col_plot = categorical_cols[0]
    sns.countplot(data=df, x=col_plot, hue='target')
    plt.title(f"Distribution : {col_plot} (Premier Feature Catégoriel)")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# C. Matrice de Corrélation
if len(numerical_cols) > 1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(X_clean[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matrice de Corrélation")
    plt.show()

print("\n")

# ------------------------------------------------------------------------------
# 6. ENCODAGE ET SPLIT (Train / Test)
# ------------------------------------------------------------------------------
print("6. Préparation pour le Machine Learning...")

# A. Encodage One-Hot (Transformer le texte en nombres pour l'IA)
print("   >>> Encodage des variables catégorielles (One-Hot)...")
X_encoded = pd.get_dummies(X_clean, columns=categorical_cols, drop_first=True)

# B. Split Train/Test
# On garde 20% pour l'examen final
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print(f"   >>> Données d'Entraînement : {X_train.shape}")
print(f"   >>> Données de Test (Cachées) : {X_test.shape}\n")

# ------------------------------------------------------------------------------
# 7. MODÉLISATION (Random Forest)
# ------------------------------------------------------------------------------
print("7. Entraînement du Cerveau (Random Forest)...")

# Création du modèle (100 arbres de décision qui votent)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement (Fit)
model.fit(X_train, y_train)
print("   >>> Modèle entraîné avec succès.\n")

# ------------------------------------------------------------------------------
# 8. ÉVALUATION (L'Heure de Vérité)
# ------------------------------------------------------------------------------
print("8. Résultats et Performance...")

# Prédictions
y_pred = model.predict(X_test)

# A. Accuracy Globale
acc = accuracy_score(y_test, y_pred)
print(f"   >>> Accuracy Score : {acc*100:.2f}%")

# B. Rapport détaillé (Précision, Rappel par classe)
print("\n   >>> Rapport de Classification (Extrait) :")
# Note : Avec 72 classes, le rapport complet est long, on l'affiche quand même
print(classification_report(y_test, y_pred, labels=actual_target_labels, target_names=target_names))

# C. La Matrice de Confusion (Visualisation des erreurs)
cm = confusion_matrix(y_test, y_pred, labels=actual_target_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Blues', cbar=True) # Annot=False car 72x72 c'est illisible avec des chiffres
plt.title(f'Matrice de Confusion ({len(actual_target_labels)} Classes)')
plt.xlabel('Classe Prédite')
plt.ylabel('Classe Réelle')
plt.show()

print("\n--- FIN DU RAPPORT ---")
```
1.  **Acquisition :** Chargement de 3000 lignes.
2.  **Simulation d'erreurs :** Introduction artificielle de valeurs manquantes (NaN) dans 1350 cellules pour tester la robustesse du nettoyage.
3.  **Nettoyage & Imputation :** Traitement différencié des variables numériques et catégorielles.
4.  **Modélisation & Évaluation :** Entraînement du modèle et visualisation de la performance sur 72 classes.

---

## 3. Analyse Approfondie : Nettoyage (Data Wrangling)

### La Mécanique de l'Imputation dans ce Notebook
Le notebook a dû gérer deux types de données, contrairement au projet médical purement numérique :
1.  **Imputation Numérique :** Pour des colonnes comme `Financial Loss`, le code a utilisé la **Moyenne** (Mean). Les trous ont été bouchés par la valeur moyenne calculée (~50.63 Millions $).
2.  **Imputation Catégorielle :** Pour les colonnes textuelles (ex: type d'attaque), le code a utilisé le **Mode** (la valeur la plus fréquente).

### 💡 Le Coin de l'Expert (Data Leakage)
*Observation Critique :* Dans le notebook, le nettoyage (Étape 4) semble avoir été effectué sur l'ensemble du dataset *avant* le split Train/Test.
* **Verdict :** Il y a un risque de **Data Leakage**. En calculant la moyenne des pertes financières sur les 3000 lignes (y compris celles qui serviront au test), le modèle a "triché" en voyant indirectement des informations du futur. Dans un environnement de production strict, il faudrait `fit` l'imputer uniquement sur le Train Set.

---

## 4. Analyse Approfondie : Exploration (EDA)

L'analyse des statistiques descriptives (étape 5 du notebook) révèle la structure des données :

### Décrypter `.describe()`
* **Symétrie Parfaite (Distribution Normale ?) :**
    * Pour `Financial Loss`, la Moyenne est de **50.63** et la Médiane (50%) est de **50.63**.
    * Pour `Affected Users`, la Moyenne est de **503,899** et la Médiane est de **503,899**.
* **Interprétation :** Contrairement aux données médicales souvent asymétriques (skewed), ces données (probablement simulées ou très équilibrées) suivent une distribution parfaitement symétrique. Il n'y a pas d'outliers massifs qui tirent la moyenne vers le haut.
* **Dispersions (Std) :** Les écarts-types sont significatifs (28M$ de perte), indiquant une grande variété dans la gravité des attaques, ce qui est une bonne nouvelle pour l'apprentissage du modèle (il a de la variance à expliquer).

---

## 5. Analyse Approfondie : Méthodologie (Split)

Le protocole expérimental reste le garant de la généralisation. Avec 3000 lignes et 72 classes, le split (probablement 80/20 standard) laisse environ 600 exemples pour le test.
* **Le Défi Multiclasse :** Avec 72 classes, certaines classes peuvent être rares. Un split aléatoire simple (`train_test_split`) risque de ne mettre *aucun* exemple d'une classe rare dans le jeu d'entraînement. Une séparation **stratifiée** (`stratify=y`) serait ici fortement recommandée pour s'assurer que le modèle voit au moins une fois chaque type de menace.

---

## 6. FOCUS THÉORIQUE : L'Algorithme Random Forest 🌲

Dans ce contexte de cybersécurité avec des données mixtes (catégorielles et numériques) et un grand nombre de classes :

### La Pertinence du Random Forest
* **Robustesse aux dimensions :** Avec 72 classes en sortie, un arbre de décision unique serait gigantesque et ferait du sur-apprentissage (overfitting) massif.
* **Le Bagging à la rescousse :** En moyennant les décisions de plusieurs arbres, le Random Forest lisse les frontières de décision. Si un arbre se trompe sur une cyber-attaque spécifique (ex: confondre un Malware Russe avec un Phishing Chinois), les autres arbres peuvent corriger le tir par vote majoritaire.

---

## 7. Analyse Approfondie : Évaluation (L'Heure de Vérité)

### A. La Matrice de Confusion (72x72)
La visualisation générée dans le notebook (`sns.heatmap`) est une grille massive de 72x72 cases.
* **Diagonale :** Les cases sur la diagonale représentent les **Succès** (Attaque prédite = Attaque réelle).
* **Hors Diagonale :** Tout le reste est du bruit.
* **Lecture :** Contrairement au cas binaire (4 cases), on cherche ici des "clusters" d'erreurs. Par exemple, le modèle confond-il souvent les attaques "Ransomware" avec "Malware" ?

### B. Les Métriques Avancées (Adaptation Multiclasse)
* **Accuracy (Précision Globale) :** Avec 72 classes, une accuracy de 50% serait en réalité excellente (le hasard ferait 1/72 ≈ 1.4%). Il ne faut donc pas juger ce chiffre avec les standards du binaire (où 50% est nul).
* **Précision & Rappel (Macro/Weighted Average) :**
    * Si le **Rappel** est bas pour une classe critique (ex: "Attaque Étatique"), cela signifie que le système de défense laisse passer des menaces majeures sans les détecter.
    * Si la **Précision** est basse, le système génère trop de fausses alertes, noyant les analystes de sécurité sous du bruit (fatigue d'alerte).

### Conclusion
Le projet présenté dans `CODE.ipynb` est techniquement plus complexe que le projet médical sur un point : la **cardinalité de la cible** (72 classes). Le nettoyage a réussi (0 NaN restants), mais la vigilance sur le Data Leakage et l'interprétation des résultats multiclasses reste primordiale pour un déploiement industriel.
