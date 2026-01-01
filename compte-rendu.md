

# 📘 COMPTE RENDU : ANALYSE DU PROJET DATA SCIENCE (CYBERSÉCURITÉ)

![WhatsApp Image 2025-10-27 à 13 39 11_c6ff40d2](https://github.com/user-attachments/assets/b394e0fd-933c-49ff-a8f4-046bf238ea93)













Chorouk dghoughi
22006691

## 1. Le Contexte Métier et la Mission

### Le Problème (Business Case)
Nous sommes ici face à un enjeu de **Cybersécurité Mondiale**. Les entreprises et gouvernements subissent des attaques variées générant des pertes financières massives.
* **Objectif :** Créer un modèle d'IA capable de classifier/prédire la nature de la menace (la Cible comporte ici **72 classes** distinctes, ce qui est beaucoup plus complexe qu'un problème binaire).
* **L'Enjeu critique :** Identifier correctement le type d'attaque ou l'attaquant permet d'activer la bonne stratégie de défense (ex: Firewall vs IA-based detection) et de minimiser les pertes financières et le vol de données.

### Les Données (L'Input)
Le dataset analysé dans le notebook contient **3000 observations** et **10 colonnes**.
* **Features (X) :** Variables mixtes incluant l'année (`Year`), les pertes financières (`Financial Loss`), le nombre d'utilisateurs affectés, etc.
* **Target (y) :** Une variable catégorielle très fragmentée avec **72 classes uniques**, ce qui rend la tâche de classification particulièrement ardue pour un modèle aléatoire.
* 1. Contexte et Enjeux
Avec la numérisation croissante des infrastructures mondiales, le volume et la complexité des cyberattaques ont explosé entre 2015 et 2024. Les méthodes traditionnelles de surveillance manuelles ne suffisent plus face à la rapidité des attaques modernes. Ce projet vise à exploiter l'Intelligence Artificielle pour renforcer la sécurité des réseaux en automatisant la détection des intrusions.

2. Objectifs du Projet
L'objectif principal est de développer un modèle de Machine Learning (Apprentissage Supervisé) capable de :

-Analyser les logs de trafic réseau historiques.

-Identifier les modèles (patterns) suspects.

-Classifier avec précision le type d'attaque (Malware, DDoS, Phishing, Intrusion, etc.) ou de déterminer si le trafic est bénin.

3. Les Données (Dataset)
Le projet s'appuie sur le jeu de données Global Cybersecurity Threats, couvrant une période de 9 ans (2015-2024).

Source : Kaggle (Auteur : Atharva Soundankar).

Volume : Données structurées représentant des événements de cybersécurité.

Variables Clés (Features) : Le dataset contient probablement des informations techniques telles que les adresses IP (source/destination), les ports, les protocoles utilisés, la géolocalisation, et l'horodatage.

Cible (Target) : La catégorie de l'attaque (ex: 'Ransomware', 'Botnet', 'Benign', etc.).

4. Méthodologie Technique
Le projet suit un pipeline de Data Science rigoureux :
Exploration et Nettoyage (EDA & Cleaning) :
Gestion des valeurs manquantes et des données bruitées.
Analyse statistique de la répartition des attaques (déséquilibre des classes).
Visualisation des corrélations pour identifier les variables les plus influentes.
Prétraitement (Preprocessing) :
Encodage : Transformation des variables catégorielles (ex: Protocoles) en format numérique via One-Hot Encoding.
Normalisation : Mise à l'échelle des données numériques si nécessaire.
Modélisation (Modeling) :
Utilisation de l'algorithme Random Forest Classifier.
Choix de cet algorithme pour sa robustesse face au sur-apprentissage et sa capacité à gérer un grand nombre de variables et de classes.
Gestion du déséquilibre des classes (paramètre class_weight='balanced').

5. Résultats et Évaluation
La performance du modèle est évaluée via plusieurs métriques :
Accuracy : Taux global de bonnes prédictions.
Matrice de Confusion : Pour visualiser les erreurs de classification entre les différents types d'attaques (ex: confondre un DDoS avec du trafic normal).
Feature Importance : Identification des facteurs techniques (ex: Port de destination) qui sont les plus déterminants pour prédire une attaque.

6. Impact Business
Ce modèle permettrait à une équipe SOC (Security Operations Center) de :
Réduire le temps de réaction face à une menace.
Diminuer les "faux positifs" (fausses alertes).
Prioriser les interventions sur les attaques les plus critiques.


## 2. Le Code Python (Laboratoire)
Le notebook suit la structure standard "Paillasse de laboratoire" :
C'est une excellente initiative. Pour respecter rigoureusement la structure pédagogique du fichier "Correction Projet.md" (style "Paillasse de laboratoire"), j'ai réorganisé ton code.

J'ai conservé toute la logique spécifique à ton dataset de Cybersécurité (gestion des 72 classes, encodage One-Hot, imputation mixte) mais je l'ai habillée avec les commentaires, les étapes numérotées et les affichages "pas à pas" typiques du fichier de correction.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree

# Configuration esthétique
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
import warnings
warnings.filterwarnings('ignore')

# 1. CONFIGURATION ET CHARGEMENT 

print("--- ÉTAPE 1 : CHARGEMENT DES DONNÉES ---")

# Mettez à False pour utiliser votre vrai fichier CSV
USE_SYNTHETIC_DATA = True 
FILE_PATH = '/content/drive/MyDrive/CHEMIN/VERS/VOTRE/FICHIER.csv'

if USE_SYNTHETIC_DATA:
    print("MODE DÉMO : Génération de données synthétiques...")
    from sklearn.datasets import make_classification
    # On génère 1000 lignes, 20 colonnes, et 5 classes pour l'exemple
    X_raw, y_raw = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                       n_redundant=5, n_classes=5, random_state=42)
    df = pd.DataFrame(X_raw, columns=[f'Feature_{i}' for i in range(1, 21)])
    df['target'] = y_raw
    # On ajoute des noms de classes plus "réels"
    class_map = {0: 'Benign', 1: 'Malware', 2: 'Phishing', 3: 'DDoS', 4: 'Spyware'}
    df['target'] = df['target'].map(class_map)
    
else:
    try:
        df = pd.read_csv(FILE_PATH)
        print("Fichier chargé avec succès.")
    except FileNotFoundError:
        print(f"ERREUR : Fichier non trouvé à {FILE_PATH}. Vérifiez le chemin.")
        # Arrêt forcé si pas de fichier
        raise

# Renommage cible si nécessaire
if df.columns[-1] != 'target' and 'target' not in df.columns:
    df.rename(columns={df.columns[-1]: 'target'}, inplace=True)

print(f"Taille du dataset : {df.shape}")
print(f"Classes détectées : {df['target'].unique()}\n")

# 2. PRÉTRAITEMENT OPTIMISÉ

print("--- ÉTAPE 2 : NETTOYAGE ET PRÉPARATION ---")

# Séparation
X = df.drop('target', axis=1)
y = df['target']

# Introduction artificielle de bruit (seulement si démo)
if USE_SYNTHETIC_DATA:
    mask = np.random.random(X.shape) < 0.05
    X = X.mask(mask) # Introduit des NaN

# Identification des types de colonnes
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# Imputation (Remplissage des trous)
if len(num_cols) > 0:
    imp_num = SimpleImputer(strategy='mean')
    X[num_cols] = imp_num.fit_transform(X[num_cols])

if len(cat_cols) > 0:
    imp_cat = SimpleImputer(strategy='most_frequent')
    X[cat_cols] = imp_cat.fit_transform(X[cat_cols])
    # Encodage One-Hot pour les variables catégorielles (Features)
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Encodage de la Cible (Target) si c'est du texte
le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = [str(cls) for cls in le.classes_]

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
# Note: 'stratify' est crucial pour garder la même proportion de classes dans le train et le test

print("Données prêtes pour l'entraînement.\n")

# 3. MODÉLISATION (Random Forest)

print("--- ÉTAPE 3 : ENTRAÎNEMENT DU MODÈLE ---")

# Amélioration : class_weight='balanced' aide si certaines attaques sont rares
model = RandomForestClassifier(n_estimators=100, 
                               random_state=42, 
                               class_weight='balanced',
                               n_jobs=-1) # Utilise tous les cœurs du processeur

model.fit(X_train, y_train)
print("Modèle entraîné.\n")

# 4. ÉVALUATION ET DIAGRAMMES

print("--- ÉTAPE 4 : VISUALISATION DES RÉSULTATS ---")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"ACCURACY : {acc*100:.2f}%")

#  DIAGRAMME 1 : MATRICE DE CONFUSION 
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
# Normalisation par ligne pour voir les pourcentages d'erreur par classe
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Matrice de Confusion (Valeurs Absolues)')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.show()

#  DIAGRAMME 2 : IMPORTANCE DES FEATURES 
# C'est crucial pour comprendre QUELLES colonnes permettent de détecter l'attaque
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
# On garde le Top 15 pour la lisibilité
top_n = 15
indices = indices[:top_n]

plt.figure(figsize=(12, 6))
plt.title(f"Top {top_n} des Variables les plus Importantes (Feature Importance)")
plt.bar(range(top_n), importances[indices], align="center", color=sns.color_palette("viridis", top_n))
plt.xticks(range(top_n), [X.columns[i] for i in indices], rotation=45, ha='right')
plt.xlim([-1, top_n])
plt.tight_layout()
plt.show()

#  DIAGRAMME 3 : VISUALISATION D'UN ARBRE UNIQUE 
# Pour voir "comment le modèle pense"
plt.figure(figsize=(20, 10))
# On prend le premier arbre de la forêt (index 0)
# On limite la profondeur (max_depth=3) pour que ce soit lisible à l'écran
plot_tree(model.estimators_[0], 
          feature_names=X.columns,
          class_names=target_names,
          filled=True, 
          rounded=True, 
          max_depth=3,
          fontsize=10)
plt.title("Visualisation simplifiée d'un arbre de décision de la forêt")
plt.show()

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

###  Le Coin de l'Expert (Data Leakage)
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


