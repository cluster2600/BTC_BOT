import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
import joblib
import time
import os

print("[INFO] Début de l'entraînement du modèle Random Forest...")

# Nombre d'exemples à utiliser (10 000 exemples)
num_samples = 10000
print(f"[INFO] Génération d'un jeu de données de {num_samples} exemples...")

# Création d'un DataFrame fictif avec 10 000 exemples
data = {
    'price': np.random.rand(num_samples) * 10000,
    'sma': np.random.rand(num_samples) * 10000,
    'rsi': np.random.rand(num_samples) * 100,
    'macd': np.random.rand(num_samples) * 10 - 5,
    'signal': np.random.rand(num_samples) * 10 - 5,
    'lower_bb': np.random.rand(num_samples) * 10000,
    'sma_bb': np.random.rand(num_samples) * 10000,
    'upper_bb': np.random.rand(num_samples) * 10000,
    'social_feature': np.random.randint(0, 100, num_samples),
    'target': np.random.choice([1, 0, -1], num_samples)  # 1 = BUY, 0 = HOLD, -1 = SELL
}

df = pd.DataFrame(data)
print(f"[INFO] Jeu de données créé. Forme du DataFrame : {df.shape}")

# Séparation des données en caractéristiques (X) et variable cible (y)
X = df[['price', 'sma', 'rsi', 'macd', 'signal', 'lower_bb', 'sma_bb', 'upper_bb', 'social_feature']]
y = df['target']

print("[INFO] Séparation des données en ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[INFO] Taille de l'ensemble d'entraînement : {X_train.shape}")
print(f"[INFO] Taille de l'ensemble de test : {X_test.shape}")

# Création du modèle RandomForestClassifier de base
rf = RandomForestClassifier(random_state=42)

# Définition d'une grille d'hyperparamètres pour entraîner un modèle plus grand
param_grid = {
    'n_estimators': [400, 600, 800],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5]
}
print(f"[INFO] Démarrage de la recherche par grille avec les hyperparamètres : {param_grid}")

start_time = time.time()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1_macro', verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
end_time = time.time()
print(f"[INFO] Recherche terminée en {end_time - start_time:.2f} secondes.")

# Récupération du meilleur modèle et affichage des résultats
best_rf = grid_search.best_estimator_
print("[INFO] Meilleurs hyperparamètres trouvés :", grid_search.best_params_)
print("[INFO] Meilleur score de validation croisée :", grid_search.best_score_)

# Évaluation sur l'ensemble de test
y_pred = best_rf.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("[INFO] F1-score sur l'ensemble de test :", f1)

# Sauvegarde du modèle entraîné dans un fichier 'model_rf.pkl'
joblib.dump(best_rf, "model_rf.pkl")
file_size = os.path.getsize("model_rf.pkl")
print(f"[INFO] Modèle sauvegardé sous le nom 'model_rf.pkl' (taille : {file_size} octets).")