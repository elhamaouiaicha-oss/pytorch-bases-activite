# Activité pratique : Bases de PyTorch

Ce dépôt contient mon travail sur l'activité pratique « Bases de PyTorch » (module IA générative).
Tout le travail est dans le notebook **`Activite_Bases_de_Pytorch.ipynb`**.

## Contenu

### Exercice 1 : Régression
Prédiction du **salaire** d'un employé en fonction de son **ancienneté** et de son **degré d'expertise**, avec un modèle de régression construit avec PyTorch (300 exemples, données normalisées).
Résultat : erreur absolue moyenne de **381,64 DH** sur le jeu de test. Pour un salarié de 7 ans d'ancienneté et d'expertise 4/5, le salaire estimé est de 14 201,94 DH (valeur théorique ≈ 14 200 DH).

### Exercice 2 : Classification (dataset IRIS)
Classification des fleurs d'iris en 3 espèces (setosa, versicolor, virginica) à partir de 4 mesures.
Les données sont normalisées avec `StandardScaler`. Le réseau de neurones (4 → 16 → 3) est comparé à un modèle linéaire sans couche cachée, puis il prédit l'espèce d'une nouvelle fleur.
Accuracy sur le test : **100 %** (réseau de neurones) contre **90 %** (modèle linéaire).

### Exercice 3 : MLP sur MNIST
Reconnaissance de chiffres manuscrits (images 28×28) avec un perceptron multicouche (MLP), évaluée avec l'accuracy, la précision, le rappel et le F1-score (moyenne macro). Le modèle est sauvegardé puis rechargé, avec des prédictions identiques.
Résultats sur le test : Accuracy **97,40 %**, Précision 97,41 %, Rappel 97,36 %, F1 97,37 %.

### Exercice 4 : CNN sur MNIST
Réseau convolutif sur le même jeu de données :
- 2 couches de convolution (16 puis 32 filtres, noyau 3×3) avec ReLU et MaxPooling
- une couche linéaire cachée de 128 neurones, puis la sortie à 10 classes
- entraînement sur 3 époques

Accuracy sur le jeu de test : **98,85 %**.
Le modèle est sauvegardé dans `mnist_cnn.pth` (non versionné).

## Installation et exécution

```bash
git clone https://github.com/elhamaouiaicha-oss/pytorch-bases-activite.git
cd pytorch-bases-activite
pip install torch torchvision numpy matplotlib scikit-learn jupyter
jupyter notebook Activite_Bases_de_Pytorch.ipynb
```

Avec `uv` :
```bash
uv sync
```

Les données MNIST sont téléchargées automatiquement au premier lancement.

## Technologies
Python 3.11+, PyTorch, torchvision, scikit-learn, matplotlib, NumPy.

## Auteur
Aicha, 5ᵉ année Génie Informatique, EMSI Casablanca