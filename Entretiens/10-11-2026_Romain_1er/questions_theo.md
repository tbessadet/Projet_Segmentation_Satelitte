## 1. Cadrage du projet
1. **Périmètre :** on ne travaille que sur Toulouse ? Est-ce que ça a vocation à être généralisé, et sur quelle zone ? (Q1)
2. **Objectif :** partir des 380 000 pixels définis pour généraliser à toute la ville ? Ça implique une généralisation à grande échelle (spatiale et spectrale). (Q3, début)

## 2. Données et labellisation
3. **Labellisation :** comment les pixels sont-ils labellisés ? Y a-t-il un objectif d'en faire plus ? (Q10)
4. **Répartition des données :** je comprends que les pixels labellisés utilisés pour l'entraînement ne représentent que ~7 % des données d'entraînement. Est-ce bien ça ? Quelle est la logique des ensembles (labeled training set, labeled pool, validation, test, unlabeled pool) ? (Q4)
5. **Sources de données :** y a-t-il plusieurs relevés hyperspectraux, ou OpenStreetMap en complément ? (Q8)
   - S'il n'y en a qu'un : comment traiter les voitures, les ombres, etc. ?
   - S'il y en a plusieurs : quel critère de sélection entre deux données différentes ?

## 3. Modèles et choix méthodologiques
6. **DINO :** va-t-on utiliser le modèle DINO ? Lien avec mon stage. (Q5)
7. **Deep clustering / MAE :** peux-tu préciser ces approches ? Va-t-on en avoir l'utilité ? (Q6)

## 4. Cas particuliers
8. **Ombres :** comment sont-elles gérées en détail ? (Q2)
9. **Voitures :** comment les gère-t-on, puisqu'elles ne sont pas immobiles et pas liées au sol ? (Q7)
10. **Pixels mixtes et pixels entiers :** comment les distingue-t-on ? (Q9)

## 5. Évaluation, la partie la plus délicate
11. **Écart entre donnée terrain et sortie du modèle :** on peut avoir deux résultats cohérents mais non identiques. Comment savoir automatiquement si le modèle se trompe ou non ? (Q3, fin)

Pour la définition du modèle panoptique, ce n'est pas une question mais une note. Tu peux la garder en rappel avant la partie 3 si tu veux l'utiliser en entretien.