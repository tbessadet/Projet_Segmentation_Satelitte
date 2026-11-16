## 1. Cadrage du projet
1. **Périmètre :** on ne travaille que sur Toulouse ? Est-ce que ça a vocation à être généralisé, et sur quelle zone ? (Q1)

2. **Objectif :** partir des 380 000 pixels définis pour généraliser à toute la ville ? Ça implique une généralisation à grande échelle (spatiale et spectrale). (Q3, début)  


## 2. Données et labellisation
3. **Labellisation :** comment les pixels sont-ils labellisés ? Y a-t-il un objectif d'en faire plus ? (Q10)

4. **Répartition des données :** je comprends que les pixels labellisés utilisés pour l'entraînement ne représentent que ~7 % des données d'entraînement. Est-ce bien ça ? Quelle est la logique des ensembles (labeled training set, labeled pool, validation, test, unlabeled pool) ? (Q4)

5. **Sources de données :** y a-t-il plusieurs relevés hyperspectraux, ou OpenStreetMap en complément ? (Q8)
   - S'il n'y en a qu'un : comment traiter les voitures, les ombres, etc. ?
   - S'il y en a plusieurs : quel critère de sélection entre deux données diffé*rentes ?


## 3. Modèles et choix méthodologiques
1. **DINO :** va-t-on utiliser le modèle DINO ? (Q5)

2. **Deep clustering / MAE :** peux-tu préciser ces approches ? Va-t-on en avoir l'utilité ? (Q6)

## 4. Cas particuliers
8. **Ombres :** comment sont-elles gérées en détail ? (Q2)

9.  **Voitures :** comment les gère-t-on, puisqu'elles ne sont pas immobiles et pas liées au sol ? (Q7)

10. **Pixels mixtes et pixels entiers :** comment les distingue-t-on ? (Q9)

## 5. Évaluation, la partie la plus délicate
11. **Écart entre donnée terrain et sortie du modèle :** on peut avoir deux résultats cohérents mais non identiques. Comment savoir automatiquement si le modèle se trompe ou non ? (Q3, fin)


Pour la définition du modèle panoptique, ce n'est pas une question mais une note. Tu peux la garder en rappel avant la partie 3 si tu veux l'utiliser en entretien.

## Rappels (notes)

**Modèle panoptique :** chaque pixel reçoit un couple (classe, identifiant d'instance).
- *Stuff* (route, herbe, ciel…) : pas d'instance, un seul segment par classe.
- *Things* (voiture, personne, bâtiment…) : chaque objet a son identifiant (pas de lien intra-classe).

**Ensembles de données :**

| Ensemble | Indice | Rôle | Part moyenne des pixels labellisés |
|---|---|---|---|
| Labeled training set | 1 | Supervision | ~13 % |
| Labeled pool | 2 | Non-labellisé « propre » : uniquement des classes connues (apprentissage actif ou semi-supervisé) | ~29 % |
| Validation set | 3 | Choix des hyperparamètres | ~14 % |
| Test set | 4 | Évaluation finale | ~46 % |
| Unlabeled pool | — | ~2,6 millions de pixels vraiment non labellisés, pouvant contenir des classes inconnues | — |

→ Les pixels labellisés utilisés pour l'entraînement ne représentent que ~7 % de toutes les données d'entraînement.

---

Véritée terrain sous forme de polygone (shape_file)  
Ne pas serparer les pixels du meme groupes dans jeux train/validation $\to$ dataleak sinon  
Matériel = 0 : usage des sols annoté, mais pas l'occupation  
matériel = 40 : zone utilisé pour apprentissage non supervisé (pixels non annotés)  
380.000 pixels tous annotés à la main, photo interprétation (lecture des spectre ... annoté par expert sur photo) + RPG (registre parcellaire graphique : declaration PAC)  
Capteur : 
- refléchi par le sol
- dispercé par l'atmosphere
- reflechi par voisinage du sol + dispersé  

$\to$ reflectance calcul/estimé mais =! de ce que recoit le capteur (qui depend lumiere, atmosphere ...)  
Mais imparfait car algo suppose sol plat (angle réel biaisé) $\to$ variabilité intra-classe  
Valeur de reflectance (sur le spectre) : moyenne sur ce qu'il y a  
Les pixels mixtes : aucun d'annoté, et c'est ce qu'on voudrait justement traiter dans modele -> faire en sorte, avec segmentation panoptique, que ces pixels soient les bords des objets (ce qui est la réalité) sans forcement donner de classe  
OpenStreetMap : Données vectorielles codifiées.  
Approche auto-supervisé : extraire des representation spectrale à l'échelle d'un pixel. (a partir de tout le spectre).  
Images superspectral 10aine cannaux / hyperspectral dans les 100aine  
Segmentation panoptique : centre instance, banding box, classe sementique instance  

1. Obtenir un modele qui extrait des representation a partir d'un pixel + selection d'un core-set de pixel non labélisé  
$\to$ transformer et light-weight transformer à revoir  
MAE : Cacher un certains nombre d'entrée et reconstruire l'entrée ... Ce transfers au spectre, c'est pourquoi on utilise transformer (meme si MLP fonctionne bien pour spectre, mais pas possible d'avoir cette transformation)  
Vision transformer computationnelement très lourd (train et prod)  

2.   


Pour l'instant, aller lire les papiers et en sortir les infos/méthodes qui pourraient être interessantes et/ou utile pour le projet  
Article données, light-weight transformer, segmentation panoptique  
