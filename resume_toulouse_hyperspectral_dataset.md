# Toulouse Hyperspectral Data Set — résumé (page web + article)

> Contexte : base de travail pour le projet fil rouge *« Panoptic segmentation of airborne hyperspectral images for land cover mapping in large urban areas »* (R. Thoreau, AgroParisTech / MIA Paris-Saclay).

## Sommaire

1. [La page web en bref](#s1)
2. [Le problème de départ](#s2)
3. [Construction du jeu de données](#s3)
4. [Les découpages train / test spatialement disjoints](#s4)
5. [Comparaison avec Pavia et Houston](#s5)
6. [Auto-supervision pour apprendre des représentations spectrales](#s6)
7. [Expériences et résultats](#s7)
8. [Conclusions et perspectives de l'article](#s8)
9. [Lien avec le projet fil rouge](#s9)
10. [Pièges à éviter](#s10)

---

<a id="s1"></a>
## 1. La page web en bref

Le site présente le jeu de données comme l'association de **deux éléments** :

- une **image hyperspectrale aéroportée** (capteur AisaFENIX) acquise sur Toulouse pendant la campagne **CAMCATT-AI4GEO** ;
- une **vérité terrain d'occupation du sol** (*land cover*), livrée avec des **découpages train / test standards** pour évaluer des modèles de ML (classification pixel, segmentation sémantique…).

| Caractéristique | Valeur |
|---|---|
| Grandeur physique | Réflectance au sol (après correction atmosphérique) |
| Résolution spatiale (GSD) | **1 m** |
| Domaine spectral | **0,4 – 2,5 µm** (visible → SWIR) |
| Résolution spectrale | < 8 nm |
| Nombre de bandes | **310** (420 bandes brutes, moins les bandes d'absorption atmosphérique et les bandes trop bruitées) |
| Pixels annotés | **> 380 000**, annotations **éparses** |
| Surface couverte | ≈ 90 km² |
| Nomenclature | **32 classes** hiérarchiques : 16 imperméables + 16 perméables (+ une nomenclature secondaire d'usage du sol) |

**Ressources pratiques**

- **Bibliothèque Python `TlseHypDataSet`** (doc sur readthedocs) : construit directement des `DataLoader` PyTorch. Paramètres utiles : `pred_mode='pixel'`, `patch_size` (1 = pixel seul ; plus grand = patchs spatiaux), `dataset.standard_splits[i]` pour charger le i-ème découpage standard via `DisjointDataSplit`.
- **Téléchargement de l'image** : découpée en tuiles, disponibles sur le catalogue CAMCATT (sedoo) ou sur Hugging Face (`Romain3Ch216/TlseHypDataSet`). Tuiles utilisées : **1b, 1c, 1d, 1e, 3a, 3d, 3e, 5c, 9c**.
- **Vérité terrain** : incluse dans la bibliothèque, ou téléchargeable séparément sur le GitHub du projet.
- **Citation** : citer **les deux** articles (le *data paper* CAMCATT-AI4GEO dans *Data in Brief* et l'article du jeu de données, publié dans *ISPRS Journal of Photogrammetry and Remote Sensing*).
- **Licence** : CC-BY-NC 2.0. Contact : romain.thoreau@agroparistech.fr.

```python
import torch
from TlseHypDataSet.tlse_hyp_data_set import TlseHypDataSet
from TlseHypDataSet.utils.dataset import DisjointDataSplit

dataset = TlseHypDataSet('/path/to/dataset/', pred_mode='pixel', patch_size=1)
ground_truth_split = DisjointDataSplit(dataset, split=dataset.standard_splits[0])
train_loader = torch.utils.data.DataLoader(ground_truth_split.sets_['train'],
                                           shuffle=True, batch_size=1024)
```

> ⚠️ L'image entière ne tient pas en mémoire : c'est la raison d'être de la bibliothèque (chargement par morceaux).

---

<a id="s2"></a>
## 2. Le problème de départ

**Pourquoi cartographier les matériaux de surface en ville ?** Les surfaces artificielles imperméables influencent :

- l'**hydrologie** des bassins versants (sécheresses, inondations, ruissellement) ;
- les **îlots de chaleur urbains** (propriétés radiatives des matériaux) ;
- le **stockage de carbone** dans les sols (sols scellés).

**Pourquoi l'hyperspectral ?** Après correction atmosphérique, on obtient la **réflectance**, qui est **intrinsèque à la composition chimique** du matériau → très discriminante pour le *land cover*.

> 💡 Exemple de l'article : un court de tennis orange peut être en béton poreux ou en revêtement synthétique. En RGB c'est indiscernable ; avec le spectre complet, on peut les séparer. L'information **spatiale** apporte donc peu pour identifier le *matériau* — mais le **contexte à grande échelle** peut lever certaines ambiguïtés (voir §8).

**Le verrou principal : la rareté des labels.** Annoter des classes « bas niveau » (gravier vs asphalte…) demande de l'expertise et des campagnes terrain. En général **≤ 1 % des pixels** sont labellisés, ce qui représente mal la variabilité spectrale de l'image → intérêt pour l'apprentissage actif, semi-supervisé et auto-supervisé.

**Défaut des jeux publics existants** : couverture géographique limitée, peu de classes, et surtout **pas de découpage train / test standard et spatialement disjoint**.

> 🚩 Plusieurs travaux montrent qu'un **tirage aléatoire** des pixels train / test **surestime** les performances : des pixels voisins se ressemblent beaucoup plus que des pixels de même classe pris loin l'un de l'autre.

---

<a id="s3"></a>
## 3. Construction du jeu de données

### 3.1 Acquisition

- Capteur **AisaFENIX 1K** embarqué sur un avion Safire volant à ~1 500 m ; fauchée de 1 024 m ; GSD 1 m.
- Résolution spectrale : **3,6 nm en VNIR**, **7,8 nm en SWIR**.
- Chaîne de traitement : corrections radiométriques et géométriques → radiance au niveau avion → **réflectance au sol** via l'algorithme de correction atmosphérique **COCHISE**.
- Mesures **in situ** au spectromètre (ASD) sur divers matériaux : elles ont servi à définir la nomenclature.
- Sont aussi fournis : irradiance **directe et diffuse** au sol, **angle zénithal solaire** (≈ 22°).

### 3.2 Vérité terrain

- Construite par **photo-interprétation**, campagnes terrain complémentaires et **données exogènes** (le Registre Parcellaire Graphique pour les cultures).
- Annotations **éparses** : des **polygones disjoints**, pas une carte dense comme dans les jeux de segmentation classiques.
- Les **pixels mixtes** (bords de toits, petits objets comme des tuyaux sur les toits) ont été **volontairement exclus** pour garantir des labels exacts.

### 3.3 Nomenclature *land cover* (32 classes)

| Imperméables (16) | | Perméables (16) | |
|---|---|---|---|
| **Tuile** | 1 tuile orange, 2 tuile sombre | **Matériau synthétique** | 17 piste synthétique, 18 gazon synthétique |
| **Ardoise** | 3 ardoise | **Végétation** | 19 herbe saine, 20 herbe stressée, 21 arbre |
| **Fibrociment** | 4 clair, 5 sombre | **Sol / graviers & roches** | 22 sol nu, 23 graviers, 24 roches |
| **Tôle** | 6 tôle sombre, 7 tôle peinte claire | **Béton poreux** | 25 vert, 26 rouge |
| **Asphalte** | 8 sombre, 9 rouge, 10 beige, 11 peint en vert, 12 marquage blanc | **Eau** | 27 algues, 28 eau de piscine, 29 eau |
| **Pavés / autres** | 13 ciment, 14 pavé brun, 15 pavé clair, 16 bâche plastique claire | **Cultures** | 30 sorgho, 31 blé, 32 féverole |

### 3.4 Nomenclature *land use* (12 classes, plus abstraites)

Routes · voies ferrées · toits · parkings · chantiers · équipements sportifs · lacs / rivières / ports · piscines · forêts · champs cultivés · bateaux · espaces ouverts.

> 💡 Cette seconde nomenclature est **centrale pour le projet** : le *land cover* décrit **de quoi** est fait un pixel (matériau), le *land use* décrit **à quoi sert** une zone (objet / fonction). La segmentation panoptique doit justement articuler les deux.

---

<a id="s4"></a>
## 4. Les découpages train / test spatialement disjoints

### 4.1 Les 4 (+1) ensembles

Chaque découpage standard (il y en a **8**) contient :

| Ensemble | Indice | Rôle | Part moyenne des pixels labellisés |
|---|---|---|---|
| **Labeled training set** | 1 | Supervision | ~13 % |
| **Labeled pool** | 2 | Non-labellisé « propre » : uniquement des classes connues (utile pour l'apprentissage actif ou semi-supervisé) | ~29 % |
| **Validation set** | 3 | Choix des hyperparamètres | ~14 % |
| **Test set** | 4 | Évaluation finale | ~46 % |
| **Unlabeled pool** | — | ~2,6 millions de pixels **vraiment** non labellisés, pouvant contenir des classes **inconnues** | — |

→ Les pixels labellisés utilisés pour l'entraînement ne représentent que **~7 % de toutes les données** d'entraînement.

> 💡 **Pourquoi séparer spatialement ?** La variabilité intra-classe a trois origines, toutes **corrélées à la position géographique** :
> - **physique** : conditions d'éclairement (ombre / soleil, pente) ;
> - **intrinsèque** : petites variations chimiques ou effets non lambertiens (vieillissement de l'asphalte, mousse sur les toits orientés nord, teneur en eau / chlorophylle) ;
> - **sémantique** : plusieurs matériaux regroupés dans une même classe (ex. plusieurs essences d'arbres).

### 4.2 Le découpage comme problème d'optimisation en nombres entiers mixtes

On regroupe les polygones voisins en $n_{groups}$ groupes, puis on affecte chaque groupe à un ensemble.

**Hypothèses / notations préalables** : $u_{ij} \in \{0,1\}$ pour tout groupe $i \in \{1,\dots,n_{groups}\}$ et tout ensemble $j \in \{1,2,3,4\}$ ; $S = \{1, 3, 4\}$ ; l'unlabeled pool n'intervient pas.

$$
\min_{u} \; \sum_{s \in S} \sum_{i=1}^{n_{groups}} \sum_{k=1}^{c} P[i,k]\, u_{is}
$$

sous les contraintes :

$$
\forall i,\quad \sum_{j=1}^{4} u_{ij} = 1
$$

$$
\forall s \in S,\; \forall k \in \{1,\dots,c\},\quad \sum_{i=1}^{n_{groups}} P[i,k]\, u_{is} \;\ge\; p_s \sum_{i=1}^{n_{groups}} P[i,k]
$$

- $u_{ij}$ : vaut 1 si le groupe $i$ est affecté à l'ensemble $j$, 0 sinon
- $P[i,k]$ : nombre de pixels de la classe $k$ dans le groupe $i$
- $c$ : nombre de classes (32)
- $S$ : ensembles soumis à contrainte (train labellisé, validation, test)
- $p_s$ : proportion minimale de chaque classe exigée dans l'ensemble $s$

> 💡 Lecture : on **minimise** le nombre de pixels mis dans train / validation / test tout en **garantissant** pour **chaque classe** une proportion minimale dans chacun. Tout le reste part dans le *labeled pool*. Chaque groupe va dans **exactement un** ensemble.

> 🔎 Le texte de l'article dit « au moins un ensemble » pour la première contrainte, mais l'égalité à 1 impose bien **un et un seul**.

> ⚠️ Les proportions 13 / 29 / 14 / 46 % sont un choix des auteurs (priorités : test représentatif > validation représentative > assez d'exemples en train), **pas** le résultat d'une analyse statistique. Seulement 8 splits car les auteurs n'ont pas trouvé d'autres solutions vraiment différentes.

---

<a id="s5"></a>
## 5. Comparaison avec Pavia et Houston

| Jeu de données | GSD | Domaine spectral | Bandes | Classes | Pixels annotés | Ratio de déséquilibre |
|---|---|---|---|---|---|---|
| Pavia University | 1,3 m | 0,43 – 0,86 µm | 103 | 9 | 43 000 | 20 |
| Houston University | 1 m | 0,38 – 1,0 µm | 48 | 20 | 510 000 | 329 |
| **Toulouse** | **1 m** | **0,4 – 2,5 µm** | **310** | **32** | **380 000** | **225** |

**Ratio de déséquilibre** :

$$
IR = \frac{n_{\max}}{n_{\min}}
$$

- $n_{\max}$ : nombre d'échantillons de la classe la plus représentée
- $n_{\min}$ : nombre d'échantillons de la classe la moins représentée

**Ce qui distingue Toulouse :**

1. **Domaine spectral bien plus large (SWIR inclus)** → bien plus discriminant, surtout pour les **matériaux minéraux**, nombreux en ville.
2. **Variabilité plus grande** : visualisation t-SNE de descripteurs « faits main » calculés sur des patchs 64×64 :
   - partie spectrale : 6 indices (NDVI, ANVI, CI, NDVI_RE, VgNIR_BI, SAVI) + 20 bandes échantillonnées uniformément → 26 cartes ;
   - partie spatiale : 24 filtres de Gabor (4 fréquences de 1 à 10 m⁻¹ × 6 orientations) appliqués à la moyenne spectrale → 24 cartes ;
   - 50 cartes × 8 statistiques (moyenne, écart-type, min, max, 1er/9e déciles, 1er/3e quartiles) = **vecteur de 400 dimensions**.
   - Résultat : Toulouse occupe une région bien plus étendue ; la variabilité supplémentaire vient **surtout du spectral**, conséquence d'un *land cover* plus varié.
3. **Distribution à longue traîne** : la plus grosse classe ne pèse que 24 % (contre 43 % pour Houston), mais **beaucoup de classes ont très peu d'exemples** — plus réaliste.
4. **Labels moins bruités** : Houston contient beaucoup de pixels mal étiquetés ou mixtes, ce qui pénalise les modèles qui misent sur le spectral.

> 💡 Différence déséquilibre « classique » / longue traîne : la longue traîne se caractérise surtout par le **grand nombre** de classes rares.

---

<a id="s6"></a>
## 6. Auto-supervision pour apprendre des représentations spectrales

### 6.1 Panorama des méthodes (vision RGB)

| Méthode | Tâche prétexte |
|---|---|
| Rotation | Prédire la rotation appliquée (0°, 90°, 180°, 270°) |
| Exemplar | Chaque image + ses transformations = une « classe » |
| Contrastive (SimCLR…) | Rapprocher deux vues augmentées d'une même image, éloigner les autres |
| Distillation auto-supervisée (DINO) | Un réseau « élève » imite un « professeur » sur des vues différentes |

> 💡 Ces méthodes reposent sur l'**augmentation de données** et sont efficaces en RGB parce que la variabilité intra-classe y est **spatiale / contextuelle** (pose, orientation, fond).

### 6.2 Pourquoi ça coince en hyperspectral

- La variabilité intra-classe hyperspectrale est **spectrale** (physique, intrinsèque, sémantique), pas contextuelle.
- Des augmentations comme la distorsion de couleur ou le « miroir spectral » sont **physiquement douteuses** et ne préservent sans doute pas la sémantique.
- Simuler une variation d'éclairement réaliste exigerait de connaître la topographie ; les variations intrinsèques / sémantiques dépendent par nature de la chimie.

→ Les auteurs retiennent **deux méthodes sans augmentation** : **Deep Clustering** et **Masked Autoencoder (MAE)**.

### 6.3 Deep Clustering

Alterner : (1) clustering dans l'espace des features, (2) entraînement supervisé par ces **pseudo-labels**. Fonctionne en vision car les convolutions sont un **fort biais inductif** : même un CNN aléatoire produit des features utiles.

### 6.4 MAE en 1D (sur un spectre)

1. Le spectre $x$ ($1 \times B$) est découpé en **petites séquences** de bandes.
2. Une **grande partie** est masquée aléatoirement.
3. Les séquences visibles (+ *positional embedding*) passent dans un **encodeur transformer** ; un **token [CLS]** sert de représentation globale.
4. On ajoute des **mask tokens apprenables**, puis un **décodeur léger** reconstruit $\hat{x}$.

> 💡 Intuition : la combinaison de traits spectraux (pics d'absorption, inflexions) à différentes longueurs d'onde est liée à la chimie → reconstruire les parties masquées force à apprendre ces dépendances.

> ⚠️ Deux différences avec le texte / les images, soulignées par les auteurs :
> - une valeur de réflectance isolée **n'a pas de sens** (contrairement à un mot) ;
> - la **distance relative** entre deux bandes n'est pas porteuse de sens (une permutation aléatoire des bandes ne gêne pas un réseau dense).
> Le transformer n'est donc pas spécialement adapté au spectral, mais il est **pratique** pour la reconstruction masquée.

---

<a id="s7"></a>
## 7. Expériences et résultats

### 7.1 Protocole

- Comparer les représentations : **données brutes** vs **autoencodeur classique (AE)** vs **[CLS] du MAE**.
- Évaluation par **KNN** (une bonne représentation regroupe les données semblables) et **Random Forest** sur features gelées.
- Le RF sur features gelées bat largement la sonde linéaire et le *fine-tuning* → seuls les résultats RF sont rapportés.
- Hyperparamètres choisis par **recherche aléatoire** sur la validation.

### 7.2 Résultats (moyenne sur les 8 splits)

| Modèle | OA | F1 | | Modèle | OA | F1 |
|---|---|---|---|---|---|---|
| KNN | 0,78 | 0,69 | | RF | 0,75 | 0,65 |
| AE + KNN | 0,82 | 0,73 | | AE + RF | 0,81 | 0,73 |
| MAE + KNN | 0,84 | 0,76 | | **MAE + RF** | **0,85** | **0,77** |

→ Le MAE améliore nettement par rapport aux données brutes (+7 points de F1 avec KNN, +12 avec RF) et un peu par rapport à l'AE.

### 7.3 Échec du Deep Clustering

Classifieur dense entraîné sur des features **aléatoires** : F1 = 0,038 (CNN spectral aléatoire), 0,024 (réseau dense aléatoire) ; un classifieur au hasard ferait 0,024.

> 💡 Conclusion : convolutions spectrales et couches denses **n'apportent pas de bon a priori** sur les spectres → les pseudo-labels du clustering n'ont pas de sens → Deep Clustering + KNN fait **pire** que KNN sur données brutes.

### 7.4 Étude d'ablation du MAE

| Hyperparamètre | Constat |
|---|---|
| **Taux de masquage** | Erreur de reconstruction minimale à 0,5, mais **meilleure précision à 0,7–0,8** (compromis : ni trivial, ni trop dur) |
| **Têtes d'attention** | Meilleur quand **nb de têtes = dimension d'embedding** (chaque élément a une représentation **scalaire**) → cohérent avec l'idée que le spectral est une information de bas niveau |
| **Dimension latente** | Meilleurs résultats à **128 et 256** |

> 🔎 Rappel : dans un transformer multi-têtes, chaque tête travaille en dimension $d / h$ ($d$ = dimension d'embedding totale, $h$ = nombre de têtes). Avec $h = d$, chaque tête manipule un seul scalaire.

---

<a id="s8"></a>
## 8. Conclusions et perspectives de l'article

**Conclusions**

- Le jeu de Toulouse reflète mieux la complexité urbaine que les jeux publics existants.
- La reconstruction masquée (MAE) est une bonne tâche pour apprendre des représentations spectrales ; **baseline : 85 % OA, 77 % F1** (MAE + RF).
- Les méthodes à base de clustering sont inadaptées au spectral.

**Perspectives** (c'est de là que part le projet) :

- **Contexte spatial à grande échelle** : éviter des confusions absurdes (du « blé » dans un parc urbain, de la « tuile orange » dans un champ).
- Il faudrait des patchs d'**au moins 64 × 64 pixels** à ~1 m de GSD (les modèles de segmentation RGB utilisent plutôt 256 × 256 à ~50 cm).
- Problème : avec 310 bandes, ces patchs posent des **problèmes de mémoire**, ce qui empêche d'utiliser directement les gros modèles de vision (ex. SAM).
- **Piste prometteuse** : combiner un **modèle de vision profond pour le contexte** (*land use*, abstrait) et un **modèle léger pour le spectral** (*land cover*, bas niveau).
- Autres pistes : classification multi-label **hiérarchique**, **apprentissage à longue traîne**, **apprentissage actif** (grâce au *labeled pool*).

---

<a id="s9"></a>
## 9. Lien avec le projet fil rouge

**Question de recherche du projet** : comment combiner images hyperspectrales aéroportées et **données vectorielles d'usage du sol** (OpenStreetMap) pour produire des cartes d'occupation du sol **prêtes à l'emploi** par **segmentation panoptique** ?

**Deux verrous annoncés dans les diapos :**

1. Les **grandes dimensions spatiale et spectrale** des images (→ §8, problème de mémoire avec des patchs 64×64 × 310 bandes).
2. L'**écart entre la vérité terrain disponible et la sortie visée** (→ annotations éparses par pixel vs carte panoptique dense).

**Rappel : segmentation panoptique** (Kirillov et al.)

| Tâche | Sortie |
|---|---|
| Sémantique | Une classe par pixel, sans distinguer les objets |
| Instance | Un masque par objet dénombrable, le reste ignoré |
| **Panoptique** | Les deux : chaque pixel a une classe **et**, pour les objets dénombrables (*things*), un identifiant d'instance ; les régions amorphes (*stuff*) n'ont qu'une classe |

> 🔎 Une lecture possible pour ce projet (à valider avec l'encadrant) : les **objets d'usage du sol** (toits / bâtiments, parkings, piscines, champs, équipements sportifs) comme *things*, et le **matériau** (*land cover*) comme information à attribuer à chaque pixel ou à chaque instance. OSM fournit des polygones d'usage du sol qui pourraient servir de supervision au niveau instance, mais ils peuvent être incomplets ou mal alignés avec l'image.

> 🔎 Les diapos mettent aussi en avant les coefficients **de perméabilité, de ruissellement** et les **propriétés radiatives** : ce sont les grandeurs que les utilisateurs finaux (urbanistes, hydrologues) veulent tirer des classes de matériaux — d'où l'intérêt d'une carte « prête à l'emploi ».

**Références citées dans les diapos à lire ensuite :**

- Kirillov et al. — *Panoptic segmentation* (définition et métrique PQ).
- Garnot & Landrieu — segmentation panoptique de séries temporelles satellitaires (U-TAE / PaPs), utile comme exemple de panoptique en télédétection.
- Audebert, Le Saux & Lefèvre — apprentissage conjoint observation de la Terre + OpenStreetMap.

---

<a id="s10"></a>
## 10. Pièges à éviter

> 🚩 **Découper train / test au hasard** : les pixels voisins sont très corrélés → performances surestimées. Utiliser les **8 splits standards** spatialement disjoints.

> 🚩 **Ne regarder que l'OA** : avec une distribution à longue traîne, l'OA est dominée par les grosses classes. Toujours rapporter le **F1** (moyenné par classe).

> 🚩 **Confondre *labeled pool* et *unlabeled pool*** : le premier ne contient que des classes connues, le second peut contenir des classes **absentes de la nomenclature**.

> ⚠️ **Pixels mixtes exclus de la vérité terrain** : le modèle n'en voit pas à l'entraînement, mais une carte dense devra bien en prédire (bords de toits, objets fins).

> ⚠️ **Annotations éparses** : la vérité terrain ne couvre pas l'image de façon dense — une perte de segmentation classique doit ignorer les pixels non labellisés.

> ⚠️ **Augmentations « à la RGB »** (distorsion de couleur, miroir spectral) : physiquement peu crédibles en hyperspectral.

> ⚠️ **Mémoire** : un patch 64 × 64 × 310 pèse lourd ; prévoir réduction de dimension spectrale, encodeur spectral léger ou chargement par tuiles.

> ⚠️ **Spectres à trous** : les bandes d'absorption atmosphérique ont été retirées, d'où les discontinuités visibles sur les courbes de réflectance — attention si on suppose une continuité spectrale (convolutions 1D, interpolation…).

> ⚠️ **Valeur de référence du hasard** : sur ce jeu, un F1 de ~0,024 correspond au niveau d'un classifieur aléatoire.
