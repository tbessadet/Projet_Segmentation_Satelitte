# Lightweight Temporal Self-attention for Classifying Satellite Images Time Series

*V. Sainte Fare Garnot & L. Landrieu (AALTD 2020) — prise de notes à partir de ma dictée*

> **Légende** : 🔎 = précision ou réponse à une question que j'avais laissée en suspens

## Sommaire

- [0. Abstract](#0-abstract)
- [1. Introduction](#1-introduction)
- [2. Méthode](#2-méthode)
  - [Données d'entrée](#données-dentrée)
  - [2.1 Multi-headed self-attention (lecture de la figure 2)](#21-multi-headed-self-attention-lecture-de-la-figure-2)
  - [2.2 Lightweight attention](#22-lightweight-attention)
  - [2.3 Spatio-temporal classifier](#23-spatio-temporal-classifier)
- [3. Numerical experiment](#3-numerical-experiment)
  - [3.1 Dataset](#31-dataset)
  - [3.2 Metric and protocol](#32-metric-and-protocol)
  - [3.3 Evaluated methods](#33-evaluated-methods)
  - [3.4 Analysis](#34-analysis)
  - [3.5 Ablation study and robustness assessment](#35-ablation-study-and-robustness-assessment)
  - [3.6 Computational complexity](#36-computational-complexity)
- [4. Conclusion](#4-conclusion)

---

## 0. Abstract

- L'augmentation très importante des données d'observation de la Terre par satellite offre beaucoup d'opportunités, mais ça fait énormément de données à intégrer et à traiter → besoin de modèles capables de le faire **à grande échelle**.
- Les auteurs avaient déjà proposé un modèle, le **TAE** (*Temporal Attention Encoder*, encodeur d'attention temporelle). Dans cet article, ils y apportent des modifications.
- Les **canaux** de l'entrée sont **répartis entre plusieurs têtes d'attention compactes** qui travaillent en parallèle. Chaque tête extrait des **caractéristiques temporelles hautement spécialisées** ; les sorties de toutes les têtes sont ensuite regroupées (concaténées) en **une seule représentation**.
- Résultat : meilleures performances que l'état de l'art, avec **beaucoup moins de paramètres** (poids) et une **complexité computationnelle réduite**.

---

## 1. Introduction

**Le problème : le volume de données.**
- Les images satellite prises à intervalles réguliers donnent beaucoup d'informations, mais ce volume pose problème : les satellites **Sentinel** collectent plus de **25 Tb de données par an** rien que pour l'UE.
- Pour travailler à grande échelle, il faut des méthodes de traitement efficaces.

> 🔎 « À intervalles réguliers » est la formulation générale de l'introduction. Dans le jeu de données utilisé ensuite, les dates d'acquisition sont **irrégulières** (figure 1) : ce n'est pas une contradiction, mais c'est ce qui rend l'encodage des vraies dates utile (voir 2.1).

**Objectif du papier.** Extraire efficacement des **caractéristiques temporelles discriminantes** à partir de **séquences de descripteurs spatiaux**. La structure spatiale des données n'est pas le sujet de l'article.

**Méthodes existantes pour les séries temporelles.**
- Il existe plusieurs approches, mais le **deep learning** est particulièrement adapté aux gros volumes de données comme ici.
- Avec les réseaux de neurones, on peut :
  - modéliser la dimension temporelle **indépendamment** des dimensions spatiales : réseau de neurones récurrent (RNN) ou convolution 1D ;
  - ou modéliser le temps et l'espace **conjointement** : réseaux récurrents convolutifs ou convolutions 3D.

**Leur modèle : le L-TAE** (*Lightweight Temporal Attention Encoder*).
- Il se concentre sur l'**efficacité en mémoire et en calcul**, pour pouvoir traiter les grosses données en entrée.
- Leur méthode : réduire au maximum les **calculs redondants** et les **paramètres inutiles**, tout en gardant un haut niveau d'**expressivité**.

---

## 2. Méthode

### Données d'entrée

- Chaque image contient **10 canaux** : ce sont des images **super-spectrales** (plus que les 3 canaux du RGB, mais moins que les centaines de l'hyperspectral).
- On met côte à côte les $T$ images de la série temporelle, ce qui donne un « cube ». Dans le jeu de données de l'article : **24 images par parcelle, prises de janvier à octobre**, chacune avec 10 canaux.
- Chaque image passe dans un **embedding** (encodeur spatial) qui la transforme en un vecteur de taille $E$. L'entrée du module temporel est donc une matrice $E \times T$.

### 2.1 Multi-headed self-attention (lecture de la figure 2)

La figure 2 représente le **L-TAE**, avec $H = 3$ têtes et des clés de taille $K$.

**Principe des têtes.**
- On a **$H$ têtes d'attention**. Chaque tête travaille sur **un groupe distinct de canaux** de l'entrée.
- Chaque tête fait les mêmes opérations, mais avec **ses propres paramètres appris**.

**Ce qui se passe dans une tête** (entrée de taille $E' \times T$) :
1. On ajoute **P**, le **vecteur de position dans le temps** (voir plus bas).
2. Une **couche dense (fully connected), une seule couche**, produit $T$ vecteurs de taille $K$ : ce sont les **clés** (une clé par date).
3. La **requête** (*query*) « pose une question » : on la compare à chacune des $T$ clés, puis on applique un **softmax**. On obtient $T$ valeurs entre 0 et 1 dont la somme vaut 1 : c'est le **masque d'attention**, qui indique à quelles dates se trouve l'information recherchée.
4. On reprend l'**entrée de la tête** (flèche qui part de l'entrée dans le schéma), on la pondère par ce masque, puis on somme sur les $T$ dates : on passe d'une matrice $E' \times T$ à **un vecteur de taille $E'$**.

**Après les têtes.**
- On **concatène** les sorties de toutes les têtes : on obtient **un seul vecteur de taille $E$** (et non plus une matrice $E \times T$).
- Ce vecteur passe dans le **perceptron multicouche (MLP)**.

**Le vecteur de position P.**
- P donne la **position dans le temps** de chaque image : il encode le **nombre de jours écoulés** depuis le début de la séquence. On ne compare donc pas « l'image 1 à l'image 2 », mais « l'image à J+3 à l'image à J+7 ».
- Ça permet au modèle de « se repérer dans le temps », d'autant plus que **les images ne sont pas prises à intervalles réguliers** : P restitue la vraie chronologie.

**Intérêt : des têtes spécialisées.**
- Les $H$ têtes travaillent **en parallèle, indépendamment, avec des paramètres indépendants** : chacune apprend à faire ressortir des **informations différentes et spécialisées** des données d'entrée.
- Exemple (caricatural, pour l'intuition) : une tête qui indiquerait si la culture est précoce ou tardive, une autre si la culture émet beaucoup dans l'infrarouge en milieu de période, etc. Ces sorties aident ensuite le MLP à déterminer la catégorie de la culture.

> 🔎 L'idée « précoce / tardive » est cohérente avec ce que montre l'article plus loin (figure 4) : chaque tête se concentre sur une **fenêtre de temps courte et distincte**, qui dépend du type de culture. Pour l'exemple de l'infrarouge, attention : les canaux répartis entre les têtes sont des **caractéristiques apprises** par l'encodeur, pas directement les bandes spectrales.

### 2.2 Lightweight attention

Cette partie justifie les choix de la figure 2.

**Regroupement des canaux (*channel grouping*).**
- Le vecteur d'entrée de $E$ canaux est découpé en **$H$ groupes de même taille**, $E' = E/H$ ($H$ = nombre de têtes).
- Chaque tête travaille **en parallèle sur son seul groupe de canaux**, ce qui réduit le **coût computationnel**, notamment celui du calcul des clés (étape la plus coûteuse).
- Chaque tête ne voit qu'une partie des **canaux**, ce qui réduit les interactions à calculer, donc le nombre de calculs.
- Comme chaque tête se **spécialise** sur son groupe de canaux, on réduit les **opérations redondantes entre les têtes**.

**Le vecteur de position P.** C'est ici qu'ils expliquent comment il est construit : P est **dupliqué et ajouté à chaque groupe de canaux**, pour que chaque tête ait accès à l'information temporelle.

**Requête comme paramètre (*query-as-parameter*).**
- La requête devient un **paramètre appris** du modèle.
- Avantage : **moins de paramètres**, puisqu'on enlève la couche qui servait à calculer la requête.
- Inconvénient : moins de flexibilité, puisque cette requête est fixe. C'est **compensé par le nombre de têtes** : on a plusieurs requêtes, qui examinent chacune des parties différentes.

> 🔎 **Ce que ça veut dire.** Dans l'auto-attention classique, la requête est bien calculée par une couche linéaire à partir de chaque entrée : une requête par date, qui dépend des données. Dans le TAE, on calculait une seule requête « maître » en moyennant ces requêtes dans le temps. Dans le L-TAE, la requête de chaque tête est **directement un vecteur de taille $K$ appris pendant l'entraînement**, comme des poids. Elle est la même pour toutes les séquences et ne dépend plus de l'entrée. On supprime ainsi la **couche linéaire qui calculait les requêtes**.

**Masques d'attention et sorties.**
- Dans chaque tête, seules les **clés** sont obtenues par une couche apprise (couche dense) ; les **valeurs sont « bypassées »** ; la **requête** est un paramètre du modèle (elle ne passe pas par une couche dense).
- **Masque d'attention** de chaque tête = **softmax** du produit entre les clés et la requête de la tête.
- **Sortie de chaque tête** = **somme sur la dimension temporelle des entrées de la tête, pondérées par le masque d'attention**.
- On **concatène** les sorties des têtes dans un vecteur de taille $E$ (la même taille que l'entrée, mais sans la dimension temporelle $T$), qui passe ensuite dans le **MLP**.

> 🔎 **« Valeurs bypassées ».** Dans l'attention classique, on calcule aussi des valeurs $v^{(t)}$ avec une couche linéaire. Ici (comme dans le TAE), on saute cette couche : **les valeurs sont directement les entrées** ($v^{(t)} = e^{(t)}$, plus P). C'est la « flèche qui part de l'entrée » sur la figure 2. C'est possible parce que l'encodeur d'images est entraîné en même temps que le L-TAE : il apprend directement à produire des entrées utilisables comme valeurs.

**Résumé.** La **dimension temporelle est conservée jusqu'à la sortie de la tête**. La requête est appliquée aux $T$ clés, on passe dans le softmax (on a encore $T$ valeurs), on multiplie les entrées par ces valeurs, et **c'est seulement à ce moment-là** qu'on somme sur les $T$ images pour obtenir un seul vecteur. Jusque-là, chaque date garde son information de position (P).

### 2.3 Spatio-temporal classifier

Résumé du modèle complet :

$$X^{(t)} \xrightarrow{\ S\ } e^{(t)} \xrightarrow{\ \text{L-TAE}\ } o \xrightarrow{\ D\ } y$$

- **$X^{(t)}$** : image brute à la date $t$, disponible pour les $T$ dates.
- **Encodeur spatial S** : transforme chaque image (qui peut contenir des milliers de pixels) en un vecteur de **$E$ caractéristiques apprises** $e^{(t)}$. On a ce vecteur pour chaque date, donc une matrice $E \times T$.
- **L-TAE** (*Lightweight Temporal Attention Encoder*) : transforme cette séquence en **un seul vecteur $o$**, la sortie de son MLP.
- **Décodeur D** : transforme $o$ en un vecteur cible **$y$** de **20 scalaires**, un par classe (les **logits**), qui sert à la classification. Classiquement, le logit le plus élevé donne la classe prédite.

---

## 3. Numerical experiment

### 3.1 Dataset

- Ils comparent leur méthode à l'état de l'art sur le jeu de données public **Sentinel2-Agri** : **191 703 séquences** de **24 images super-spectrales** de parcelles agricoles, prises **entre janvier et octobre**.
- Résolution : **10 m par pixel**, **10 bandes spectrales**.
- Chaque parcelle est annotée dans une **nomenclature de 20 classes** de cultures agricoles.

### 3.2 Metric and protocol

**Métriques** (mesurées à l'échelle de la parcelle) :
- l'**Overall Accuracy (OA)** ;
- la **mean Intersection over Union (mIoU)**, **moyennée sur l'ensemble des classes** (*macro-averaging*).

- Le jeu de données est très **déséquilibré** : **4 classes représentent 90 % des exemples**, les 16 autres seulement 10 %. C'est pour ça que la **mIoU** est plus pertinente ici et donne une vraie idée des performances du modèle.

**Deux protocoles d'évaluation :**
1. **Comparaison à nombre de paramètres égal.** Pour une comparaison équitable, toutes les méthodes sont mises dans des configurations d'environ **150 000 paramètres** : le but est d'être sur un pied d'égalité. Le tableau des résultats donne aussi le **nombre théorique d'opérations (FLOPs)** nécessaire au module d'embedding de séquence pour traiter **une séquence** au moment de l'inférence.
2. **Comparaison de différentes tailles de modèles.** Ils testent plusieurs configurations de chaque algorithme d'embedding de séquence et tracent la performance en fonction du nombre de paramètres. Pour retirer l'effet des différents encodeurs spatiaux, ils utilisent **toujours le même encodeur d'entrée**, le **PSE** (*Pixel-Set Encoder*), et adaptent seulement sa **dernière couche dense** pour obtenir la dimension d'embedding voulue par le modèle testé ensuite.

### 3.3 Evaluated methods

Les modèles comparés :
- **PSE+TAE** : l'ancienne version (Garnot et al.). Le PSE encode chaque image, le TAE traite la séquence, et le **décodeur est un MLP à 2 couches**.
- **PSE+L-TAE** : leur nouveau réseau. Même architecture que le PSE+TAE, avec le L-TAE à la place du TAE.
- **CNN+GRU** : un **CNN** à la place du PSE et un **Gated Recurrent Unit (GRU)** à la place du TAE.
- **CNN+TempCNN** : un **CNN 2D** pour encoder les images et un **CNN 1D** pour traiter la dimension temporelle indépendamment.
- **Transformer**.
- **ConvLSTM**.
- **Random Forest** à **100 arbres**.

> 🔎 **ConvLSTM.** C'est un LSTM (réseau récurrent à mémoire) dont les multiplications matricielles internes sont remplacées par des **convolutions** : l'état caché est une carte spatiale. Il traite donc **l'espace et le temps conjointement**. C'est l'exemple de « réseau récurrent convolutif » cité dans l'introduction.

### 3.4 Analysis

**Comparaison à ~150 000 paramètres (tableau 1).**
- Toutes les méthodes sont évaluées par **validation croisée à 5 plis** (assez standard).
- Le **L-TAE est meilleur que toutes les autres méthodes** sur ce jeu de données, en OA comme en mIoU.
- Par rapport au TAE : l'**OA ne bouge quasiment pas**, mais la **mIoU augmente**, avec une **demande computationnelle bien moindre**. Le modèle est plus simple, et ses performances sont égales voire supérieures.

**Attention aux FLOPs.**
- Les FLOPs ne reflètent pas forcément la vitesse réelle du modèle : dans leur implémentation, le temps d'inférence est **dominé par le chargement des données et par l'embedding spatial d'entrée**, pas par le module temporel qu'on étudie ici.
- Les FLOPs sont donc **à titre indicatif**, mais ils **illustrent bien la simplicité et l'efficacité** du réseau.

**Performance en fonction du nombre de paramètres (figure 3).**
- Ils gardent les **4 meilleurs modules d'embedding de séquence** (L-TAE, TAE, GRU, TempCNN), testés dans différentes configurations (de 9k à 3M paramètres).
- Tous utilisent le **même décodeur et le même module spatial** : PSE + décodeur = **31 000 paramètres** au total.
- Le **plus petit L-TAE (9 000 paramètres)** obtient une meilleure mIoU qu'un **TAE à ~110 000 paramètres**, un **TempCNN à plus de 700 000** et un **GRU à 3 millions**.
- Sur les courbes mIoU / nombre de paramètres, le **L-TAE est au-dessus des autres pour toutes les tailles** de modèle.
- Bilan : d'abord, à ~150 000 paramètres, le L-TAE est le meilleur. Ensuite, le L-TAE à seulement 9 000 paramètres fait mieux que des modèles qui ont des nombres de paramètres d'un **ordre de grandeur beaucoup plus élevé**. Le L-TAE **maintient un haut niveau de précision malgré une très forte réduction du nombre de paramètres**.

### 3.5 Ablation study and robustness assessment

Ils font varier **trois hyperparamètres** (tableau 2) : le **nombre de têtes $H$**, la **dimension des clés $K$** et le **nombre de canaux $E$** de la séquence d'entrée. Les performances restent **assez stables pour toutes les configurations**.

**Nombre de têtes.**
- Il influence **très peu** les performances.
- Leur hypothèse : un grand nombre de têtes est bénéfique, mais une **petite taille de groupe $E'$ est en revanche pénalisante**, ce qui expliquerait que l'effet global reste limité.

> 🔎 Les deux sont liés : à $E$ fixé, augmenter $H$ diminue mécaniquement $E' = E/H$, le nombre de canaux que voit chaque tête.

**Dimension des clés.**
- Des clés **plus petites** que celles utilisées en NLP ou dans le TAE ($K = 32$) donnent de **meilleurs** résultats. Ici, le **maximum de mIoU est atteint pour $K = 8$**, qui bat $K = 32$, avec beaucoup moins de paramètres.
- Même des clés de **dimension 2** permettent d'obtenir des performances similaires au TAE.

**Dimension d'entrée $E$.**
- Assez logiquement, une entrée plus grande donne de meilleures performances : on conserve plus d'information et on a une représentation plus riche, mais moins compacte. Le **maximum est atteint à $E = 256$**, avec des **rendements décroissants** : $E = 512$ fait un peu moins bien que 256, avec presque deux fois plus de paramètres.

**Requête comme paramètre.**
- Pour tester ce choix d'architecture, ils entraînent une variante du L-TAE avec la **requête maître du TAE** à la place de la requête apprise. Résultat : un modèle **plus gros** (environ 170k paramètres) et une **mIoU plus faible** (49,7).
- Donc passer la requête en paramètre **réduit la taille du modèle** et **améliore les performances**.

### 3.6 Computational complexity

Ils comparent la **complexité asymptotique** de différents modules d'embedding de séquence (tableau 3), pour trois étapes :
- **Clés** : le L-TAE est en $O(TEK)$, contre $O(HTEK)$ pour le TAE et le Transformer. Grâce au regroupement des canaux, on gagne un **facteur $H$**.
- **Masque** : le TAE et le L-TAE (tous deux en $O(HTK)$) sont plus petits que le Transformer ($O(HT^2K)$) d'un **facteur $T$**.
- **Sortie** : le L-TAE est aussi plus petit d'un **facteur $H$** par rapport au TAE et au Transformer ($O(EX)$ contre $O(HEX)$).

---

## 4. Conclusion

- Ils proposent un **nouveau réseau léger** pour l'**embedding de séquences d'observations**, comme les séries temporelles d'images satellites.
- Grâce à deux choix, le **regroupement des canaux** entre les têtes et la **requête maître définie comme un paramètre entraînable**, leur méthode est **plus compacte** et **plus efficace en calcul** que les autres architectures basées sur l'attention.
- Évaluée sur un jeu de données satellites en accès libre, le **L-TAE fait mieux que les approches de l'état de l'art**, avec **beaucoup moins de paramètres** et une **charge de calcul réduite**.
- Cela ouvre la voie à une **analyse automatisée de l'observation de la Terre à l'échelle d'un continent**.
