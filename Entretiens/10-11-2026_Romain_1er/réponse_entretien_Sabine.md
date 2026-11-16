# Présentation plus en détail du projet

### Site toulouse :

utiliser Huggingface pour télécharger les données aller dans files and versions : [huggingface.co/datasets/Romain3Ch216/TlseHypDataSet](https://huggingface.co/datasets/Romain3Ch216/TlseHypDataSet)

bsq les données et hdr : métadonnées

--> ne télécharger que 1 tuile sur nos PC

#### dans hdr :

y a les numéros de bandes pour Rouge, Vert et Bleu

reflectance un float

data_ignore_avlue : 0.0000 --> dire que uqand y a 0 c'est qu'on a pas de données

UTM : la référence géométrique

bbl : bad band list --> 0 bande spectrale à retirer car soit trop bruité soit rayonnement absorbé par vapeurs d'eaux (début et fin très bruités)

visualiser les données sur QGIS

Nouveau projet et glisser/déposer le projet.

Attendtion ça va être bruité aller dans propriété symboligies et séelectionner couleur à bandes multiples et aller chercher les bandes rouges, verts et bleus (elles sont nommées (RED), (Green), (Blue)

Installer extension nommée : temporal/spectral profile tool pour avoir le spectre du pixel

#### Vérité terrain

librairie python pour faciliter la visualisation des données (attention sans doute des bugs)

et dans cette librairie y a la vérité terrain avec des données en shp (polygones) --> à ouvrir avec QGIS (attention récupérer tous les ficjiers

Changer les classes : propriétés symbologies catégories matériaux (attention 0 et 40 à retier pour l'instant) : catégories cf site toulouse

![Catégories des matériaux des](bibliographie/Categories_materiaux.png)

Ignorer l'anotation de l'usage des sols

40 : zone prises pour apprentissage non supervisé

Attention y a pleins de trucs mélangés dans les rayonnements

![Catégories des matériaux des](https://file+.vscode-resource.vscode-cdn.net/C%3A/Users/Sahbine/Projet_IODAA/Projet_Segmentation_Satelitte/image_vrac/ce_qui_est_mesure_par_camera.png)

Il y a donc un modèle pour nettoyer ça (attention le modèle prend pas en compye le fait que le sol est pas forcément plat)

Bibliographie -->

article important pour panoptique  : Garnot

Audebert : source principale d'utilisation de données OpenStreetMap et satellite (en gros les polygones sont pas bons pas suffisemment précis mais c'est plus précis si on le met

OSM : ya des arbres : ils viennent d'où ??? --> se renseigner sur données d'OSM d'où ça vient ?


Segment anything : article Kirillov et al - 2023- : faire des mask de segmentations à partir d'une image et d'un prompt --> très lourd pas forcément plus interessante


--> Chacun décortique son papier présentation au reste du groupe

Quand on fait de la biblio attention on peut se perdre. 

## Projet complexe et ambition pleins de briques à construire :

- modèle qui extrait représentation à partir d'un seul spectre : ce qui fonctionne bien pour l'instant mask auto-encoding : transformer image en un encoding, appliquer un transformer (dépend taille entrée et ordre) dans le MAE on va cacher un max de patch et le modèle soit reconstruire l'image. L'idée c'est d'appliquer ça à un spectre on découpe arbitrairement le spectre et on cache et il doit reconstruire. c'est instrinséquelent lié à l'architecture du transformer. Le transformer ets pas forcément la meilleure architecture pour un spectre par contre il est util pour apprendre sans supervision. La première étape serait d'avoir un transformer le plus efficace possible (dernier article)
