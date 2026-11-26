## Sabine   
Creation data-set puis apprentissage auto-supervisé => classification de pixels  
Campagne terrain = étiquetages + aeroportées pour denses  
Prbl de reflectance entre données mesurée et exploitable :
- etalonnage appareil
- t
- correction atomosphérique

**Dataset :**  
- 7% labelisé
- pixels mixtes ecartés
8 découpages, contenant 2 trains sets (labelisé et labelise+non-labelise = auto-supervisé), 1 validation set et 1 test set  
Attention au decoupage des données (2 pixels d meme endroit dans le meme groupe, sinon dataleaks)  

**Variabilité données spectrales et spatiales :**  
Toulouse a plus de variabilité que les 2 autres  
permet d'avoir de meilleurs perfs sur les transformers (mais surapprend sur les CNN)  

### Apprentissage auto-supervisé :
Apprentissag d'une representatino en non-supervisé :
- augmentation : plutot adapté au RGB que hyperspectral
- MAE (masque auto-encodeur)
- 3 ieme solution ...

**DeepClustering :**  
Clustering des données pour avoir des pseudos-label, et verifier que c'est good  
Un CNN aléatoire permet de faire de la classification sans meme entrainement (le CNN par construction fait un embedding correct ...), c'est pourquoi il est adapté pour traiter sortir des features a partir d'images.  
Le clustering (Kmean) sur la sortie du convnet permet donc d'avoir des clusters cohérent, et donc d'avoir des pseudo-label pour chaque cluster : on fait de la classif en comparant yhat d'un MLP sortie convnet avec les pseudos-label : boucle backprog fonctionne et s'améliore.  
Mais avec de l'hyperspectral, le convnet aléatoire ne permet pas de faire un embedding correct au debut, donc ca marche pas   
$\to$ DeepClustering ecarté

**MAE :**  
Caché une partie du spectres (70-80%) et demander au modele de reconstruire.  
Si la tache est trop facile, on appprend pas des representations discriminantes, mais si la tache est trop dur, pas assez d'info pour apprendre => 70-80% retenu  

## Julien et Luccas  
Segmentation panoptique de séries temporelles d'images satellite  
- segmentation semantique : nom de la classe du pixel
- segmentation d'instance : contour de l'objet (qui contient les pixels, fait un lien entre eux ...)  
  
DataBase SITS (base sentinel-2 ...)  
Encodeur U-TAE ()  
PaPs (Parcels-as-Points) ->  

utilisation du TAE pour embedding sur dimension temporelle (+ ca permet le filtrage des images avec nuages ...)  

### PaPs : detecter chaque parcelle par son centre
Pixel de centre + noyau gaussien (en fonction largeur*hauteur) = heatmap  

**Tête panoptique :**  
convolution appris pour determiner les centres des parcelles  

Vecteur sortie issus d'un pixels de chaque convolution (pixels "centre"). MLP entrainé pour predire : 
- classe
- taille de la parcelle
- forme de la parcelle
  - Les deux ensemble permettre la reconstruction du contour/superficiede la parcelle  


## Cahier des charges  
1. **Approche transformer :** Essayer de reproduire les perfs (ou mieux) que MAE+RF avec un transformers. Les question : 
- Comment faire l'embedding d'entré ?
- Quelle architecture prendre ?
- fixer ou non des paramètres ?
- cbm de têtes 
- Comment adapter la reduction spectrale en place de la reduction temporelle
- 

2. Constitution d'une véritée terrain, a partir des données OpenStreetMap   

3. **Approche Similarité : (mode few-shoot learning)** Faire en sorte de sortir de la dependance des classes actuelles (permettant la generalisation du modele a plusieurs situations)  
n classes avec leur embedding.

4. **Combinaison :** Essasyer de combiner les 3 autres approches pour faire un modele plus performant.

