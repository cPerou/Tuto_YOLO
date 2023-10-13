### Introduction à YOLOv8

**Ultralytics** est une stratup qui développe et rend l'accès aux IA plus simple. \
En 2015, elle a développé la première version de YOLO (You Only Look Once) qui détecte des objets en temps réel. \
En 2023, une version encore plus performante est lancée : YOLOv8. 

5 types de modèles sont disponibles :

![banner-tasks](https://github.com/cPerou/Tuto_YOLO/assets/137327551/889c9dde-a651-4293-ae9a-e6eaef94f6d0) Crédit image : Ultralytics

Nous allons utiliser le modèle pré-entrainé de détection d'image 'YOLOv8n'. Il permet de compter et localiser les objets d'une image. \
Ce modèle repose sur un réseau de neurones convolutifs sans ancres, avec 225 couches et 3 157 200 paramètres. \
Si les résultats ne sont pas assez bon, il existe des modèles avec plus de couche et de paramètres (plus long à entrainer), YOLOv8s par exemple.

Plusieurs étapes sont nécessaires pour entrainer un modèle de deep learning : Train, Val, Predict, (Export), (Track), (Benchmark).

#### Langages informatiques utilisés ?

2 langages possibles : 
- Python : Jupyter, Thonny, Google Colaboratory 
- CLI (invite de commandes)

Google Colaboratory permet d'utiliser YOLO sans installer ultralytics sur son ordinateur, il met aussi à disposition un GPU.

Si tu as déjà ton modèle entrainé, tu peux directement aller à la section **"Utiliser le modèle entrainé (Predict)"**

### Prérequis

#### Les images

Recommendations :
- 1 500 images par catégorie
- Un maximum de situations différentes (jour, nuit, de dos...)
- 640 pixels en hauteur (si ce n'est pas le cas le modèle les transforme automatiquement et l'indique avec des warning)

Comment redimensionner une grande quantité d'images :

```{cmd}
#Installer ImageMagick <https://imagemagick.org/>

#Ouvrir son invite de commandes

#Aller dans le répertoire avec les images à redimensionner 
cd C:\M1\stage\Camera_trap\lynx_entrainement\imagesTest
#(ou clic droit dans le dossier /ouvrir dans le Terminal)

#Redimensionner toutes les images du dossier et les enregistrer dans un autre répertoire
magick mogrify -resize 853x640 -path C:\M1\stage\Camera_trap\lynx_entrainement\imagesTest \*.JPG
```

Informations complémentaires dans le script "redimentionner_images.txt".

#### Les bounding box (boites englobantes)

![Boundiing_box_editor_3](https://github.com/cPerou/Tuto_YOLO/assets/137327551/c46d792c-7627-4056-bac1-49709a84f4ca)

Les boites englobantes indiquent au modèle où est la réponse sur l'image. \
Il existe de nombreux logiciels et sites qui permettent de réaliser des bounding box. Celles-ci doivent entourer l'objet le plus précisément possible.

J'ai utilisé l'application BoundingBoxEditor <https://github.com/mfl28/BoundingBoxEditor>.

Il existe le site <https://www.cvat.ai/> pour réaliser des bounding box en ligne.

Des **banques de données** avec des bounding box déjà faites sont disponibles :
- <https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F02jvh9>
- <https://www.objects365.org/explore.html>
- <https://github.com/eg4000/SKU110K_CVPR19>
- Images satellites
<https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml>\
<http://xviewdataset.org/>

#### Exporter les bounding box au format YOLO = fichiers labels

Une ligne correspond à un bounding box.\
Format YOLO (xywhn) = Numéro de la catégorie; Position x du centre de la boite; Position y du centre; Hauteur; Largeur\
Les coordonnées sont normalisées, ce qui permet d'utiliser les mêmes bounding box tant que le rapport hauteur/largeur de l'image est concervé.

![Labels](https://github.com/cPerou/Tuto_YOLO/assets/137327551/e6bbf7c6-0e08-4299-9422-1b3098a8994f)

#### Organiser ses fichiers

![fichierOrganisation](https://github.com/cPerou/Tuto_YOLO/assets/137327551/78898fe1-e5c5-4197-b8a5-ae35d4bc15e6)

Il est important de respecter les noms des fichiers : images, labels, train, val, (test).

Les bounding box sont dans le dossier "labels" (Les bounding box ont le même nom que leur image correspondante)

Mettre aléatoirement 70% des images dans train et les 30% restant dans val (val pour validation).\
Une méthode est détaillée dans les scripts "aléatoire_images_val.txt" et "deplacer_labels_val.txt".

Plus d'informations sur les data : <https://docs.ultralytics.com/datasets/detect/>

#### Installer ultralytics si je travaille en local

<https://docs.ultralytics.com/quickstart>

### Entrainer un modèle YOLO (Train)

On peut utiliser un modèle vide (seulement la structure du modèle est déjà faite) ou un modèle pré-entrainé sur d'autres données.

Une **époque** correspond à l'essai d'un modèle avec certains paramètres. Les paramètres sont ajustés à chaque époque (Machine learning) afin de trouver le meilleur modèle.

Le fichier **"data.yaml"** indique le chemin des images train et val ainsi que le nombre et les noms des classes.

Dans un premier temps, tester le script avec peu d'images et 1 époque afin de minimiser les erreurs possibles.

On peut enregistrer les performances du modèle pendant l'entrainement avec Comet, ClearML ou TensorBoard.

Information supplémentaires sur l'utilisation des différents GPU, la reprise de l'entrainement, les arguments, l'enregistrement : <https://github.com/ultralytics/ultralytics/blob/main/docs/modes/train.md>

#### Google Colaboratory

Google Colab permet d'utiliser YOLO sans l'avoir installé sur son ordinateur.

Script type qui commence avec un modèle vide : <https://github.com/cPerou/Tuto_YOLO/blob/87817748182c359068d5bd12786f2d2ae274cad8/Google_colaboratory/Train_Google.ipynb>

Script qui utilise Comet et exporte le modèle : <https://github.com/cPerou/Tuto_YOLO/blob/0d4bba773f8bbf49a4dd4f641599ea4a151fce31/Google_colaboratory/Train_Google_Comet_Export.ipynb>

#### Local (Jupiter / Thonny)

```{python}
from ultralytics import YOLO

# Charge le modèle à entrainer 
# (Modèle nano : taille = 3 millions de paramètres, rapidité = 1ms/image)
model = YOLO("yolov8n.yaml")

# Entraine le modèle avec une époque
results = model.train(data="C:/M1/stage/Camera_trap/lynx_entrainement/dataLocal.yaml", epochs=1)  
```

Plus d'informations sur l'entrainement du modèle (codé en CLI) : <https://docs.ultralytics.com/modes/train/>

### Valider le modèle (Val)

Evaluation du modèle afin de mesurer sa précision et ses performances. \
On peut utiliser le mode val pour ajuster les hyperparamètres et améliorer les performances du modèle.

#### Data à analyser

![results](https://github.com/cPerou/Tuto_YOLO/assets/137327551/4b1b12d1-7384-4eae-a830-1d1a67ae0403) 
- Fonction de loss, liée au processus d'apprentissage, calcule l'erreur entre les prédictions de ton modèle et les valeurs réelles (les 6 plots de gauche ...loss). La valeur doit diminuer et se rapprocher de 0, si elle atteint un plateau c'est que le modèle n'apprend plus, le processus d'apprentissage est au maximum. Explication des différentes fonction de loss\
<https://inside-machinelearning.com/fonction-de-loss/>
- Matrice de confusion
- Intersection d'une union (IoU) = précision de détection (pas visible avec YOLO)
- Précision moyenne principale (mAP) = prend en compte IoU et indice de confiance
- Regarder directement les détections ![val_batch0_pred](https://github.com/cPerou/Tuto_YOLO/assets/137327551/aca582ee-b5bc-47b3-95fb-d2f5719e918f)

Plus d'informations sur la validation du modèle : <https://docs.ultralytics.com/modes/val/>

#### Comment améliorer la précision du modèle ?

-   0 à 10% d'images d'arrière-plan permetent de réduire les faux positifs (image d'arrière-plan : paysage ou avec d'autres animaux, sans associer de bounding box)
-   Augmenter le nombre d'itérations (recommandation de 10 000 instances / classes)
-   1 500 images par classe
-   70% d'images train, 20 % val, 10% test
-   Uiliser un modèle plus complexe (plus de paramètres), on a utiliser le plus petit, nano "yolov8n" avec 3 000 000 de paramètres mais il y a aussi les modèles s, m, l et x.
-   Ajuster le taux d'apprentissage, la taille des lots et d'autres hyperparamètres

Compromis à faire entre vitesse d'inférence et précision

Plus d'informations pour améliorer son modèle : <https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results>

### Utiliser le modèle entrainé (Predict)

YOLO peut être utilisé sur :
- des images .bpm .dng .jpeg .jpg .mpo .png .tif .tiff .webp .pfm
- des vidéos .asf .avi .gif .m4v .mkv .mov .mp4 .mpeg .mpg .ts .wmv .webm
- des fichiers .csv contenant des path
- des vidéos youtube...

Plus d'informations sur les fichiers input et les résultats : <https://github.com/ultralytics/ultralytics/blob/main/docs/modes/predict.md>

#### Google colab

Google Colab permet d'utiliser YOLO sans l'avoir installé sur son ordinateur.

Script type : <https://github.com/cPerou/Tuto_YOLO/blob/87817748182c359068d5bd12786f2d2ae274cad8/Google_colaboratory/Predict_Google.ipynb>

#### Local (Jupiter / Thonny)

En cours

### Exporter le modèle

On peut exporter notre modèle YOLOv8 sous d'autres formats (ONNX, OpenVINO, TensorRT). Cet autre format permet le déploiement du modèle dans d'autres applications.

```{cmd}

## Python

from ultralytics import YOLO

# Load un modèle personalisé
model = YOLO('path/to/best.pt')

# Exporte le modèle
model.export(format='onnx')


## CLI

yolo export model=path/to/best.pt format=onnx  # Exporte le modèle personalisé
```

Plus d'informations sur l'export : <https://docs.ultralytics.com/modes/export/>

### Benchmark

Benchmark peut nous aider à choisir le meilleur format d'exportation en fonction de la précision et de la vitesse d'exécution.

Plus d'informations sur Benchmark : <https://docs.ultralytics.com/modes/benchmark/>

### Tracking

Faire du tracking en temps réel.

<https://github.com/ultralytics/ultralytics/blob/main/docs/modes/track.md>

### Tips

Les deux pages web à retenir :\
<https://github.com/ultralytics/ultralytics>\
<https://docs.ultralytics.com/>

Un modèle entrainé avec des images peut être utilisé sur des vidéos.

Roboflow permet d'entrainer son propre modèle YOLO sans toucher à une ligne de code. Il envoie le modèle entrainé sous une semaine gratuitement.\
<https://roboflow.com/?ref=ultralytics>

Visualiser, comparer et optimiser ses modèles avec Comet gratuitement. <https://www.comet.com/site/lp/yolov5-with-comet/?utm_source=yolov8&utm_medium=partner&utm_content=github>

Neural Magic's DeepSparse permet d'accelérer les calculs sur notre ordinateur. [https://docs.ultralytics.com/yolov5/tutorials/neural_magic_pruning_quantization/](#0){.uri}

Tutoriel complet sur YouTube : détection, classification, segmentation, pose <https://www.youtube.com/watch?v=Z-65nqxUdl4&t=2972s>

La syntaxe des commandes, tous les arguments pour train, val, predict, export <https://github.com/ultralytics/ultralytics/blob/main/docs/usage/cfg.md> <https://github.com/ultralytics/ultralytics/blob/main/docs/usage/cli.md>

Réglage efficace des hyperparamètres avec Ray Tune et YOLOv8 <https://github.com/ultralytics/ultralytics/blob/main/docs/usage/hyperparameter_tuning.md>

Faire des cartes thermiques pour visualiser comment le modèle détecte les objets de l'image. <https://github.com/pooya-mohammadi/yolov5-gradcam/tree/master>

Page YouTube d'ultralytics [https://www.youtube.com/\@Ultralytics](https://www.youtube.com/@Ultralytics)
