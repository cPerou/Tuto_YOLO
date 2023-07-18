### Introduction à YOLOv8

**Ultralytics** a plusieurs projet dont YOLO (You Only Look Once)

YOLOv8 repose sur le machine learning

Nous allons utiliser la détection d'image, elle permet de compter et localiser les objets d'une image.

![banner-tasks](https://github.com/cPerou/Tuto_YOLO/assets/137327551/889c9dde-a651-4293-ae9a-e6eaef94f6d0)
Crédit image : Ultralytics

Plus d'informations sur YOLO :
<https://github.com/ultralytics/ultralytics>

#### Langage utilisé pour coder YOLO ?

2 langages possibles : 
- Python : Jupiter, Thonny, Google Colaboratory
- CLI (invite de commandes)

Google Colaboratory permet d'utiliser un GPU si notre ordinateur n'en possède pas (traitement du modèle plus rapide).

### Prérequis

#### Les images

Recommendation : 
- 1 500 images par catégorie
- Maximum de situation
différente (jour, nuit, de dos...)
- Hauteur de 640 pixels (si ce n'est
pas le cas le modèle les transforme automatiquement et l'indique avec
des warning)

Redimensionner une grande quantité d'images :

```{bash}
#Installer ImageMagick <https://imagemagick.org/>

#Ouvrir son invite de commandes

#Aller dans le répertoire avec les images à redimensionner 
cd C:\M1\stage\Camera_trap\lynx_entrainement\imagesTest
#(ou clic droit dans le dossier /ouvrir dans le Terminal)

#Redimensionner toutes les images du dossier et les enregistrer dans un autre répertoire
magick mogrify -resize 853x640 -path C:\M1\stage\Camera_trap\lynx_entrainement\imagesTest \*.JPG
```

Plus d'informations :
<https://github.com/cPerou/Detection_objet_YOLO/blob/main/redimentionner_images.txt>

#### Les bounding box (boites englobantes)

Il existe de nombreux logiciels et sites qui permettent de réaliser des
bounding box. Elles doivent entourer l'objet de plus précisément
possible et ne pas en oublier.

J'ai utilisé l'application BoundingBoxEditor
<https://github.com/mfl28/BoundingBoxEditor>.

Il existe le site <https://www.cvat.ai/>.

Il y a également des sites avec des bounding box déjà faites :
<https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F02jvh9>
(Vérifier la qualité des Bounding box)

#### Exemple

![Boundiing_box_editor_3](https://github.com/cPerou/Tuto_YOLO/assets/137327551/c46d792c-7627-4056-bac1-49709a84f4ca)

Exporter les bounding box au format YOLO = fichiers labels

Ce label contient 2lignes donc 2 lynx.
Format YOLO : Numéro de la catégorie; Position x du centre de la boite;
Position y du centre; Hauteur; Largeur
Les coordonnées sont normalisées, ce qui permet d'utiliser les mêmes bounding box tant que le rapport
hauteur/largeur est concervé.

![Labels](https://github.com/cPerou/Tuto_YOLO/assets/137327551/e6bbf7c6-0e08-4299-9422-1b3098a8994f)

#### Organiser ses fichiers

![fichierOrganisation](https://github.com/cPerou/Tuto_YOLO/assets/137327551/78898fe1-e5c5-4197-b8a5-ae35d4bc15e6)

Important de respecter les noms des fichiers : images, labels, train,
val, (test).

Les bounding box sont dans le dossier "labels" (Les bounding box ont le
même nom que leur image de référence)

Plus d'informations : <https://docs.ultralytics.com/datasets/detect/>

#### Installer ultralytics si je travaille en local

<https://docs.ultralytics.com/quickstart>

### Entrainer un modèle YOLO (Train)

Tester le script avec peu d'images et 1 époque dans un
premier temps car le temps d'entrainement est très long.

Une époque correspond à un modèle avec certains paramètres, les
paramètres sont ajustés à chaque époque.

#### Google Colaboratory

Script type :
<https://colab.research.google.com/drive/1lV1ZKvf--I-8Glg-DGBuocyJD-0SI1va>
Permet d'utiliser YOLO sans l'avoir installé sur son ordinateur.

#### Local (Jupiter / Thonny)

```{bash}
from ultralytics import YOLO

# Charge un modèle nano à entrainer
model = YOLO("yolov8n.yaml")

# Entraine le modèle avec une époque
results = model.train(data="C:/M1/stage/Camera_trap/lynx_entrainement/dataLocal.yaml", epochs=1)  
```

Plus d'informations sur l'entrainement du modèle :
<https://docs.ultralytics.com/modes/train/>

### Valider le modèle (Val)

Afin de savoir si notre modèle est bien entrainé, on analyse les
fichiers.

#### Choses à regarder

![results](https://github.com/cPerou/Tuto_YOLO/assets/137327551/4b1b12d1-7384-4eae-a830-1d1a67ae0403) 
- Fonction de perte, liée au processus d'apprentissage (les 6 plots de
gauche ...loss).
La valeur doit diminuer, si elle atteint un plateau c'est que le modèle
n'apprend plus, le processus d'apprentissage est au maximum.
- Intersection d'une union (IoU) = précision de détection (pas visible avec YOLO)
- Précision moyenne principale (mAP) = prend en compte IoU et
indice de confiance
- Regarder directement les détections
![val_batch0_pred](https://github.com/cPerou/Tuto_YOLO/assets/137327551/aca582ee-b5bc-47b3-95fb-d2f5719e918f)

#### Comment améliorer le modèle si les résultats ne sont pas suffisants ?

-   0 à 10% d'images d'arrière-plan permetent de réduire les faux positifs (vide ou avec d'autres animaux, sans associer de bounding box)
-   Augmenter le nombre d'itérations (recommandation de 10 000 instances / classes)
-   1 500 images par classe
-   70% d'images train, 20 % val, 10% test

Plus d'informations :
<https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results>

### Utiliser le modèle entrainé (Predict)

#### Google colab

Script type :
[https://colab.research.google.com/drive/1zSByRGUnjLQltHIYk39rZ3i_Gab3JTUZ](https://colab.research.google.com/drive/1zSByRGUnjLQltHIYk39rZ3i_Gab3JTUZ)

Permet d'utiliser YOLO sans l'avoir installé sur son ordinateur.

#### Local (Jupiter / Thonny)

En cours

### Tips

Les deux pages web à retenir :
<https://github.com/ultralytics/ultralytics>
<https://docs.ultralytics.com/>

Un modèle entrainé avec des images peut être utilisé sur des vidéos.

Roboflow permet d'entrainer son propre modèle YOLO sans toucher à une
ligne de code. Il envoie le modèle entrainé sous une semaine
gratuitement. Proposé par Ultralytics en bas de leur page GitHub.
<https://roboflow.com/?ref=ultralytics>

Visualiser, comparer et optimiser ses modèles avec Comet gratuitement.
<https://www.comet.com/site/lp/yolov5-with-comet/?utm_source=yolov8&utm_medium=partner&utm_content=github>

Neural Magic's DeepSparse permetrai d'acceler les run avec le même
ordinateur
<https://docs.ultralytics.com/yolov5/tutorials/neural_magic_pruning_quantization/>

Tutoriel YOLO sur YouTube : détection, classification, segmentation, pose <https://www.youtube.com/watch?v=Z-65nqxUdl4&t=2972s>
