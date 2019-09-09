import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



from flask import Flask, render_template, url_for, request, redirect

import os, sys
from os import listdir, makedirs
from os.path import join, exists, expanduser

from functions import get_dominant_color, get_mean_color, black_or_white, recommand, recommand_couleur, predict_dog, predict
import random

app = Flask(__name__)


# ------------------------------------------------------------------------------
# ---------------------INDEX---------------------------------------------------------
# ------------------------------------------------------------------------------

@app.route('/index/')
def index():
    # images/perso/liste

    path = "./static/images/perso/"
    dirs = os.listdir(path)
    random.shuffle(dirs)

    cut = 9
    dirs2 = dirs[0:cut]

    nb_page = np.around(len(dirs)/len(dirs2))
    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    # number of pictures by line
    size = '33.33333%'
    size2 = '210px'

    return render_template('index.html',images=dirs2[:],cut=cut,npage=0,couleur=couleur_fond,couleur_logo=couleur_logo,
                           couleur_before=couleur_before,size=size,size2=size2)


@app.route('/index_pict/')
def index_pict():
    # images/perso/liste

    npage = request.args.get('npage')
    next = request.args.get('next')
    cut = int(request.args.get('cut'))


    if (str(next) == 'no'):
        npage = int(npage)
        npage = npage - 1
    elif (str(next) == 'yes'):
        npage = int(npage)
        npage = npage + 1
    else :
        npage = 0
    if (npage<0):
        npage = 0

    path = "./static/images/perso/"
    dirs = os.listdir(path)
    dirs2 = dirs[int(npage*cut):(npage+1)*cut]

    nb_page = np.around(len(dirs) / len(dirs2))

    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    # number of pictures by line
    size = '33.33333%'
    size2 = '210px'

    return render_template('index.html', images=dirs2[:], cut=cut,npage=npage, couleur=couleur_fond,couleur_logo=couleur_logo,
                           couleur_before=couleur_before,size=size,size2=size2)


# ------------------------------------------------------------------------------
# ----------------------------RECOMMAND--------------------------------------------------
# ------------------------------------------------------------------------------

@app.route('/index_recommand/')
def index_recommand():

    path_autre = "./static/images/autre/"
    path_perso = "./static/images/perso/"

    dirs_autre = os.listdir(path_autre)
    random.shuffle(dirs_autre)
    image_autre1 = dirs_autre[0]
    image_autre2 = dirs_autre[1]

    dirs_perso = os.listdir(path_perso)
    random.shuffle(dirs_perso)
    image_perso1 = dirs_perso[0]
    image_perso2 = dirs_perso[1]

    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    return render_template('index_recommand.html', couleur=couleur_fond,
                           couleur_logo=couleur_logo,
                           couleur_before=couleur_before,image_autre1=image_autre1,image_autre2=image_autre2,
                           image_perso1=image_perso1,image_perso2=image_perso2)




@app.route('/recommand_welcome/')
def recommand_index():
    # images/perso/liste
    dataset = request.args.get('perso')
    # form or color
    type = request.args.get('type')



    text = "Sélectioner une image pour obtenir une liste d'images similaires, Les recommadations sont basées sur l'utilisation d'un \
				réseau de neurones de type Resnet50 pré-entrainé sur ImageNet. Cela permet d'identifier les images contenant des objets ou des structures communes. \
				"

    if (type == 'color'):
        text = "Sélectioner une image pour obtenir une liste d'images similaires. Les recommadations sont basées sur la détection de la couleur dominante par  \
                clusteurisation. Cela permet d'identifier les images qui ont une couleur dominante communes. \
				"

    # picking up the dataset name

    # dataset path
    path = "./static/images/" + str(dataset) + "/"
    dirs = os.listdir(path)
    random.shuffle(dirs)

    if (dataset == 'perso'):
        cut = 12
    if (dataset == 'autre'):
        cut = 10

    dirs2 = dirs[0:cut]

    nb_page = np.around(len(dirs)/len(dirs2))
    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    # number of pictures by line (3 for dataset perso and 4 for dataset other)
    size = '20%'
    size2 = '210px'
    if (dataset == 'perso'):
        size = '33.33333%'
        size2 = '280px'
    if (dataset == 'autre'):
        size = '20%'
        size2 = '210px'



    return render_template('recommand_welcome.html',images=dirs2[:],cut=cut,npage=0,couleur=couleur_fond,couleur_logo=couleur_logo,
                           couleur_before=couleur_before,size=size,size2=size2,dataset=dataset,type=type,text=text)


@app.route('/recommand_welcome_bis/')
def recommand_index_bis():
    dataset = request.args.get('perso')

    type = request.args.get('type')


    npage = request.args.get('npage')
    next = request.args.get('next')
    cut = int(request.args.get('cut'))
    print (cut)

    if (str(next) == 'no'):
        npage = int(npage)
        npage = npage - 1
    elif (str(next) == 'yes'):
        npage = int(npage)
        npage = npage + 1
    else :
        npage = 0
    if (npage<0):
        npage = 0

    # dataset path
    path = "./static/images/" + str(dataset) + "/"
    dirs = os.listdir(path)

    dirs2 = dirs[int(npage*cut):(npage+1)*cut]

    nb_page = np.around(len(dirs) / len(dirs2))

    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    size = '210px'
    if (dataset == 'perso'):
        size = '33.33333%'
        size2 = '280px'
    if (dataset == 'autre'):
        size = '20%'
        size2 = '210px'

    return render_template('recommand_welcome.html', images=dirs2[:], cut=cut,npage=npage, couleur=couleur_fond,couleur_logo=couleur_logo,
                           couleur_before=couleur_before,size=size,size2=size2,dataset=dataset,type=type)



@app.route('/recommand_pict/')
def recommand_pict():

    # picking up the dataset name
    dataset = request.args.get('perso')
    print (dataset)

    # picking up the selected image name
    image_select = request.args.get('namepict')
    print (image_select)

    # picking up the selected image name
    type = request.args.get('type')

    # recommand six pictures for dataset perso
    # recommand 8 pictures for dataset other
    # using label_perso (or)
    # using label_other
    if (type == 'form'):
        list_recommand = recommand(image_select,dataset)
    if (type == 'color'):
        list_recommand = recommand_couleur(image_select,dataset)

    # dataset path
    path = "./static/images/" + str(dataset) + "/"

    # detect dominant and mean color of each picture
    dominant = np.zeros(3)
    mean_col = np.zeros(3)
    for pict in (list_recommand):
        path_picture = path + str(pict)

        # open image
        img = load_img(path_picture)  # this is a PIL image

        img = img_to_array(img)
        dom_col = get_dominant_color(img, k=4, image_processing_size=(50,50))
        dominant[0] = dominant[0] + dom_col[0]
        dominant[1] = dominant[1] + dom_col[1]
        dominant[2] = dominant[2] + dom_col[2]

        mean = get_mean_color(img)
        mean_col[:] = mean_col[:] + mean[:]

    dominant[:] = dominant[:] / len(list_recommand)
    mean_col[:] = mean_col[:] / len(list_recommand)


    color = black_or_white(color=dominant.copy())
    if (color == 'white'):
        couleur_before = 'rgba(255,255,255,1)'

    else:
        couleur_before = 'rgba(0,0,0,1)'

    couleur_fond = 'rgba({},{},{},1)'.format(mean_col[0],mean_col[1],mean_col[2])
    #couleur_logo = 'rgba({},{},{},1)'.format(dominant[0]+20,dominant[1]+20,dominant[2]+20)
    if ((mean_col[0] + 40) <255):
        mean_col[0] = mean_col[0] + 40
    else :
        mean_col[0] = mean_col[0] - 40

    if ((mean_col[1] + 40) <255):
        mean_col[1] = mean_col[1] + 40
    else :
        mean_col[1] = mean_col[1] - 40

    if ((mean_col[2] + 40) < 255):
        mean_col[2] = mean_col[2] + 40
    else :
        mean_col[2] = mean_col[2] - 40

    couleur_logo = 'rgba({},{},{},1)'.format(mean_col[0],mean_col[1],mean_col[2])

    # number of pictures by line (3 for dataset 'perso' and 4 for dataset 'other')
    size = '20%'
    size2 = '210px'
    if (dataset == 'perso'):
        size = '33.33333%'
        siz2 = '280px'
    if (dataset == 'other'):
        size = '20%'
        size2 = '210px'

    return render_template('recommand.html', images=list_recommand[:],couleur=couleur_fond,couleur_logo=couleur_logo,
                           couleur_before=couleur_before,size=size,size2=size2,dataset=dataset,type=type)


# ------------------------------------------------------------------------------

@app.route('/index_translat/')
def index_translat():
    # dataset path
    path_aurora = "./static/images/coloraurora_output/"
    path_sunset = "./static/images/colorsunset_output/"
    path_summer = "./static/images/colorsummer_output/"
    path_automn = "./static/images/colorautomn_output/"

    dirs_aurora = os.listdir(path_aurora)
    random.shuffle(dirs_aurora)
    image_aurora = dirs_aurora[0]

    dirs_sunset = os.listdir(path_sunset)
    random.shuffle(dirs_sunset)
    image_sunset = dirs_sunset[0]

    dirs_automn = os.listdir(path_automn)
    random.shuffle(dirs_automn)
    image_automn = dirs_automn[0]

    dirs_summer = os.listdir(path_summer)
    random.shuffle(dirs_summer)
    image_summer = dirs_summer[0]

    ind_page = request.args.get('ind_page')

    textA = "Nous tentons de répondre à cette problématique en utilisant des réseaux de neurones adversaires, par deux techniques différentes Pix2pix et CycleGAN.\
Ces deux techniques, bien que toutes deux génératives se basent sur des données de natures différentes. CycleGAN est un algorithme dont l'entrainement s'effectue via des données non appareillées et minimise/maximise une fonction de cout comportant un terme de reconstruction.  Pix2pix est conceptuellement plus simple puisqu'il repose sur des données appareillées, la foncton de cout mesure, en outre, l'adéquation entre l'image générée et l'image source. \
Nous avions préalablement utilisé des gans cycliques pour convertir des images de paysages d'été en paysages d'automne, de villes de jour en villes de nuits ou encore de paysages d'hiver en paysages d'été. Les résultats sur jeu de test étaient encourageant. Cependant le temps d'entrainement ainsi que la qualité de rendu dans le cadre d'entrainement non appareillé ne doit théoriquement pas pouvoir égaler ce même travail dans le cadre de données appareillées. En effet, l'information apportée par l'appariement est bien plus importante. Dans le cadre de données non appareillées, l'algorithme doit comprendre automatiquement la nature des objets constituants l'images. Il en déduit la manière dont les objets doivent variés. Dans le cadre de données appareillées, les objets émergent naturellement des invariances entre les deux images.\
Le problème est que nous ne disposons pas de données appareillées dans le sens propre du terme. Il nous faudrait par exemple pour effectuer un translation hiver > automne de disposer d'une banque d'images conséquentes et variée de paysage prise par le même appareil, au même endroit mais à des periodes différentes.\
Cependant, dans certains cas, nous pouvons 'simuler' un apparaiment. Il suffit de remarquer que dans certains types de translation, la forme des objets est invariante, de même que, en première approximation, les différence de constraste entre les objets."


    textB = "Le problème est ramené à un exercice de colorisation. On associe à l'image en couleur d'une certaine catégorie A la même image en noir et blanc. La phase de test générera une image colorisée de catégorie A en prenant, en entrée, une image test décolorisée de catégorie B.\
Ainsi, si l'on désire effectuer la translation d'une image d'une forêt en été en une forêt d'automne, la forme des arbres, des feuilles ainsi que la topographie de l'image ne change pas. Autrement dit, si l'on prend au hasard des images d'automne ou d'été et qu'on les décolorise il devient impossible de distinguer si ces images ont été prises en automne ou en été.  La translation automne/hiver se prête donc à l'utilisation de données non appareillées simulées.\
A l'inverse, la forme des objets, dans le cadre d'une translation hiver/été est modifiée. La neige sur les arbres transforme la forme des feuilles et celle au sol cache l'herbe ou les fleurs. Il est donc difficile d'utiliser pix2pix pour ce type de translation.\
De même, si l'on désire effectuer une translation ville de jour/ville de nuit, la forme des objets n'est pas modifiée. Pourtant le contraste entre les objets est lui modifié. Prenons l'exemple des fenêtes des imeubles. La nuit, certains appartements laissent entrevoir de la lumière tandis que d'autres restent sombres. Cela induit une différence de contraste entre les différentes fenêtres si l'on compare l'image de jour avec l'image de nuit.\
Les résultats sont entrainés sur un jeu de donné dont les images proviennent de pinterest ou google images et sont volontairement représentative d'un large spectre de paysages (canyons, plages, lacs, montagnes, forêts etc..). Nous entrainons plusieurs modèles sur les conversions suivantes : automne/ete, jour/coucher de soleil, coucher de soleil/aurore boreale. Nous comparons les résultats sur jeu de test, générées par les deux techniques et notons un gain de performance notable lors de l'utilisation de données appareillées simulées."

    A = 'non_act'
    B = 'non_act'

    if (ind_page != None):

        if (int(ind_page) == 1):
            A = 'active'
            B = 'non_act'
            text = textA
            text_title = "Recommandation d'images par contenu"
        if (int(ind_page) == 2):
            A = 'non_act'
            B = 'active'
            text = textB
            text_title = ""


    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    return render_template('index_translat.html', couleur=couleur_fond,
                           couleur_logo=couleur_logo,
                           couleur_before=couleur_before,image_summer=image_summer,image_automn=image_automn,
                           image_sunset=image_sunset,image_aurora=image_aurora,A=A,B=B,ind_page=int(ind_page),text=text,
                           text_title=text_title)

@app.route('/index_translat_color/')
def index_translat_color():
    # dataset path
    path_aurora = "./static/images/coloraurora_output/"
    path_sunset = "./static/images/colorsunset_output/"
    path_summer = "./static/images/colorsummer_output/"
    path_automn = "./static/images/colorautomn_output/"

    dirs_aurora = os.listdir(path_aurora)
    random.shuffle(dirs_aurora)
    image_aurora = dirs_aurora[0]

    dirs_sunset = os.listdir(path_sunset)
    random.shuffle(dirs_sunset)
    image_sunset = dirs_sunset[0]

    dirs_automn = os.listdir(path_automn)
    random.shuffle(dirs_automn)
    image_automn = dirs_automn[0]

    dirs_summer = os.listdir(path_summer)
    random.shuffle(dirs_summer)
    image_summer = dirs_summer[0]

    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    ind_page = request.args.get('ind_page')

    textA = "Nous tentons de répondre à cette problématique en utilisant des réseaux de neurones adversaires, par deux techniques différentes Pix2pix et CycleGAN.\
    Ces deux techniques, bien que toutes deux génératives se basent sur des données de natures différentes. CycleGAN est un algorithme dont l'entrainement s'effectue via des données non appareillées et minimise/maximise une fonction de cout comportant un terme de reconstruction.  Pix2pix est conceptuellement plus simple puisqu'il repose sur des données appareillées, la foncton de cout mesure, en outre, l'adéquation entre l'image générée et l'image source. \
    Nous avions préalablement utilisé des gans cycliques pour convertir des images de paysages d'été en paysages d'automne, de villes de jour en villes de nuits ou encore de paysages d'hiver en paysages d'été. Les résultats sur jeu de test étaient encourageant. Cependant le temps d'entrainement ainsi que la qualité de rendu dans le cadre d'entrainement non appareillé ne doit théoriquement pas pouvoir égaler ce même travail dans le cadre de données appareillées. En effet, l'information apportée par l'appariement est bien plus importante. Dans le cadre de données non appareillées, l'algorithme doit comprendre automatiquement la nature des objets constituants l'images. Il en déduit la manière dont les objets doivent variés. Dans le cadre de données appareillées, les objets émergent naturellement des invariances entre les deux images.\
    Le problème est que nous ne disposons pas de données appareillées dans le sens propre du terme. Il nous faudrait par exemple pour effectuer un translation hiver > automne de disposer d'une banque d'images conséquentes et variée de paysage prise par le même appareil, au même endroit mais à des periodes différentes.\
    Cependant, dans certains cas, nous pouvons 'simuler' un apparaiment. Il suffit de remarquer que dans certains types de translation, la forme des objets est invariante, de même que, en première approximation, les différence de constraste entre les objets."

    textB = "Le problème est ramené à un exercice de colorisation. On associe à l'image en couleur d'une certaine catégorie A la même image en noir et blanc. La phase de test générera une image colorisée de catégorie A en prenant, en entrée, une image test décolorisée de catégorie B.\
    Ainsi, si l'on désire effectuer la translation d'une image d'une forêt en été en une forêt d'automne, la forme des arbres, des feuilles ainsi que la topographie de l'image ne change pas. Autrement dit, si l'on prend au hasard des images d'automne ou d'été et qu'on les décolorise il devient impossible de distinguer si ces images ont été prises en automne ou en été.  La translation automne/hiver se prête donc à l'utilisation de données non appareillées simulées.\
    A l'inverse, la forme des objets, dans le cadre d'une translation hiver/été est modifiée. La neige sur les arbres transforme la forme des feuilles et celle au sol cache l'herbe ou les fleurs. Il est donc difficile d'utiliser pix2pix pour ce type de translation.\
    De même, si l'on désire effectuer une translation ville de jour/ville de nuit, la forme des objets n'est pas modifiée. Pourtant le contraste entre les objets est lui modifié. Prenons l'exemple des fenêtes des imeubles. La nuit, certains appartements laissent entrevoir de la lumière tandis que d'autres restent sombres. Cela induit une différence de contraste entre les différentes fenêtres si l'on compare l'image de jour avec l'image de nuit.\
    Les résultats sont entrainés sur un jeu de donné dont les images proviennent de pinterest ou google images et sont volontairement représentative d'un large spectre de paysages (canyons, plages, lacs, montagnes, forêts etc..). Nous entrainons plusieurs modèles sur les conversions suivantes : automne/ete, jour/coucher de soleil, coucher de soleil/aurore boreale. Nous comparons les résultats sur jeu de test, générées par les deux techniques et notons un gain de performance notable lors de l'utilisation de données appareillées simulées."

    A = 'non_act'
    B = 'non_act'

    if (ind_page != None):

        if (int(ind_page) == 1):
            A = 'active'
            B = 'non_act'
            text = textA
            text_title = "Recommandation d'images par contenu"
        if (int(ind_page) == 2):
            A = 'non_act'
            B = 'active'
            text = textB
            text_title = ""

    return render_template('index_translat_color.html', couleur=couleur_fond,
                           couleur_logo=couleur_logo,
                           couleur_before=couleur_before,image_summer=image_summer,image_automn=image_automn,
                           image_sunset=image_sunset,image_aurora=image_aurora,A=A,B=B,ind_page=ind_page,text=text,
                           text_title=text_title)



@app.route('/translat/')
def translat():
    # picking up dataset names
    dataset = request.args.get('perso')
    dataset2 = request.args.get('perso2')

    orig = request.args.get('orig')

    if (orig == 'color'):
        url = "\index_translat_color\?ind_page=1"
    elif (orig == 'translat'):
        url = "\index_translat\?ind_page=1"
    elif (orig == 'reconstruction') or (orig == 'rotation'):
        url = "\index_rotproject2\?ind_page=2"
    elif (orig == 'zoom'):
        url = "\index_rotproject\?ind_page=1"


    # dataset path
    path = "./static/images/" + str(dataset) + "/"
    dirs = os.listdir(path)
    random.shuffle(dirs)


    cut = 25
    if (dataset == 'rot_input') or (dataset == 'rot_output'):
        cut = 40

    dirs2 = dirs[0:cut]

    nb_page = np.around(len(dirs)/len(dirs2))
    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    # number of pictures by line (3 for dataset perso and 4 for dataset other)
    size = '20%'
    size2 = '210px'
    if (dataset == 'perso'):
        size = '33.33333%'
        size2 = '280px'
    if (dataset == 'autre'):
        size = '20%'
        size2 = '210px'

    if (dataset == 'rot_input'):
        size = '10%'
        size2 = '105px'


    if (dataset == 'rot_output'):
        size = '10%'
        size2 = '105px'


    return render_template('translat.html',images=dirs2[:],cut=cut,npage=0,couleur=couleur_fond,couleur_logo=couleur_logo,
                           couleur_before=couleur_before,size=size,size2=size2,dataset=dataset,dataset2=dataset2,url=url,
                           orig=orig)

@app.route('/translat_bis/')
def translat_bis():
    dataset = request.args.get('perso')
    dataset2 = request.args.get('perso2')

    npage = request.args.get('npage')
    next = request.args.get('next')
    cut = int(request.args.get('cut'))

    orig = request.args.get('orig')
    if (orig == 'color'):
        url = "\index_translat_color\?ind_page=1"
    elif (orig == 'translat'):
        url = "\index_translat\?ind_page=1"
    elif (orig == 'rotation'):
        url = "\index_rotproject2\?ind_page=2"
    elif (orig == 'zoom'):
        url = "\index_rotproject\?ind_page=1"

    print (cut)

    if (str(next) == 'no'):
        npage = int(npage)
        npage = npage - 1
    elif (str(next) == 'yes'):
        npage = int(npage)
        npage = npage + 1
    else:
        npage = 0
    if (npage < 0):
        npage = 0

    # dataset path
    path = "./static/images/" + str(dataset) + "/"
    dirs = os.listdir(path)

    dirs2 = dirs[int(npage * cut):(npage + 1) * cut]

    nb_page = np.around(len(dirs) / len(dirs2))

    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    size = '20%'
    size2 = '210px'
    if (dataset == 'perso'):
        size = '33.33333%'
        size2 = '280px'
    if (dataset == 'autre'):
        size = '20%'
        size2 = '210px'

    if (dataset == 'rot_input'):
        size = '10%'
        size2 = '105px'


    if (dataset == 'rot_output'):
        size = '10%'
        size2 = '105px'



    return render_template('translat.html', images=dirs2[:], cut=cut, npage=npage, couleur=couleur_fond,
                           couleur_logo=couleur_logo,
                           couleur_before=couleur_before, size=size, size2=size2, dataset=dataset, dataset2=dataset2,
                           url=url,orig=orig)



@app.route('/index_rotproject/')
def index_rotproject():


    ind_page = (request.args.get('ind_page'))


    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    text1 = "La problématique de ce projet est de reconstituer une image représentant un paysage incliné en la même image représentant le même paysage non incliné. De plus, on cherche à effectuer ce travail sans tronquer l'image initiale.\
Lors de la prise de photo, particuliérement dans le cadre de l'utilisation du zoom, une légére rotation non désirée de l'appareil peut aboutir à un résultat plus ou moins inclinée. Cela peut aboutir au rejet de la photo car la symétrie de ce qui est représenté est brisée et l'esthetique altérée.\
Bien entendu, des applications existent pour recadrer la photo. Cependant celles-ci doivent effectuer une combinaisons de deux procésus : une rotation et un zoom.\
La phase de zoom, s'accompagne donc necessairement d'une troncation de l'image et donc d'une perte d'information.\
L'objectif de ce projet est double : premièrement détecter automatiquement l'angle d'inclinaison de l'image. Cet angle varie entre -pi/2 et +pi/2 radians. Cela constitue un exercice non trivial puisque l'algorithme doit comprendre les objets inhérants à l'image. Il doit par exemple identifier des arbres sur une image de forêt et comprendre que ces derniers poussent à la verticale, il doit aussi pouvoir comprendre la topographie de ce que représente l'image et pouvoir identifier des montagnes (la zone de démarcation entre le ciel et la terre n'est donc pas forcément horizontale).\
Pour effectuer ce travail nous utilisons un réseau de neurones pré-entrainés sur imageNet, et réalisons un exercice assez analogue à celui que nous avions réalisés dans le cadre de l'identification de race de chiens. Nous utilisons un réseaux de type ResNet50 car ce dernier est de poid relativement modéré en comparaison d'un Xception. Nous assumons que le gain de performance de réseaux plus profonds ou contenant plus de paramètres n'est pas utile puisque nous cherchons à identifier des structures assez générales (ciel, nuages, arbres, collines). Nous assumons par ailleurs que le corpus d'entrainement (ImageNet) contient suffisament d'images de paysages pour identifier les structures caractéristiques. Nous ajoutons deux couches denses de 500 et respectivement 50 neurones, un dropout de 0.25 et une batch-normalisation, nous utilisons en sortie un seul neurone dont le rôle est de fournir l'angle de rotation. Nous ramenons les angles en radians à une valeur entre 0 et 1 et utilisons une fonction d'activation de type sigmoide. Nous utilisons une erreur de type RMSE. Nous partagons dés le début notre dataset d'images appareillées générées en trois groupes (entrainement, validation et test) et calibrons l'arret de l'entrainement en visualisant la fonction de cout sur le jeu de validation. Nous analysons les résultats sur le jeu de test. Après 150 epochs et en utilisants une taille de mini-batch de 10 nous obtenons une erreur de quelques degrés, bien inferieure à l'erreur statistique moyenne d'environ 25 degrés. Notons que le corpus d'entrainement contient des images de paysages variées dont l'angle d'inclinaison peut parfois être difficilement interprétable à l'oeuil nu. Nous pourrions envisager d'obtenir de meilleurs résultats en réentrainnant les dernières couches du resnet50 en partant du principe que les structures qui interviennent dans le calcul de l'angle ne sont pas forcément bien représentées par la sortie du resnet50.\
            Il est tout à fait possible de générer des données d'entrainement. Il suffit de prendre des images dont le contenue représenté est jugé droit, d'effectuer une combinaison rotaton zoom de sorte à obtenir une nouvelle image dont le contenu est incliné de l'angle désiré. Nous effectuons ce travail en faisant varier l'angle d'inclinaison de -pi/5 à +pi/5 de manière complétement aléatoire. Notre dataset initial contient 3600 images et nous effectuons 3 itérations ce qui est équivalant à effectuer de la data-augmentation."

    text2 = "La seconde partie de ce projet est de faire pivoter l'image pour reconstituer une image au sein de laquelle les objets représentés ne sont pas inclinés sans aucune perte de donnée. Si l'on souhaite effectuer ce travail, on aboutie à la création d'une image de taille plus grande contenant l'image initiale réinclinée. En considérant une image initiale de dimmension l*L, la nouvelle image est de dimension (l + Lcos(alpha)), L+lcos(alpha)).\
La nouvelle image contient donc 4 espaces vides (au niveau des angles) d'autant plus grands que l'angle d'inclinaison est important. Il faut donc reconstituer ces espaces. \
Ce probléme peut être résolu par des algorithmes génératifs. Dans l'espoir d'un gain de performances, nous aimerions nous ramener, si possible à un problème de données appareillées. Par chance, il est tout à fait possible de simuler ces données. Il suffit de prendre des images dont le contenu représenté est jugé droit et de venir découper à l'interieur de telle sorte d'obtenir une image tronquée dont les tailles des éléments supprimées dépendent de l'angle de rotation que l'on applique. On applique aléatoirement des angles de rotations compris entre -pi/5 et pi/5.\
On utilise ensuite un algorithme  pix2pix en laissant l'hyperparamétre pondérant la fonction de cout 'adversarial' et la fonction de cout 'L1' à sa valeur initiale. Nous n'utilisons pas de métriques particulière pour analyser le contenu générée mais observons toutes les epochs les images générées sur un jeu de test pour éviter l'overfitting. Cette méthode artisanale est suceptible d'être automatisée en utilisant des scores dédiés à cet effet. Cependant il n'y a toujours pas de concensus dans l'évaluation des images générées par des algorithmes génératifs."

    A = 'non_act'
    B = 'non_act'

    if (ind_page != None):

        if (int(ind_page) == 1):
            A = 'active'
            B = 'non_act'
            text = text1
            text_title = "(1/2)"

        if (int(ind_page) == 2):
            A = 'non_act'
            B = 'active'
            text = text2

    return render_template('index_rotproject.html',ind_page=ind_page, couleur=couleur_fond,
                           couleur_logo=couleur_logo,
                           couleur_before=couleur_before,A=A,B=B,text=text,text_title=text_title)

@app.route('/index_rotproject2/')
def index_rotproject2():


    ind_page = (request.args.get('ind_page'))


    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    text1 = "La problématique de ce projet est de reconstituer une image représentant un paysage incliné en la même image représentant le même paysage non incliné. De plus, on cherche à effectuer ce travail sans tronquer l'image initiale.\
Lors de la prise de photo, particuliérement dans le cadre de l'utilisation du zoom, une légére rotation non désirée de l'appareil peut aboutir à un résultat plus ou moins inclinée. Cela peut aboutir au rejet de la photo car la symétrie de ce qui est représenté est brisée et l'esthetique altérée.\
Bien entendu, des applications existent pour recadrer la photo. Cependant celles-ci doivent effectuer une combinaisons de deux procésus : une rotation et un zoom.\
La phase de zoom, s'accompagne donc necessairement d'une troncation de l'image et donc d'une perte d'information.\
L'objectif de ce projet est double : premièrement détecter automatiquement l'angle d'inclinaison de l'image. Cet angle varie entre -pi/2 et +pi/2 radians. Cela constitue un exercice non trivial puisque l'algorithme doit comprendre les objets inhérants à l'image. Il doit par exemple identifier des arbres sur une image de forêt et comprendre que ces derniers poussent à la verticale, il doit aussi pouvoir comprendre la topographie de ce que représente l'image et pouvoir identifier des montagnes (la zone de démarcation entre le ciel et la terre n'est donc pas forcément horizontale).\
Pour effectuer ce travail nous utilisons un réseau de neurones pré-entrainés sur imageNet, et réalisons un exercice assez analogue à celui que nous avions réalisés dans le cadre de l'identification de race de chiens. Nous utilisons un réseaux de type ResNet50 car ce dernier est de poid relativement modéré en comparaison d'un Xception. Nous assumons que le gain de performance de réseaux plus profonds ou contenant plus de paramètres n'est pas utile puisque nous cherchons à identifier des structures assez générales (ciel, nuages, arbres, collines). Nous assumons par ailleurs que le corpus d'entrainement (ImageNet) contient suffisament d'images de paysages pour identifier les structures caractéristiques. Nous ajoutons deux couches denses de 500 et respectivement 50 neurones, un dropout de 0.25 et une batch-normalisation, nous utilisons en sortie un seul neurone dont le rôle est de fournir l'angle de rotation. Nous ramenons les angles en radians à une valeur entre 0 et 1 et utilisons une fonction d'activation de type sigmoide. Nous utilisons une erreur de type RMSE. Nous partagons dés le début notre dataset d'images appareillées générées en trois groupes (entrainement, validation et test) et calibrons l'arret de l'entrainement en visualisant la fonction de cout sur le jeu de validation. Nous analysons les résultats sur le jeu de test. Après 150 epochs et en utilisants une taille de mini-batch de 10 nous obtenons une erreur de quelques degrés, bien inferieure à l'erreur statistique moyenne d'environ 25 degrés. Notons que le corpus d'entrainement contient des images de paysages variées dont l'angle d'inclinaison peut parfois être difficilement interprétable à l'oeuil nu. Nous pourrions envisager d'obtenir de meilleurs résultats en réentrainnant les dernières couches du resnet50 en partant du principe que les structures qui interviennent dans le calcul de l'angle ne sont pas forcément bien représentées par la sortie du resnet50.\
            Il est tout à fait possible de générer des données d'entrainement. Il suffit de prendre des images dont le contenue représenté est jugé droit, d'effectuer une combinaison rotaton zoom de sorte à obtenir une nouvelle image dont le contenu est incliné de l'angle désiré. Nous effectuons ce travail en faisant varier l'angle d'inclinaison de -pi/5 à +pi/5 de manière complétement aléatoire. Notre dataset initial contient 3600 images et nous effectuons 3 itérations ce qui est équivalant à effectuer de la data-augmentation."

    text2 = "La seconde partie de ce projet est de faire pivoter l'image pour reconstituer une image au sein de laquelle les objets représentés ne sont pas inclinés sans aucune perte de donnée. Si l'on souhaite effectuer ce travail, on aboutie à la création d'une image de taille plus grande contenant l'image initiale réinclinée. En considérant une image initiale de dimmension l*L, la nouvelle image est de dimension (l + Lcos(alpha)), L+lcos(alpha)).\
La nouvelle image contient donc 4 espaces vides (au niveau des angles) d'autant plus grands que l'angle d'inclinaison est important. Il faut donc reconstituer ces espaces. \
Ce probléme peut être résolu par des algorithmes génératifs. Dans l'espoir d'un gain de performances, nous aimerions nous ramener, si possible à un problème de données appareillées. Par chance, il est tout à fait possible de simuler ces données. Il suffit de prendre des images dont le contenu représenté est jugé droit et de venir découper à l'interieur de telle sorte d'obtenir une image tronquée dont les tailles des éléments supprimées dépendent de l'angle de rotation que l'on applique. On applique aléatoirement des angles de rotations compris entre -pi/5 et pi/5.\
On utilise ensuite un algorithme  pix2pix en laissant l'hyperparamétre pondérant la fonction de cout 'adversarial' et la fonction de cout 'L1' à sa valeur initiale. Nous n'utilisons pas de métriques particulière pour analyser le contenu générée mais observons toutes les epochs les images générées sur un jeu de test pour éviter l'overfitting. Cette méthode artisanale est suceptible d'être automatisée en utilisant des scores dédiés à cet effet. Cependant il n'y a toujours pas de concensus dans l'évaluation des images générées par des algorithmes génératifs."

    A = 'non_act'
    B = 'non_act'

    if (ind_page != None):

        if (int(ind_page) == 1):
            A = 'active'
            B = 'non_act'
            text = text1
            text_title = "(1/2)"

        if (int(ind_page) == 2):
            A = 'non_act'
            B = 'active'
            text = text2
            text_title = "(2/2)"




    return render_template('index_rotproject2.html',ind_page=ind_page, couleur=couleur_fond,
                           couleur_logo=couleur_logo,
                           couleur_before=couleur_before,A=A,B=B,text=text,text_title=text_title)



app.config["ALLOWED_IMAGE_EXTENSIONS"] = ['PNG','JPG','JPEG']
app.config["MAX_IMAGE_FILESIZE"] = 10 * 1024 * 1024

def allowed(image):

    if not "." in (image.filename):
        return (False)

    ext = image.filename.rsplit(".",1)[1]

    if ext.upper() in (app.config["ALLOWED_IMAGE_EXTENSIONS"]):
        return (True)
    else :
        return (False)

def filesize_limit(filesize):
    if (int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]):
        return True
    else :
        return False


@app.route('/api_result/',methods=['GET','POST'])
def api_result():

    text = ''

    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"


    if (request.method == 'POST'):
        if (request.files):

            print ('avant')
            at = request.cookies.get('filesize')
            print ('après')

            print (request.cookies.get('filesize'))

            #if not (filesize_limit(request.cookies.get('filesize'))):
               # text = "votre fichier est trop lourd"
               # return render_template('index_api.html', couleur=couleur_fond,
                   #        couleur_logo=couleur_logo,text=text,
                    #       couleur_before=couleur_before)

            image = request.files['image']


            if (image.filename == ''):
                text = "l'image n'a pas de nom"
                print (text)
                return render_template('index_api.html', couleur=couleur_fond,
                                       couleur_logo=couleur_logo, text=text,
                                       couleur_before=couleur_before)
            if not (allowed(image=image)):
                text ="Ce format n'est pas accepté, veuillez charger une image au format JPG ou PNG"
                print (text)
                return render_template('index_api.html', couleur=couleur_fond,
                           couleur_logo=couleur_logo,text=text,
                           couleur_before=couleur_before)

            image.save('./static/images/image_test.png')



            text = ""
            print (text)
            print (image)



            dog_breed, dog_breed_prob = predict_dog()
            br_1 = dog_breed[0]
            br_2 = dog_breed[1]
            br_3 = dog_breed[2]

            p_1 = dog_breed_prob[0]
            p_2 = dog_breed_prob[1]
            p_3 = dog_breed_prob[2]

            print (dog_breed)

            return render_template('api.html', couleur=couleur_fond,
                                   couleur_logo=couleur_logo, text=text,
                                   couleur_before=couleur_before, race1=br_1, race2=br_2, race3=br_3, prob1=p_1,
                                    prob2=p_2, prob3=p_3)

        else:
            text = "il semble que vous n'avez pas chargé de fichier"
            return render_template('index_api.html', couleur=couleur_fond,
                           couleur_logo=couleur_logo,text=text,
                           couleur_before=couleur_before)
    else :
        text = "le chargement ne fonctionne pas"
        return render_template('index_api.html', couleur=couleur_fond,
                           couleur_logo=couleur_logo,text=text,
                           couleur_before=couleur_before)




    return render_template('index_api.html', couleur=couleur_fond,
                           couleur_logo=couleur_logo,text=text,
                           couleur_before=couleur_before),#race1=br_1,race2=br_2,race3=br_3,prob1=p_1,prob2=p_2,prob3=p_3)



@app.route('/index_api/')
def index_api():



    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    return render_template('index_api.html', couleur=couleur_fond,
                           couleur_logo=couleur_logo,
                           couleur_before=couleur_before)


@app.route('/cycle_gan/')
def cycle_gan():
    # images/perso/liste

    # dataset path
    path = "./static/images/cycleGAN/"
    dirs = os.listdir(path)
    random.shuffle(dirs)



    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"

    size = '100%'
    size2 = '725px'



    return render_template('index_rotproject_cycleGAN.html',couleur=couleur_fond,couleur_logo=couleur_logo,
                           couleur_before=couleur_before,images=dirs,size=size,size2=size2)


@app.route('/contact/')
def contact():
    # images/perso/liste


    couleur_fond = '#214093'
    couleur_logo = "#b2b1ff"
    couleur_before = "#b1b1ff"



    return render_template('contact.html',couleur=couleur_fond,couleur_logo=couleur_logo,
                           couleur_before=couleur_before)




if (__name__ == '__main__'):
    app.run(port=1000)


















