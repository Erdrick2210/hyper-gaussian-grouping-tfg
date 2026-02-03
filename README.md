# 3D Gaussian Splatting per a dades multiespectrals

Aquesta implementació està basada en el repositori de Gaussian Grouping i amplia el model de **3D Gaussian Splatting (3DGS)** perquè pugui operar eficientment amb dades **multiespectrals i hiperespectrals**.  
L'objectiu principal és millorar la qualitat de reconstrucció i el valor semàntic de les escenes generades, permetent extreure informació sobre els materials presents.

## Índex

- [Extensió principal](#extensió-principal)
- [Estructura del projecte](#estructura-del-projecte)
- [Datasets](#datasets)
- [Entrenament](#entrenament)
- [Avaluació](#avaluació)
- [Experiments i Resultats](#experiments-i-resultats)

---

## Extensió principal  
- Cada gaussiana aprèn un **embedding espectral latent** en lloc de només un color RGB.  
- Aquest embedding és processat per un **MLP** per predir múltiples bandes espectrals.  
- Permet una representació compacta, eficient i apta per tasques posteriors com segmentació semàntica o classificació de materials.

---

## Estructura del projecte

hyper-gaussian-grouping-tfg/
- arguments/ Definició de paràmetres d'entrenament per optimització i del model.
- config/ Conté paràmetres d'entrenament.
- data/ Carpeta on s'ubiquen els datasets, conté codi per preparar-los.
- docs/ Documentació d'instal·lació.
- gaussian_renderer/ Codi relacionat amb la representació 3D de les gaussianes i renderitzat diferencial.
- media/ Imatges i resultats de mostra.
- scene/ Fitxers i utilitats per fer servir escenes 3D, configuració de càmeres, dades d'entrenament, etc.
- submodules/ Conté codi CUDA per a la rasterització de les gaussianes.
- utils/ Funcions comunes reutilitzades en diverses parts del codi.
- *.py Scripts python

### Scripts de nivell arrel

- avaluacio.py: Codi per avaluar els models entrenats. Serveix per mostrar imatges comparatives de groundtruth, train i test; mesurar mètriques (PSNR, SSIM i LPIPS) i mostrar gràfiques.
- convert.py: S'utilitza per executar el COLMAP amb els arguments necessaris per deixar els datasets llestos per l'entrenament.
- get_data.py: Script auxiliar molt senzill per mostrar els punts inicials i gaussianes que conté un point_cloud.ply.
- hyper.py: Conté les definicions dels models utilitzats (MLP, Conv) per predir el color a partir dels embeddings renderitzats i una funció per calcular una part de la Loss (L1).
- hyper.sh: Script bash que s'encarrega d'executar els scripts importants del projecte per generar el núvol de punts inicial i les poses de les càmeres, entrenar i renderitzar.
- hyper_render.py: Conté el codi per carregar una iteració del model i poder renderitzar les imatges de test.
- hyper_train.py: Codi per l'entrenament del model.
- metrics.py: Mètriques per avaluar els resultats.
- reshape.py: Script auxiliar per redimensionar imatges.

---

## Datasets

Per preparar els datasets, aquests s'han de situar en la carpeta data/ i han de contenir el format:
```
data/
  nom_del_dataset/
    channels_distorted/
      0/
      1/
      ...
      n_canals-1/
    input/
```

La carpeta channels_distorted/ ha de contenir les imatges separades per canals i la carpeta input/ ha de contenir imatges per poder executar el COLMAP, aquestes es poden formar amb un rgb artificial juntant 3 canals o es pot agafar un dels canals.

La carpeta data/ ja conté carpetes dels datasets utilitzats en aquest treball amb els scripts per poder preparar-los.

### Per preparar els datasets utilitzats

- **MultimodalStudio**: Consta de 32 escenes amb 50 imatges i 9 canals cadascuna. Aquest es pot descarregar en la seva pàgina en el següent enllaç: https://lttm.github.io/MultimodalStudio/pages/dataset.html. Hi ha dues opcions, descarregar únicament l'escena Birdhouse (6 GB) o descarregar el dataset complet (128 GB). En qualsevol dels 2 casos, la carpeta descarregada s'ha de situar en la carpeta data/multi-modal-studio/. Allà hi ha un script mms.py per preparar el datatset. Només s'han de modificar les variables "path_input" i "path_output" en funció de l'escena que s'utilitza. De manera predeterminada està posada l'escena Birdhouse. Llavors s'excuta:
``` python mms.py ```

- **Basement**: Consta d'una escena de 50 imatges i 9 canals cadascuna. Aquest dataset ha estat proveït per Arnau Marcos Almansa. Per preparar-lo s'ha de posar dins de la carpeta data/basement i llavors executar ``` python basememt.py ```. Si les imatges són molt grans, es pot aprofitar per fer el COLMAP, però després per l'entrenament potser que no hi càpiguen en la GPU. Llavors es pot executar ``` python reshape.py ``` Al final d'aquest script es pot canviar la variable "route" per incloure el path de les imatges que es volen redimensionar. En aquest cas: data/basement/images.

- **X-NeRF**: Consta d'una escena de 30 imatges i 10 canals cadascuna. Aquest es pot descarregar en el següent enllaç: https://amsacta.unibo.it/id/eprint/7142/. Un cop descarregat, s'ha de crear la carpeta penguin/ i ubicar-la a data/xnerf. Dins de penguin/ s'ha de co·locar l'arxiu descarregat "ms_imgs.npy" i llavors executar ``` python xnerf.py ```.

- **Spec-NeRF**: Consta d'una escena de 9 imatges i 20 canals cadascuna. Es pot descarregar des del seu github: https://github.com/CPREgroup/SpecNeRF-v2?tab=readme-ov-file. Es decarrega una carpeta anomenada RAW/ que s'ha de situar dins de data/Spec-NeRF. Llavors s'executa el script ``` python spec-nerf.py ```.
 
---

## Entrenament



---

## Avaluació



---

## Experiments i Resultats


