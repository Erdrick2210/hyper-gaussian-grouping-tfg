### **Instal·lació Estàndar**


Clonar el repositori localment
```
git clone https://github.com/Erdrick2210/hyper-gaussian-grouping-tfg.git
cd hyper-gaussian-grouping-tfg
```

El mètode d'instal·lació per defecte basat en el de Gaussian Grouping, es basa en la gestió de paquets i entorns Conda:
```bash
conda create -n hyper_gaussian_grouping python=3.8 -y
conda activate hyper_gaussian_grouping 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
Altres dependències s'inclouen en el fitxer enton.txt
