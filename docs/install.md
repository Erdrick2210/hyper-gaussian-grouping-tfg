### **Standard Installation**


Clone the repository locally
```
git clone https://github.com/lkeab/gaussian-grouping.git
cd gaussian-grouping
```

Our default, provided install method is based on Conda package and environment management:
```bash
conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
