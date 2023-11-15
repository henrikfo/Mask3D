## Implementing <a href="https://github.com/JonasSchult/Mask3D">Mask3D</a> , for 3D instance segmentation of <a href="https://github.com/meidachen/STPLS3D">STPLS3D</a>


## The code structure are kept just like the one in the original Mask3D repository. 

```
├── mix3d
│   ├── main_instance_segmentation.py <- the main file
│   ├── conf                          <- hydra configuration files
│   ├── datasets
│   │   ├── preprocessing             <- folder with preprocessing scripts
│   │   ├── semseg.py                 <- indoor dataset
│   │   └── utils.py        
│   ├── models                        <- Mask3D modules
│   ├── trainer
│   │   ├── __init__.py
│   │   └── trainer.py                <- train loop
│   └── utils
├── data
│   ├── processed                     <- folder for preprocessed datasets
│   └── raw                           <- folder for raw datasets
├── scripts                           <- train scripts
├── docs
├── README.md
└── saved                             <- folder that stores models and logs
```

But we add the folder "results" to store the validation scores

### Dependencies :memo: are still the same
```yaml
python: 3.10.9
cuda: 11.3
```

#### Please, do install these in a save enviorment (i.e. a docker container), to not ruin your other projects. And these install instructions is what worked for me, they might not for you.

Install conda
```
Download from https://docs.conda.io/projects/miniconda/en/latest/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Install python 3.10.9
```
wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz
tar xzf Python-3.10.9.tgz
Cd Python-3.10.9 
./configure 
make / make altinstall
```

Install the cuda toolkit with cuda 11.3
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sh cuda_11.3.0_465.19.01_linux.run
```

If you get an error with gcc version compatability, installing a lower version of gcc and g++ might do the trick:
```
sudo apt -y install gcc-9 g++-9 
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
```
If you get an error with nvidia driver version compatability, try this:
```
sudo apt --purge remove "*nvidia*" "libxnvctrl*"
* reboot your system *
sudo apt install nvidia-driver-470
```

## Installing mask3d
```
apt-get install libopenblas-dev
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
```
Remove/comment pip packages in enviorment.yaml  
Create a req.txt with all pip packages that would have been installed from eviorment.yaml  
```
conda env create -f environment.yml
conda activate mask3d_cuda113 

pip3 install "cython<3.0.0" && pip install --no-build-isolation pyyaml==5.4.1
pip3 install -r req.txt

pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html 
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps
mkdir third_party 
cd third_party

git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" 
cd MinkowskiEngine 
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas
```

The installation of the MinkowskiEngine might not work and throw a error.
If that happens try
```
export MAX_JOBS=1
```

```
cd .. 
git clone https://github.com/ScanNet/ScanNet.git 
cd ScanNet/Segmentator 
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2 
Make

cd ../../pointnet2 
python setup.py install 
cd ../../ 
pip3 install pytorch-lightning==1.7.2
```

Install for vizualization
```
pip install pyntcloud
```

If there is an error about torchmetrics
```
pip3 install torchmetrics==0.11.4
```

## Running the model

First, download the Synthetic Instance data from <a href="https://docs.google.com/forms/d/e/1FAIpQLSf0jsHw4Q6FFB6AjEgTkF2tgHdMMFyLjC-7fDHrmV01Kci0aA/viewform">STPLS3D</a>  
Make sure that the data is put in the appropriate folders after creating them (data/raw/stpls3d/    train-validation) file 1-24 in train, 24-25 in validation  

Then run the following to prepare the data  
```
python -m datasets.preprocessing.stpls3d_preprocessing preprocess \
 --data_dir="data/raw/stpls3d" \
 --save_dir="data/processed/stpls3d"
```

When the data has been prepared, run the script  
```
python scripts/stpls3d/train_stpls3d.sh
```
to train a model.  
Run the test script
```
python scripts/stpls3d/test_stpls3d.sh
```
for validation results and vizualisations to be saved.  

Edit the scripts for appropriate needs.  
Increase the batch size and lower the voxel size when large amout of vRAM is available.  
Minimum requirements to run this unedited scripts
```
~ 20 GB RAM
~ 12 GB vRAM
```

## Visualization of results.
The code has been modified to be able to save visualization in two ways instead of one.  
You can use pyviz3d as the repo does, or a general 3D visualization program like CloudCompare.  

For pyviz3d, set verbose in visualizer.save() to True for instructions on how to use it. (Found on line 105 in /utils/pc_visualizations.py)  

With CloudCompare, use the .ply files saved in /saved/"*project*"/visualizations/"*crop*"/  
In CloudCompareuse you might need to change the cloud point size from Default to be able to see the points (i.e. 10)  

This below is some of the visual results achieved in this implementation
![Visual results](https://github.com/henrikfo/Mask3D/blob/main/docs/results.png)  

Using the metric of mean average precision we achieved similar results to the authors
|  | AP | AP_50 | AP_25 | 
|:-:|:-:|:-:|:-:|
| Mask3D | 57.3 | 74.3 | 81.6 | 
| Mine | 55.0 | 73.2 | 80.4



Checkout the <a href="https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/stpls3d/">visualizations</a> provided by Mask3D.

Papers for this work  
<a href="https://arxiv.org/abs/2210.03105">Mask3D</a>  
<a href="https://arxiv.org/abs/2203.09065">STPLS3D</a>  

And other papers and repos of interest  
<a href="https://arxiv.org/abs/2309.16375">Tree detection with point clouds, a review (paper) </a>  
<a href="https://github.com/murtiad/Tree_segmentation-using_PointNet/tree/main">Urban segmentation using PointNet++ (repo)</a>  
