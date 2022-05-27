# A Continuous Time Framework for Discrete Denoising Models
## Notebooks
Pre-trained models are available at https://www.dropbox.com/scl/fo/zmwsav82kgqtc0tzgpj3l/h?dl=0&rlkey=k6d2bp73k4ifavcg9ldjhgu0s

To generate CIFAR10 samples, open the `notebooks/image.ipynb` notebook.
Change the paths at the top of  the `config/eval/cifar10.py` config file to point to a folder where CIFAR10 can be downloaded and the paths to the model and config downloaded from the dropbox link. 

To generate piano samples, open the `notebooks/piano.ipynb` notebook.
Change the paths at the top of the `config/eval/piano.py` config file to point to the dataset downloaded from the dropbox link as well as the model weights and config file.

The sampling settings can be set in the config files, switching between standard tau-leaping and with predictor-corrector steps.

## Training
### CIFAR10
The CIFAR10 model can be trained using
```
python train.py cifar10
```
Paths to store the output and to download the CIFAR10 dataset should be set in the training config, `config/train/cifar10.py`.
To train the model over multiple GPUs, use
```
python dist_train.py cifar10
```
with settings found in the `config/train/cifar10_distributed.py` config file.

### Piano
The piano model can be trained using
```
python train.py piano
```
Paths to store the output and to the dataset downloaded from the dropbox link should be set in `config/train/piano.py`.


## Dependencies
```
pytorch
ml_collections
tensorboard
pyyaml
tqdm
scipy
torchtyping
matplotlib
```



