# Dataset-transfer

This is the 1st version for the implementation of dataset-transfer, which only includes loss function for task classification. In 2nd implementation, GAN-based augmentation method will be added. 

Be free to play with it. It's better to use under Linux.

## Getting Started
### Installation
- Create a new environment with Python 3.7:
```bash
conda create -n dataset_transfer python=3.7
conda activate dataset_transfer
```
- Install [PyTorch](http://pytorch.org) with CUDA 10.2:
```bash
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
```
- You also need to install [wandb](https://wandb.ai/quickstart/pytorch) for visualization. Please follow the website instruction to configure.

- For other dependencies, you may need: 
```bash
pip install scipy
pip install matplotlib
```

### Datasets
- *Competition IV Data sets 2a*. [Link](http://www.bbci.de/competition/iv/#dataset2a)
- *EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy*. [Link](http://gigadb.org/dataset/100542)

#### Preprocess the data
For the correspondence of datasets, only 21 shared channels are reserved. (['Fz', 'FC3', 'FC1', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'])

- The original file format of *Competition IV Data sets 2a* is *.gdf. As an alternative, we provide preprocessed .mat files. Click [here](https://drive.google.com/file/d/1CIYTVuTqGNJuAWJ4e3HC7AXegTw_WuiJ/view?usp=sharing) to download.

- We also provide the preprocessed .npy data for Gigascience dataset. Click [here](https://drive.google.com/file/d/1VPyuBzL9Y43vqAfe2yoXrf9EUPWOBNLe/view?usp=sharing) to download.

After downloading the datasets, unzip them into `./data/BCICompetition-IV2a/` and `./data/sub54/`, respectively. You may have to create the folder for each dataset yourself.

### Training

- Help function:
```
usage: main.py [-h] [-dataset DATASET] [-load LOAD] [-train] [-full_train]
               [-save] [-batch_size BATCH_SIZE] [-epoch EPOCH] [-lr LR]
               [-n_workers N_WORKERS] [-set_weight_decay SET_WEIGHT_DECAY]
               [-lambd LAMBD] [-step] [-test] [-visualizer]

Baseline method for 2-class MI task classification

optional arguments:
  -h, --help            Show this help message and exit
  -dataset DATASET      Specify ataset for train/test
  -load LOAD            Load model
  -train                A complete train
  -full_train           No validation and test
  -save                 Save model
  -batch_size BATCH_SIZE
                        Batch size
  -epoch EPOCH          Total epochs
  -lr LR                Learning rate
  -n_workers N_WORKERS  Num_workers
  -set_weight_decay SET_WEIGHT_DECAY
                        Weight_decay
  -lambd LAMBD          Weight for balancing loss function
  -step                 Cancel scheduler
  -test                 Only test
  -visualizer           Visualization
```
- To simply start a new train in the terminal, at least you have to specific a dataset([*iv_2a*, *sub54*]) to train:
```
python main.py -dataset iv_2a -train 
```
- As for real-time training, go to [wandb.ai](https://wandb.ai).


### Visualization

- Visualization is only for Linux temporarily, I will improve it ASAP to make it possible on Window as well.

- Visualization by class:

<img src="https://github.com/RoboHaptic/dataset-tranfer-v1/blob/main/visualization/iv_2a_class.png" width="800"/>

- Visualization by subject:

<img src="https://github.com/RoboHaptic/dataset-tranfer-v1/blob/main/visualization/iv_2a_sub.png" width="800"/>




