# Dataset-transfer

This is the 1st version for the implementation of dataset-transfer, which only includes loss function for task classification. In 2nd implementation, subject discriminator will be added. 

Be free to play with it.

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

### Datasets
- **Competition IV Data sets 2a**. [Link](http://www.bbci.de/competition/iv/#dataset2a)
- **EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy**. [Link](http://gigadb.org/dataset/100542)

#### Preprocess the data
For the correspondence of datasets, only 21 shared channels are reserved. (['Fz', 'FC3', 'FC1', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'])

- The original file format of **Competition IV Data sets 2a** is *.gdf. As an alternative, we provide preprocessed .mat file. Click [here](https://drive.google.com/file/d/1CIYTVuTqGNJuAWJ4e3HC7AXegTw_WuiJ/view?usp=sharing).

- We also provide the preprocessed data for KU data. Click [here](https://drive.google.com/file/d/1VPyuBzL9Y43vqAfe2yoXrf9EUPWOBNLe/view?usp=sharing).

After downloading the datasets, unzip them into `./data/BCICompetition-IV2a/` and `./data/sub54/`, respectively. You have to create the folder for each dataset yourself.
