## Introduction
CREX is a tool to match semantically similar functions based on transfer learning. 

## Installation
We recommend `conda` to setup the environment and install the required packages.

First, create the conda environment,

`conda create -n crex python=3.8 numpy scipy scikit-learn requests`

and activate the conda environment:

`conda activate crex`

Then, install the latest PyTorch (assume you have GPU):

`conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

Enter the crex root directory: e.g., `path/to/crex`, and install crex:

`pip install --editable .`

For large datasets install PyArrow: 

`pip install pyarrow`


## Preparation

### Pretrained models:

Create the `checkpoints` and `checkpoints/similarity` subdirectory in `path/to/crex`

`mkdir checkpoints`, `mkdir checkpoints/similarity`

Download [pretrained weight parameters](https://drive.google.com/file/d/1xNcW8r01_J2OTZFh1B0eOG5ikj73zhwe/view?usp=sharing) and put in `checkpoints/pretrain`

### Sample data for Micro-traces and embeddings

For full Micro-traces:

`./tools/preprocess.sh`

For function Micro-traces:

`./tools/extract02.sh`

For embeddings:

`./command/inference/get_embedding.sh`


## Dataset

We put our dataset here

`dataset-funtion.zip`

.
├── bugcode          : buggy file
├── Coconut          : APR patch file
├── GT                   : Ground-truth(correct) file


Files in `dataset-funtion/xxx(Bug ID)/xxx(File type)/modify` are the functional Micro-traces


