# Gaussian Process Playground

We will move to TU Berlin Gitlab or create a new repository after first group meeting
with supervisor. This repository is just a playground and it is not an official repository
to present our work.

## Google Colab Instruction Guide
https://colab.research.google.com/

### Step 1: Allow access to your Google Drive
```
# This will prompt for authorization.
from google.colab import drive
drive.mount('/content/drive')
```
Go to the specify directiory 
```
%cd /content/drive/MyDrive/Colab\ Notebooks/
```

### Step 2: Git Clone this repository
```
!git clone https://github.com/twbabyduck/TUB_GP/
```
Go into TUB_GP directory
```
%cd TUB_GP/
```

### Step 3. Install important packages
```
python3 -m pip install -r requirements.txt   
```

## GPFlow Sampling Module (Reference)
```
@inproceedings{wilson2020efficiently,
    title={Efficiently sampling functions from Gaussian process posteriors},
    author={James T. Wilson
            and Viacheslav Borovitskiy
            and Alexander Terenin
            and Peter Mostowsky
            and Marc Peter Deisenroth},
    booktitle={International Conference on Machine Learning},
    year={2020},
    url={https://arxiv.org/abs/2002.09309}
}
```