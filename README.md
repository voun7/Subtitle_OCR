# Subtitle OCR

![python version](https://img.shields.io/badge/Python-3.11-blue)
![support os](https://img.shields.io/badge/OS-Windows-green.svg)

Program that uses deep learning to detect and recognize texts.
The training data is optimized for subtitle text images.

## Setup Instructions:

### Install Packages:

```commandline
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```commandline
pip install -r requirements.txt
```

### Build Package:

```commandline
pip install build
```

```commandline
python -m build
```