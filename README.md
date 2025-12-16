This repository includes the data preprocessing logic, data extraction scripts, model architectures, final training curves, and model checkpoints. Model training and evaluation are available on this Colab notebook:
https://colab.research.google.com/drive/12lFGjYipbE0MPZtidRNwfnscnyfy8evX?usp=sharing

# Environment
1. Ensure you have Python 3.12+ installed

2. Run `uv sync` to sync your local environment with the dependencies listed in pyproject.toml

3. Download the MJSynth and IIIT-5k datasets and place them in the root of this repository.

MJSynth: https://www.robots.ox.ac.uk/~vgg/data/text/
IIIT-5k: https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset

# Code
`main.py` contains the preprocessing pipeline to prepare images for training. The segmentation steps are also included. See `preprocessing.py` and `segmentation.py` for the preprocessing and segmentation code.

`utils.py` contain

`output/` contains the training curves and final checkpoints after training. To load these checkpoints, see the Colab notebook.

`src/extract_chars.py` and `src/extract_words.py` extract preprocessed characters and words respectively for both types of models and store them in directories `char_train/`, `char_val/`, `word_train/`, and `word_val/`. To train the models on Colab, ZIP these directories, upload the ZIP files to Colab, and then unzip them within the notebook.

`src/model/view_model` is a simple script that displays basic metadata about the specified model, such as its validation accuracy and the number of epochs it trained over. The `model` object is a dictionary with two keys: `metadata` and `model_state_dict`, the latter of which contains the model's state upon finishing training, which can be used for reloading the checkpoint later for inference.

`src/char` and `src/sequential` relate to the character classifiers (LeNet and VGG-16) and the sequential classifier (CRNN) respectively.

`src/model/char/lenet.py` and `src/model/char/vgg16.py` define the LeNet and VGG-16 model architecture.
`src/model/char/train_char.py` defines the training script for both character classifiers.
`src/model/sequential/crnn.py` defines the CRNN model architecture.
`src/model/sequential/decoder.py` defines the decoding algorithm to convert the CRNN's raw output into a coherent string.
`src/model/sequential/train_seq.py` defines the training script for the CRNN.