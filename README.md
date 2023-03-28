# LandmarkClassification

Using PyTorch to train a custom CNN from scratch and to train a pre-trained ResNet18 model to identify famous landmarks in images.

## Project Motivation

The objective of this project is to explore the process of building a custom CNN model for image classification while also utilizing transfer learning techniques and pre-trained CNN models to improve performance. One possible application of this project is social media tagging, where the model could automatically identify famous landmarks in user-uploaded photos.

## Repository Structure and File Description

```markdown
├── src
│   ├── data.py                       # functions used to process and visualize the data
│   ├── helpers.py                    # helper functions
│   ├── models.py                     # code containing our custom CNN model
│   ├── optimization.py               # setting up optimizer
│   ├── predictor.py                  # code used to predict class on image
│   ├── train.py                      # training script
│   └── transfer.py                   # code for transfer learning
├── cnn_from_scratch.ipynb            # notebook used to run our custom CNN model
├── transfer_learning.ipynb           # notebook used to run pre-trained CNN model
├── requirements.txt                  # requirements
├── README.md                         # Readme file            
```

## Installations

To run this project, install the libraries listed in the `requirements.txt` file.

## How to Interact with the project

1. Clone the repository to your local machine using the following command:

```
git clone https://github.com/zhikaichen99/LandmarkClassification.git
```

2. Run the cells in the `cnn_from_scratch.ipynb` notebook.

3. Run the cells in the `transfer_learning.ipynb` notebook.
