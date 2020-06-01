# Polyp Localization for Colorectal Cancer Diagnosis

Final project for [6.S897/HST.956: Machine Learning for Healthcare](https://mlhc19mit.github.io/) in Spring 2019 at MIT.

## Overview
In our work, we attempt to provide tools to aid gastroenterologists in the space of colorectal cancer. We employ machine learning, specifically neural networks, which have found significant success in applications around image and video analysis, natural language processing, and other fields. Colorectal cancer is typically screened using colonoscopies, in which gastroenterologists look for polyps through a real-time video feed in the colon. As accurate detection of polyps is essential to preventing onset of colorectal cancer, we use neural networks to provide a tool to doctors to assist with polyp classification and detection. Using **VGG-19** and **Resnet-50**, we achieve a 0.999 AUC on image-level polyp classification on the Kvasir dataset and high PPV and NPV values. We also try **patch-based neural network models** for classification and localization as well as an assortment of neural networks specifically for localization (e.g. **YOLO**), though we are unable to achieve as significant results for these models.

We used Python, Keras, and Tensorflow in our implementation.

For details about our project, refer to our [paper](https://github.com/atwang16/sp19-6s897-colon/blob/master/HST_956_Final_Project_Report.pdf).

## Datasets
- Kvasir dataset (used for classification task): https://datasets.simula.no/kvasir/
- CVC-ClinicDB dataset (used for classification and localization tasks): https://polyp.grand-challenge.org/CVCClinicDB/
- ETIS-Larib dataset (used for localization task): https://polyp.grand-challenge.org/EtisLarib/

## Data Preprocessing / Methods
See Section 2.1 and Section 3, respectively, in our paper.

## Results
With our classification models, we achieved results that outperformed the state-of-the-art model in this [2018 paper](https://www.sciencedirect.com/science/article/abs/pii/S0016508518346596) by ~3%. With our segmentation models, we achieved results that performed above baseline but not by a significant amount and not competitively with state of the are polyp segmentation. See Section 4 of our paper for more details on the performance of our models.

## Model Interpretability
Model interpretability is an extrememly important topic in the field of ML for Healthcare -- how can we trust our models to give the correct decision in a life-or-death situation when we have no idea how it makes decisions in the first place? We used [Grad-CAM](https://arxiv.org/abs/1610.02391) to produce visual explanations for the decisions of our classification models, which showed that the models' predictions were indeed most influenced by the pixels that made up the polyps rather than some confounding factor (e.g. a surgery tool). See Section 4.2 of the paper for details about the math behind Grad-CAM, as well as our results in Appendix A Figure 4.

## Code

#### Classification
All of the code and data for the classificaton task is located in the `cnn_classification` folder. In `cnn_classification`, you can use `split_data.py` to create a train/test/validation data split, or you can use the split provided in the `data` folder. To train a model, run `train.py` with the desired arguments (specifying paths for training and validation data, model type, and other configurations). Model architectures implemented/used in this project (VGG-16, VGG-19. ResNet50) can be found in the `models` folder. 

To evaluate a model, run `evaluate.py` with the desired arguments (test data path, model to evaluate). Finally, you can use `overlay_grad_cam.py` or `grad_cam_guided_no_overlay.py` to view the Grad-CAM results for a specified image and model. The shell files (`rebuild.sh` and `docker_run.sh`) and docker folder were used in training models on an AWS GPU instance.

#### Localization
Localization code can be found in the main directory, including the patch-based network method and the classic deep CNN method (using VGG, ResNet, and YOLO).

For the patch-based network method, relevant files include `patch_dataset.py` and `patch_majority_voting.py`.

For the classic deep CNN method, relevant files include `train_images_localization.py` and `train_images_regression.py` for training models, files in the `models` folder in the main directory for the model architectures used, and `evaluate.py` for evaluation. Data can be found in the `data/segmentation` folder.


## Contributions of Team Members
- Elaine Xiao: Implemented pipeline for data preprocessing, training, and evaluating the deep CNN classification models; trained, fine-tuned, and evaluated models with different architectures; implemented and tested Grad-CAM methods.
- Miguel Del Rio: Implemented pipeline for training and evaluating the patch-based models and the patch-based dataset class; implemented, trained, and evaluated patch-based network models.
- Austin Wang: Implemented pipeline for training and evaluating the deep CNN localization models; trained and evaluated various models including YOLO; implemented training on videos.

