

### Recreation, U-net: Convolutional networks for biomedical image segmentation

by Jacob Stachowicz, Max Joel Söderberg and Anton Ivarsson

<p align="center"><img align = "center" src="images/unet.png" width="50%"></p>
Figure 1: Example U-net structure


##### **Abstract:**

The subject of the project was biomedical image segmentation. Namely to repro
duce the results of the paper "U-net: Convolutional networks for biomedical image
segmentation" by Olaf Ronnenberger, Philipp Fischer and Thomas Brox. After
the reproduction the goal was to implement another model for segmentation and
compare how the strategies used by Ronneberger et al. (2015) performed on the
other model. Due to time constraints the other model was not implemented. An
experiment was conducted in order to validate the data augmentation strategy used
by Ronneberger et al. (2015). A conclution could be made that the strategy was
sound. Without the strategy, our implemented U-net reached an IoU score of 0.332
and a F1 score of 0.605. With the strategy, our implemented U-net reached an
IoU score of 0.770 and a F1 score of 0.891. An algorithm was implemented to
perform another experiment for the custom loss function used by Ronneberger
et al. (2015). The experiment was not completed due to the chosen deep learning
framework being opinionated. However, studying the computed weights indicate
that the weights could help the model learn better borders between cells.


#### 1. Introduction

In this project we will work with biomedical image segmentation for cell images. Segmentation of
biomedical images is useful for a different number of purposes. For instance, Punitha et al. (2018)
used a Feed Forward Neural Network to segment images of benign and malignant breast cancer
(Punitha et al. 2018). This project aims to reproduce the results of the paper "U-net: Convolutional
networks for biomedical image segmentation" by Olaf Ronneberger, Philipp Fischer and Thomas
Brox. The authors of this paper mentions how there is a large consent that successful training of
deep learning networks requires many thousands of annotated training samples. However, in this
paper the authors present a strategy that relies heavily on data augmentation instead of a large sample
size, using an network architecture called "U-net". The purpose of the data augmentation approach
is to utilize the available annotated training samples more efficiently (Ronneberger et al. 2015). To
reproduce the results we will first implement the network. After the implementation of the network
we will validate that the strategies of Ronneberger et al. (2015), besides from the architecture gave
the improvements claimed. In addition to reproducing the results, this project will be conducted
with the intention to compare the segmentation network with another method of segmenting images.
That is, try training another network architecture for image segmentation with the same surrounding
strategies as Ronneberger et al. (2015).
Our initial assessment of the workload required to reproduce the results achieved by Ronneberger
et al. (2015) where incorrect. This flawed assessment led to us not having enough time left to compare
the results to a network with another architecture. Due to the chosen deep learning framework being
opinionated, we had trouble adding the custom loss function which Ronneberger et al. (2015) used
for instance segmentation. The custom loss function used precomputed weights for each training
sample. Even though we could not experiment with this custom loss function, we did implement an
algorithm for calculating the weights. Studying the computed weights indicate that the weights could
help the model learn better borders between cells that are stuck together. The lack of precomputed
border weights in this project forced us to focus on semantic segmentation, while Ronneberger et al.
(2015) focused on instance segmentation.
We could conclude that the strategy used for data augmentation by Ronneberger et al. (2015) was
sound. Training our U-net without the data augmentation the network achieved a IoU score of 0.332
and a F1 score of 0.605. With the data augmentation strategy we achieved a IoU score of 0.770 and a
F1 score of 0.891. These scores show that segmentation of cell images can be achieved accurately
without large amounts of data using this data augmentation strategy. All scores were calculated on
test data where the model with the lowest validation loss were used

#### 2 Related work

As mentioned in the introduction, this project was partly an attempt to reproduce the results by
Ronneberger et al. (2015). Because of this, the main source of reference was the aforementioned
paper. Ronneberger et al. (2015). implemented a convolutional neural network with the architecture
as seen in figure 1.
The authors implemented data augmentation that was mostly based on elastic deformations, as
well as Gaussian noise. The authors achieved well defined borders. These well defined borders
can seemingly be accredited towards the custom loss function implemented. This custom loss
function was a weighted pixel wise binary cross entropy that took advantage of precomputed weights
for each pixel in the data set. The initial weights of the network was drawn from a Gaussian
distribution with standard deviation p
2/N where N denotes the number of incoming nodes of one
neuron(Ronneberger et al. 2015). The authors network was implemented using Caffe with MATLAB,
as well as some parts in C++.


There are a lot of areas to improve in. Below follows some suggestions:

- **Preprocessing**
  Improving computational time and potentially reducing noice
  Possible classification gain by down sampling the frequencies in the signals, e.g. from 200 hz to 100 hz or 50 hz, for reducing noise.
- **Feature extraction**
  Other features could be extracted, that could potentially be better than the existing features. 
- **Tuning the models.**
  We have only tested the models without tuning them independently for this specific task. Tuning the two classifiers would show which classifier is best, the untuned results does not. 
- **Feature selection methods.**
  We used a questionable method for finding the best features, a feature selection method that is optimized for linear regression. A suggestion for future work is to test different feature selection methods, e.g. *selectKbest* from the python library sklearn or other that are specifically 
- **K-Cross validation**
  Implementating K-cross validation for a better calibration of the models.
- **Better balancing**
- **Evaluation methods**
  We used AUC ROC in this project to measure performance. All the projects who participated in the Physionet challenge where evaluated using *The Area Under Precision-Recall Curve* (AUPRC). To compare the results an AUPRC evaluation method would would enable a possibility for comparison.
- **More memory**
  The limitations in not meeting the memory requirements made it impossible to perform training and testing on the whole data set. Our calculations show that we needed at least 41 GB of RAM. A alternative to acquiring more RAM is to down sample the data, this in turn could affect the classification accuracy

### Installing needed packages for the usage of extraction

* Install anaconda if not installed already
* conda install -c conda-forge hdf5storage
* If you can't find conda add the path variable by running: export PATH=~/anaconda3/bin:$PATH