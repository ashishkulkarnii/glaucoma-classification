# retinal-fundus-glaucoma-classification

Using CNNs to classify an image into normal or glaucomatous, using retinal fundus images by transfer learning.
The dataset used is called [ACRIMA](https://figshare.com/s/c2d31f850af14c5b5232), containing 705 labelled images: _396_ glaucomatous images and _309_ normal images.

The CNN models were fit using 70% of the dataset for training, 10% for validation and 20% for testing.

The model architecture used is as follows:

- An input layer (256, 256, 3)
- A data augmentation layer involving random change in contrast, flip along horizontal or vertical, rotation, and translation of the images
- The base model with _image-net_ weights
- A flatten layer
- A dense layer with _ReLU_ activation
- A 0.5 dropout layer
- A dense output layer with _softmax_ activation


In the case of ResNet-50 based model, a comparison was made between efficacy of different histogram equalization techniques.

- Histogram Equalization
- Contrast-Limited Adaptive Histogram Equalization
