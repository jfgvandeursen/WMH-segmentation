# Thesis
## Master Information Studies: Data Science Track
### Fully automated White Matter Hyperintensities segmentation in medical images using U-net
**Author**: Jente van Deursen, University of Amsterdam, submitted in partial fulfillment for the degree of master of science.

White matter hyperintensities (WMH) are abnormalities that ap- pear within the white matter region on human brain magnetic resonance images (MRI). These abnormalities are one of the main indicators of small vessel disease, and research shows that they play an important role in stroke, dementia, and aging. Characterization of WMH is important for clinical research, but it is a laborious and time-consuming task. It has the potential to be automated using a computer vision approach. In this work, this potential is explored through semantic segmentation of T1-weighted and fluid attenuation inversion recovery (FLAIR) images using U-Net-based Convolutional Neural Networks (CNNs). First, the performance po- tential of the U-net model was explored using a pre-trained U-net model. Second, this model is improved by building it from scratch and training it from the beginning to the end. Finally, the impact of an ensemble approach on the performance of the model is ex- plored. The model is trained and tested on the publicly available data set from the 2017 MICCAI WMH challenge. The results on the evaluation metrics of the ensemble approach are compared to the rebuilt U-net model. The rebuilt model gets an average dice score and precision (recall) of 77% en 81% respectively. The ensemble approach did not improve the result in these experimental settings.

- Code: This folder contains the notebook with the code for the U-net model.
- Thesis: Link to the thesis
