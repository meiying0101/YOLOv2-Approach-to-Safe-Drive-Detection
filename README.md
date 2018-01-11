# YOLOv2-Approach-to-Safe-Drive-Detection

##Abstract

Inspired by the MS-FRCNN [2], which focus on the subject of safe driving, our main objection is to detect hands in a sequence of the driving scene in the front seat. By successfully detecting the left and right hand of the driver and the passenger, we are able to find an approach to determine whether the driver’s hands are on the steering wheel. For our scenario is real-time detection while driving, we utilize the real-time object detection system of YOLOv2 [1], which features in fast detection, as our original model. We experimented on the VIVA dataset [3], and tried some approaches to modify our model for better performance. In hope of achieving higher accuracy, we are tempted to make a tradeoff with the high speed of our original model. Extracted from the idea of MS-FRCNN [2], an ROI pooling layer is added to our model, with L2 normalization afterward before concatenation of the output of the convolutional layers. We also did some analysis on our training process to explain the outcome.

##Prepare the training dataset

VIVA Hand Detection:
http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/

##Details

voc_conversion_scripts contains  scripts for converting the dataset to either HDF5 format for easier training with Keras or Tensorflow.
