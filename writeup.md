# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[training-loss]: ./assets/training_validation_loss.png "Training & Validation Loss"
[track-1-sample]: ./assets/track_1_sample.png "Track One Sample"
[track-1-sample2]: ./assets/track_1_sample2.png "Track One Sample"
[track-2-sample]: ./assets/track_2_sample.png "Track Two Sample"
[track-2-sample2]: ./assets/track_2_sample2.png "Track Two Sample"
[track-1-sample-cropped]: ./assets/track_1_sample_cropped.png "Track One Sample Cropped"
[track-2-sample-cropped]: ./assets/track_2_sample_cropped.png "Track Two Sample Cropped"
[brightness]: ./assets/brightness_example.png "Brightness Example"
[flipped]: ./assets/flipped_example.png "Flipped Example"
[scale]: ./assets/scale_example.png "Scale Example"
[shadow]: ./assets/shadow_example.png "Shadow Example"

[original-input]: ./assets/original_visualization_input.png "Orignal Input"
[conv-1-fm]: ./assets/conv_1_fm.png "Conv 1 Feature Map"
[conv-2-fm]: ./assets/conv_2_fm.png "Conv 2 Feature Map"
[conv-3-fm]: ./assets/conv_3_fm.png "Conv 3 Feature Map"
[conv-4-fm]: ./assets/conv_4_fm.png "Conv 4 Feature Map"
[conv-5-fm]: ./assets/conv_5_fm.png "Conv 5 Feature Map"

[avg-fm]: ./assets/averaged_fm.png "Averaged Feature Map"
[salient-objs]: ./assets/salient_object_masks.png "Salient Object Masks"



## Files & Code Usability

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Additional Model operations are as follows:

```bash
# Display sample content from driving_log.csv
python model.py -op explore_data
```

```bash
# Print out summary of model architecture
python model.py -op model_summary
```

```bash
# Display a single, random example of a training image.
# Possible augmentations included.
python model.py -op show_training_sample
```

```bash
# Train the model for `n` epochs
python model.py -op train -epochs 10
```

```bash
# Display a graph of the training/validation loss history
python model.py -op show_training_history
```

```bash
# See what the Convolutional Neural Network sees
python model.py -op show_activation
```
## Data Collection

I decided on a methodical approach to data collection.  I recorded three laps 
of each track in each direction, as it would seem to help the model generalize 
more effectively.  Each recorded frame captures images from cameras positioned
in the center, left and right side of the windshield. 

Care was taken to drive as steady as possible and, in the case of the second,
mountainous track, to stay in the correct lane (right lane, in this case).
Additionally, several recordings of recovering to the center of the lane were
taken.

After training, the model was evaluated in the simulator and three more samples 
were taken for each section of the track that proved to be problematic (ie. drove
off the road or jumped lanes).

Some examples of the data are shown below:

**Track One:**

![track-1-sample]

![track-1-sample2]

**Track Two:**

![track-2-sample]

![track-2-sample2]


## Model Architecture

The weights of the network are trained to minimize the **mean squared error**
between the predicted steering angle and the actual angle recorded during training.
The model is using the **Adam** optimizer with a learning rate of 0.001. 

The architecture I used for the model was based on Nvidia's *PilotNet* system.

The layers, output shape and parameter information are as follows:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 160, 320, 3)       0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 75, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 36, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 33, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 2, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               422500    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0
_________________________________________________________________
```
*NOTE: The Lambda layer normalizes the image data values between -0.5 to 0.5, and the
Cropping layer crops 60 pixels off the top and 20 pixels off the bottom of each
image.  These adjustments should improve learning and remove unnecessary noise.*

### Overfitting Protection

To help guard against overfitting, the data set was split into separate
training and validation sets, with an 80/20 distribution respectively.
Additionally, a Dropout layer is included after the fifth convolutional layer.
Using these techniques, the model was trained for **30 epochs**
 
![training-loss]



## Training Strategy

The training and validation sets are divided into batches, and are produced
on the fly using generator methods.  This helps to limit system RAM usage,
and trains rather quickly, since it can access available CPU power, while
the model trains on the GPU.  For my system, a **batch size of 512** worked
quite well.

### Batch Generator

The batch generator shuffles the samples, and then divides them into
batches.  For each sample in each batch, a left, right or center image
is selected at random.  If a left or right image is selected, the steering
angle is adjusted 0.2 towards the center.

Next, the generator initiates some possible image augmentation:

### Image Augmentation

The augmentation step randomly selects 50% of all images for augmentation. Once
selected, each image gets one of the following augmentations applied to it:

1. Horizontal flip
    - Helps to perceptually provide additional training data
    - ![flipped]    
2. Randomly adjust brightness (using HSL color space)
    - Helps to provide tolerance to different lighting conditions
    - ![brightness] 
3. Scale down horizontally and add padding back to original size
    - Helps the model learn that lanes come in many sizes
    - ![scale] 
4. Randomly add shadows shadows
    - Helps the model learn to ignore shadows on the road
    - ![shadow] 

## Evaluation & Results

The results are evaluated in the same driving simulator that the data was
collected from.  A socketio server, configured in `drive.py`, is used to 
receive image data inputs and send back steering wheel and throttle outputs.
The steering angle is handled by our model, while the throttle is handled by
a simple controller class that aims at keeping a consistent speed throughout
the course. 

As the videos demonstrate, the trained model is able to successfully drive 
around each track and stay in the proper lane for the two-lane, mountainous
track.

## CNN Visualization

Inspired by Nvidia's techniques for visualizing the internal workings of the
PilotNet CNN, I decided to create some visualizations as well.  

**Orignal Input Image**

![original-input]

**Feature Maps:**

![conv-1-fm]

![conv-2-fm]

![conv-3-fm]

![conv-4-fm]

![conv-5-fm]

**Averaged Convolutional Filters Per Layer:**

![avg-fm]

**Salient Object Masks:**

![salient-objs]

*NOTE: These are created by taking the final averaged layer, up-scaling them 
to the size of the previous layer and combining them with pointwise 
multiplication.  This is repeated until the final layer is up-scaled to the
size of the original cropped input image.  See [Nvidia's article](https://arxiv.org/pdf/1704.07911.pdf)
for a much more in-depth explanation.*

As you can see, the network activations seem to be strongest at the edges of 
the road and lane markings.  The networks seems to have identified the salient
objects with a reasonable degree of accuracy.

## Future Research & Improvements

Nvidia's PilotNet used deconvolution to up-scale their salient object masks.
My approach is just using OpenCV's resize method.  A future improvement 
could be to use a similar deconvolutional up-scaling technique

Additionally, I found that the model's driving on the first track got more
unstable as I added data for the second track.  An interesting point of
further research would be to explore techniques to keep driving smooth on
both tracks.
