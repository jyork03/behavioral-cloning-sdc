# Behaviorial Cloning Project

[//]: # (Image References)

[new-predictions]: ./writeup-assets/new-predictions.png "New Image Predictions"

![new-predictions]

## Overview

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

For a detailed explanation of the problem, and how it was solved, please
read [writeup.md](./writeup.md).

## Running the Code

Clone the project:

```bash
git clone git@github.com:jyork03/behavioral-cloning-sdc.git
```

Download the simulator:

- [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
- [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
- [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)

NOTE * On Windows 8 there is an issue where drive.py is unable to establish a data connection with the simulator.

unzip the simulator and run/execute it to start it.

To use the provided model, run the bash code below to start and connect the server to the simulator:

```bash
python drive.py model.h5
```

Dependencies:

- python3
- numpy
- matplotlib
- opencv3
- pickle
- PIL
- sklearn
- scipy
- csv
- tensorflow 1.4.0
- h5py
- eventlet
- flask
- socketio
- moviepy

## Contents

Code files:

- model.h5
- utils.py
- model.py
- drive.py
- video.py


### Using the Files

### `model.py`

Responsible for loading training data, constructing the model architecture, training the model and displaying information about it

NOTE: this file expects the training data to be stored in `./data/`, so if you record your own data, either put it there or change where this file looks for it.

Model operations are as follows:

```bash
python model.py -op explore_data
```

```bash
python model.py -op model_summary
```

```bash
python model.py -op show_training_sample
```

```bash
python model.py -op train -epochs 10
```

```bash
python model.py -op show_training_history
```

```bash
python model.py -op show_activation
```

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-12-19 16:10:23 EST]  12KiB 2017_19_09_21_10_23_424.jpg
[2017-12-19 16:10:23 EST]  12KiB 2017_19_09_21_10_23_451.jpg
[2017-12-19 16:10:23 EST]  12KiB 2017_19_09_21_10_23_477.jpg
[2017-12-19 16:10:23 EST]  12KiB 2017_19_09_21_10_23_528.jpg
[2017-12-19 16:10:23 EST]  12KiB 2017_19_09_21_10_23_573.jpg
[2017-12-19 16:10:23 EST]  12KiB 2017_19_09_21_10_23_618.jpg
[2017-12-19 16:10:23 EST]  12KiB 2017_19_09_21_10_23_697.jpg
[2017-12-19 16:10:23 EST]  12KiB 2017_19_09_21_10_23_723.jpg
[2017-12-19 16:10:23 EST]  12KiB 2017_19_09_21_10_23_749.jpg
[2017-12-19 16:10:23 EST]  12KiB 2017_19_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.
