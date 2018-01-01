import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.python.keras.layers import Input, Lambda, Cropping2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K

from PIL import Image
import csv
import cv2
import pickle
import math

import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# Define command line flags
flags.DEFINE_integer('epochs', 10, 'The number of epochs')
flags.DEFINE_integer('batch_size', 512, "the batch size")
flags.DEFINE_string('op', 'train', "The operation to use. 'explore_data', 'model_summary', 'show_training_sample',\
'train', 'show_training_history', 'show_feature_maps', 'show_averaged_outputs', 'show_salient_obj_masks', \
'show_full_visualization', 'show_random_sample")


class DriveModel:
    def __init__(self):
        print('initializing')
        self.model = None
        self.train_samples = None
        self.validation_samples = None

    def __load_data(self):
        with open('./data/driving_log.csv', 'r') as f:
            samples = list(csv.reader(f))

        self.train_samples, self.validation_samples = train_test_split(samples, test_size=0.2)

    def __generator(self, samples, training=False):
        num_samples = len(samples)
        while 1: # Loop forever so the generator never terminates
            np.random.shuffle(samples)
            for offset in range(0, num_samples, FLAGS.batch_size):
                batch_samples = samples[offset:offset+FLAGS.batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    # get random number between 0-2 and read in an image
                    random_choice = np.random.choice(3)
                    random_image = np.asarray(Image.open(batch_sample[random_choice]))
                    
                    # get steering angle and set correction param
                    steer_angle = float(batch_sample[3])
                    correction = 0.2

                    # set if the image is left/right camera, adjust based on correction
                    if random_choice == 2:
                        # right camera
                        steer_angle -= correction

                    if random_choice == 1:
                        # left camera
                        steer_angle += correction
                        
                    if training is True:
                        random_image, steer_angle = utils.augment(random_image, steer_angle)

                    images.append(utils.preprocess(random_image))
                    angles.append(steer_angle)

                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)

    def __architecture(self):
        """
        Inspired by NVIDIA's PilotNet:
        https://arxiv.org/pdf/1704.07911.pdf &
        https://arxiv.org/pdf/1604.07316.pdf
        """

        print('initializing model architecture')
        center_image = cv2.imread(self.train_samples[0][0])
        input_shape = center_image.shape

        print('shape', input_shape)
        inp = Input(shape=input_shape)
        x = Lambda(lambda image_data: (image_data / 255.0) - 0.5)(inp)
        x = Cropping2D(cropping=((60, 25), (0, 0)))(x)
        x = Conv2D(24, 5, 2, activation='elu')(x)
        x = Conv2D(36, 5, 2, activation='elu')(x)
        x = Conv2D(48, 5, 2, activation='elu')(x)
        x = Conv2D(64, 3, 1, activation='elu')(x)
        x = Conv2D(64, 3, 1, activation='elu')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(100, activation='elu')(x)
        x = Dense(50, activation='elu')(x)
        x = Dense(10, activation='elu')(x)
        x = Dense(1)(x)
        self.model = Model(inp, x)
        self.model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

    def __get_random_image(self):
        self.__load_data()

        # The validation set is already shuffled, so grabbing the first will essentially work as "random"
        return np.asarray(Image.open(self.validation_samples[0][0]))

    def __calculate_layer_outputs(self, x=None):

        self.model = load_model('model.h5')

        if x is None:
            x = self.__get_random_image()

        sess = K.get_session()

        # 5 convolutional layers: 3, 4, 5, 6, 7
        layers = self.model.layers[3:8]

        out = []
        for layer in layers:
            # calculate the output activations for a given input image
            out.append(sess.run(layer.output, feed_dict={self.model.input: x[None, :, :, :]}))

        return out

    def __get_feature_maps(self, x=None):
        """ Get an array of convolutional filter outputs for each layer """
        if x is None:
            x = self.__get_random_image()

        outputs = self.__calculate_layer_outputs(x)

        feature_maps = []
        for idx, out in enumerate(outputs):
            # get each feature map
            feature_maps.append([])
            num_feature_maps = out.shape[3]

            for feature_map_idx in range(num_feature_maps):
                # add output of each filter
                feature_maps[idx].append(out[0, :, :, feature_map_idx])

        return feature_maps

    def __get_averaged_outputs(self, x=None):
        """ get an array of averaged filter outputs """
        if x is None:
            x = self.__get_random_image()

        outputs = self.__calculate_layer_outputs(x)

        avg_outs = []
        for idx, out in enumerate(outputs):
            avg_outs.append(np.mean(out, axis=3)[0, :, :] + 0.5)

        return avg_outs

    def __get_salient_obj_masks(self, x=None):
        """ get an array of salient object masks """
        if x is None:
            x = self.__get_random_image()

        avg_outputs = self.__get_averaged_outputs(x)[::-1]

        # create an array of multiplied, averaged outputs
        # NOTE: Nvidia's PilotNet used deconvolution to up-scale their images.
        #       This method is just using OpenCV's resize method.  A future
        #       improvement could be to use a similar deconvolution technique
        mult_outs = []
        for idx in range(len(avg_outputs)):
            if idx == 0:
                # first one
                mult_outs.append(cv2.resize(avg_outputs[idx], avg_outputs[idx + 1].shape[::-1]))

            elif idx == len(avg_outputs) - 1:
                # last one
                mult_outs.append(cv2.resize(cv2.multiply(mult_outs[len(mult_outs) - 1], avg_outputs[idx]), (320, 75)))
            else:
                # all others
                mult_outs.append(
                    cv2.resize(
                        cv2.multiply(mult_outs[len(mult_outs) - 1], avg_outputs[idx]),
                        avg_outputs[idx + 1].shape[::-1]))

        return mult_outs[::-1]

    def explore_data(self):
        """ Display sample content from driving_log.csv """

        self.__load_data()

        print(self.train_samples[:1])
        print()
        print(self.validation_samples[:1])
        # print(np.array(self.X_train).shape, np.array(self.X_valid).shape)

    def model_summary(self):
        """ Print out summary of model architecture """

        self.__load_data()
        self.__architecture()
        self.model.summary()

    def show_random_sample(self, crop=False):
        """ Display a random sample: left, center and right images together """

        # read csv
        self.__load_data()

        # select random image
        row = np.random.choice(len(self.train_samples))

        plt.figure(1, figsize=(9, 2))
        camera = "center"

        # iterate through the left, center and right images and place them in the subplot
        for idx, src in enumerate(self.train_samples[row][:3]):
            image = cv2.imread(src)
            steer_angle = float(self.train_samples[row][3])

            plot_order = 2
            if idx == 2:
                # right camera
                steer_angle -= 0.2
                camera = "right"
                plot_order = 3

            if idx == 1:
                # left camera
                steer_angle += 0.2
                camera = "left"
                plot_order = 1

            if crop is True:
                image = image[60:-25, :, :]

            plt.subplot(1, 3, plot_order)
            plt.title(camera + ", " + str(steer_angle), fontsize=10)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.tight_layout()

        plt.show()

    def show_training_sample(self):
        """ Display a single, random example of a training image. Possible augmentations included. """

        # read csv
        self.__load_data()

        # select random image
        random_choice = np.random.choice(3)
        # row = np.random.choice(len(self.train_samples))
        row = 0
        image = cv2.imread(self.train_samples[row][random_choice])
        steer_angle = float(self.train_samples[row][3])
        print(self.train_samples[row][random_choice])
        
        # crop
        image = image[60:-25, :, :]
        print(image.shape)

        camera = "center"

        if random_choice == 2:
            # right camera
            steer_angle -= 0.2
            camera = "right"

        if random_choice == 1:
            # left camera
            steer_angle += 0.2
            camera = "left"

        # augment
        image, steer_angle = utils.augment(image, steer_angle)
        print('steering angle', steer_angle)

        # show
        plt.figure(1, figsize=(9, 2))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(camera + ", " + str(steer_angle))
        plt.show()

    def train(self):
        """ Train the model for `FLAGS.epochs` epochs with generators """

        print('training')
        self.__load_data()
        self.__architecture()
        train_generator = self.__generator(self.train_samples, training=True)
        validation_generator = self.__generator(self.validation_samples)

        training_history = self.model.fit_generator(
                train_generator,
                steps_per_epoch=len(self.train_samples)/FLAGS.batch_size,
                validation_data=validation_generator,
                validation_steps=len(self.validation_samples)/FLAGS.batch_size,
                epochs=FLAGS.epochs)

        # save the model for later
        self.model.save('model.h5')

        # save the training history for later
        with open('training_history.pickle', 'wb') as f:
            pickle.dump(training_history.history, f)

        # show the training history
        self.show_training_history(training_history.history)

    def show_training_history(self, training_history=None):
        """ Display a graph of the training/validation loss history """

        if training_history is None:
            with open('training_history.pickle', 'rb') as f:
                training_history = pickle.load(f)

        # plot the training and validation loss for each epoch
        plt.plot(training_history['loss'])
        plt.plot(training_history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()

    def show_averaged_outputs(self, x=None):
        """ See what the Convolutional Neural Network sees via averaged filters"""
        if x is None:
            x = self.__get_random_image()

        avg_outputs = self.__get_averaged_outputs(x)

        plt.figure(1)
        plt.title('original input')
        plt.imshow(x)
        plt.tight_layout()

        plt.figure(2, figsize=(9, 4))
        for idx, avg_out in enumerate(avg_outputs):
            plt.subplot(2, 3, idx + 1)
            plt.title('Conv Layer:' + str(idx + 1))
            plt.imshow(avg_out, cmap='gray')

        plt.show()

    def show_feature_maps(self, x=None):
        """ See what the Convolutional Neural Network sees via featuremaps"""

        if x is None:
            x = self.__get_random_image()

        feature_maps = self.__get_feature_maps(x)

        plt.figure(1)
        plt.title('original input')
        plt.imshow(x)
        plt.tight_layout()

        for idx, feature_map in enumerate(feature_maps):

            col_n = 3
            row_n = math.ceil(len(feature_map)/col_n)

            # add 1 for a bit more vertical space in the figsize
            plt.figure(idx + 2, figsize=(9, row_n+1))
            plt.suptitle('Conv Layer ' + str(idx + 1) + ' Feature Map')

            for i, filter_output in enumerate(feature_map):
                plt.subplot(row_n, col_n, i + 1)
                plt.title('FeatureMap ' + str(i), fontsize=8)
                plt.imshow(filter_output, interpolation="nearest", cmap="gray")
                plt.tick_params(axis='both', which='both', labelsize=8)

        plt.show()

    def show_salient_obj_masks(self, x=None):
        """ See what the Convolutional Neural Network sees via salient object masks"""

        if x is None:
            x = self.__get_random_image()

        salient_obj_masks = self.__get_salient_obj_masks(x)

        plt.figure(1)
        plt.title('original input')
        plt.imshow(x)
        plt.tight_layout()

        plt.figure(2, figsize=(9, 4))

        for idx, salient_obj_mask in enumerate(salient_obj_masks):
            # print(salient_obj_mask)
            plt.subplot(3, 2, idx + 1)
            plt.title('Conv Layer:' + str(idx + 1))
            plt.imshow(salient_obj_mask, cmap='gray')

        plt.show()

    def show_full_visualization(self):
        """ show all internal visualization on the same image """
        x = self.__get_random_image()

        self.show_feature_maps(x)
        self.show_averaged_outputs(x)
        self.show_salient_obj_masks(x)

    def show_salient_obj_highlights(self, x=None):
        """ TODO: Highlight original image with salient object mask """
        print("this method is under construction")
        if x is None:
            x = self.__get_random_image()

        salient_obj_masks = self.__get_salient_obj_masks(x)

        final_mask = salient_obj_masks[0]

        def rescale_distribution(img):
            # scale so lowest value is 0 and highest is 1
            max_val = np.max(img.flatten())
            min_val = np.min(img.flatten())

            # return (img - min_val) / (max_val - min_val)
            return (img - min_val) * (1 / (max_val - min_val))

        rescaled_final_mask = rescale_distribution(final_mask)

        backtorgb = cv2.cvtColor(rescaled_final_mask, cv2.COLOR_GRAY2RGB)

        print(backtorgb.shape, x.shape)

        backtorgb = np.pad(backtorgb, ((60, 25), (0, 0), (0, 0)), mode='constant', constant_values=0)

        print(backtorgb)

        backtorgb = (255 * backtorgb).astype(np.uint8)

        print(backtorgb.shape, x.shape)

        final_masked = cv2.subtract(x, backtorgb)

        # ret, thresh_mask = cv2.threshold(rescaled_final_mask, 0.65, 1.0, cv2.THRESH_TOZERO)
        #
        plt.imshow(final_masked)
        plt.show()
        # plt.imshow(thresh_mask, cmap="gray")
        # plt.show()
        #
        # print(thresh_mask)

        # This is a work in progress...


def main(_):
    drive_model = DriveModel()

    # Call the method specified in FLAGS.op
    getattr(drive_model, FLAGS.op)()


# parses flags and calls the 'main' function
if __name__ == '__main__':
    tf.app.run()
