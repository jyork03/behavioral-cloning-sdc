import cv2
import numpy as np


def preprocess(img):
    # return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img


def augment(img, steering_angle):

    # leave 50% of the images alone
    if np.random.random() < 0.50:

        choice = np.random.randint(4)
        print(choice)
        if choice == 0:
            # flip half of the images
            img, steering_angle = flip(img, steering_angle)

        if choice == 1:
            # adjust brightness
            img = brightness(img)

        if choice == 2:
            # scale down and pad images
            img = scale_pad(img)

        if choice == 3:
            # add random shadows
            img = add_random_shadow(img)

    return img, steering_angle


def flip(img, steering_angle):
    # flip the images        
    image_flipped = cv2.flip(img, 1)
    steering_flipped = -steering_angle

    return image_flipped, steering_flipped


def brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # The 3rd value of the color channel (height, width, color channel)
    # represents brightness or luminosity
    adjustment = (255 - hsv[:, :, 2]) * (np.random.random() - 0.50)
    hsv[:, :, 2] = np.clip(adjustment + hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def scale_pad(img):
    # scale the image and add padding to keep shape consistency
    # this should help add lane size invariance
    height, width, depth = img.shape
    factor = np.random.randint(50, 100) / 100

    new_width = int(width * factor)
    #new_height = int(height * factor)
    new_height = height

    # make sure they're even numbers, so we can safely divide by 2
    if new_width % 2 != 0:
        new_width -= 1

    if new_height % 2 != 0:
        new_height -= 1

    img = cv2.resize(img, (new_width,
                           new_height))
    tb_border = int((height - new_height) / 2)
    lr_border = int((width - new_width) / 2)

    img = cv2.copyMakeBorder(img, tb_border, tb_border, lr_border, lr_border, cv2.BORDER_CONSTANT)

    return img


def add_random_shadow(image):
    # Source for this function from a great medium post:
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    shadow_mask[((X_m - top_x)*(bot_y - top_y) - (bot_x - top_x)*(Y_m-top_y) >= 0)] = 1

    #random_bright = .25+.7*np.random.uniform()
    # if np.random.randint(2) == 1:
    random_bright = .5
    cond1 = shadow_mask == 1
    cond0 = shadow_mask == 0
    if np.random.randint(2) == 1:
        image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
    else:
        image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright

    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return image

