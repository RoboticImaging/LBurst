# Visualisation of robotic burst

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def translate_images(image, shift_x, shift_y, burst_size=5):

    burst_length = int((burst_size - 1) / 2)
    if shift_x == 0:
        translate_images_left = np.concatenate(
            ([np.roll(image, (i + 1) * (-1) * shift_y, axis=0) for i in range(burst_length)]), axis=2)
        translate_images_right = np.concatenate(
            ([np.roll(image, (i + 1) * 1 * shift_y, axis=0) for i in range(burst_length)]), axis=2)
        translated_stack = np.concatenate((translate_images_left, image, translate_images_right), axis=2)

    elif shift_y == 0:
        translate_images_left = np.concatenate(
            ([np.roll(image, (i + 1) * (-1) * (shift_x), axis=1) for i in range(burst_length)]), axis=2)
        translate_images_right = np.concatenate(
            ([np.roll(image, (i + 1) * (1) * (shift_x), axis=1) for i in range(burst_length)]), axis=2)
        translated_stack = np.concatenate((translate_images_left, image, translate_images_right), axis=2)

    else:
        translate_images_left = np.concatenate(
            ([np.roll(image, [((i + 1) * (-1) * (shift_x)),((i + 1) * (-1) * (shift_y))], axis=(0,1)) for i in range(burst_length)]), axis=2)
        translate_images_right = np.concatenate(
            ([np.roll(image, [((i + 1) * (1) * (shift_x)), ((i + 1) * (1) * (shift_y))], axis=(0, 1)) for i in range(burst_length)]), axis=2)
        translated_stack = np.concatenate((translate_images_left, image, translate_images_right), axis=2)
    return translated_stack

def generate_singleburst(image, crop_size=500, burst_size=5, shift_x=2, shift_y=0, mean=0, variance=10):

    shift_x = int(np.random.default_rng().uniform(-3, 3, 1))
    shift_y = int(np.random.default_rng().uniform(-3, 3, 1))
    translated_stack = translate_images(image, shift_x, shift_y, burst_size)
    return translated_stack


def generate_burst(image1, image2, crop_size=500, burst_size=5, shift_x=2, shift_y=0, mean=0, variance=10):
    shift_x = int(np.random.default_rng().uniform(-3, 3, 1))
    shift_y = int(np.random.default_rng().uniform(-3, 3, 1))
    translated_stack_1 = translate_images(image1, shift_x, shift_y, burst_size)
    translated_stack_2 = translate_images(image2, shift_x, shift_y, burst_size)

    return translated_stack_1, translated_stack_2

if __name__ == "__main__":
    image = np.array(Image.open('/home/ahalya/local_burstnet/testimg/test_r2d2.png').convert('RGB'))
    plt.imshow(image, interpolation='nearest')
    plt.show()
    translated_stack = translate_images(image, shift_x=2, shift_y=2)

    merge_translatedstack = np.mean(translated_stack, axis=2)
    plt.imshow(merge_translatedstack, interpolation='nearest')
    plt.show()
