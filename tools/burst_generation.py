# Robotic burst generation

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def translate_images(image, shift_x, shift_y, burst_size=5):
    """Returns translated stack of images
    Args:
    image: reference image
    shift_x: pixel motion in along  rows
    shift_y: pixel motion in along  columns
    burst_size: use odd numbers in the stack to select the middle image as the reference
    """
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

def center_crop(image, dim):
    """Returns center cropped reference image, and crop stack as required
    Args:
    image: reference image
    dim: dimensions (width, height) to be cropped
    """
    width, height = image.shape[1], image.shape[0]
    crop_width = dim[0] if dim[0] < image.shape[1] else image.shape[1]
    crop_height = dim[1] if dim[1] < image.shape[0] else image.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_image = image[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_image


def crop_center_array(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx, :]


def add_noise(image, mean=0, variance=255):
    """Returns noisy image for an input image
    Args:
    image: reference image
    variance (noise): sqrt(scale)
    """
    gauss_noise = np.random.normal(mean, np.square(variance), (image.shape[0], image.shape[1]))
    image = image.astype("int16")
    noisy_image = image + gauss_noise
    noisy_image = ceil_floor_image(noisy_image)
    return noisy_image


def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image


def normalization2(image, max, min):
    """Normalization to range of [min, max]
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """
    normalized_image = (image - np.min(image)) * (max - min) / (np.max(image) - np.min(image)) + min
    return normalized_image


def generate_singleburst(image, crop_size=500, burst_size=5, shift_x=2, shift_y=0, mean=0, variance=10):
    """Returns two burst stacks by taking two input images, translate, center crop and adding noise
    Args:
    image1: input image
    burst_size: burst frames (odd number)
    shift_x: pixel variation in rows
    shift_y: pixel variation in columns
    variance (noise): sqrt(scale) in gaussian distribution
    """
    shift_x = int(np.random.default_rng().uniform(-3, 3, 1))
    shift_y = int(np.random.default_rng().uniform(-3, 3, 1))
    translated_stack = translate_images(image, shift_x, shift_y, burst_size)
    crop_stack = center_crop(translated_stack, [crop_size, crop_size])
    burst_stack = np.stack(
        ([add_noise(crop_stack[:, :, k], mean, variance) for k in range(crop_stack.shape[2])]), axis=2)

    return burst_stack


def generate_burst(image1, image2, crop_size=500, burst_size=5, shift_x=2, shift_y=0, mean=0, variance=10):
    """Returns two burst stacks by taking two input images, translate, center crop and adding noise
    Args:
    image1: first input image
    image2: second input image
    burst_size: burst frames (odd number)
    shift_x: pixel variation in rows
    shift_y: pixel variation in columns
    variance (noise): sqrt(scale) in gaussian distribution
    """
    shift_x = int(np.random.default_rng().uniform(-3, 3, 1))
    shift_y = int(np.random.default_rng().uniform(-3, 3, 1))
    translated_stack_1 = translate_images(image1, shift_x, shift_y, burst_size)
    crop_stack_1 = center_crop(translated_stack_1, [crop_size, crop_size])
    burst_stack_1 = np.stack(
        ([add_noise(crop_stack_1[:, :, k], mean, variance) for k in range(crop_stack_1.shape[2])]), axis=2)

    translated_stack_2 = translate_images(image2, shift_x, shift_y, burst_size)
    crop_stack_2 = center_crop(translated_stack_2, [crop_size, crop_size])
    burst_stack_2 = np.stack(
        ([add_noise(crop_stack_2[:, :, k], mean, variance) for k in range(crop_stack_2.shape[2])]), axis=2)

    return burst_stack_1, burst_stack_2


# For visualisation, uncomment the following.

# if __name__ == "__main__":
    # image = np.array(Image.open('/imgs/toyimg2.png').convert('RGB'))
    # plt.imshow(image, interpolation='nearest')
    # plt.show()
    # translated_stack = translate_images(image, shift_x=2, shift_y=0)
    # merge_translatedstack = np.mean(translated_stack, axis=2)
    # plt.imshow(merge_translatedstack, interpolation='nearest')
    # plt.show()

    # crop_stack = center_crop(translated_stack, [500, 500])
    # merge_cropstack = np.mean(crop_stack, axis=2)
    # plt.imshow(merge_cropstack, interpolation='nearest')
    # plt.show()

    # generate_burst1, generate_burst2 = generate_burst(crop_stack[:, :, 0:3], crop_stack[:, :, 12:15], crop_size=500,
    #                                                 burst_size=5, shift_x=2, shift_y=0, mean=0, variance=10)

    # generate_burst = generate_singleburst(image), crop_size=500,
    #                                                  burst_size=5, shift_x=2, shift_y=0, mean=0, variance=10)

    # print(generate_burst.shape)
    # plt.imshow(generate_burst2[:, :, 0:3], interpolation='nearest')
    # plt.show()
    # burst_stackimage1 = np.mean(generate_burst1, axis=2)
    # burst_stackimage2 = np.mean(generate_burst2, axis=2)
    # plt.imshow(burst_stackimage1, interpolation='nearest')
    # plt.show()
    # print(generate_burst1.shape)
    # plt.imshow(burst_stackimage2, interpolation='nearest')
    # plt.show()

# """
# for n in range(crop_stack.shape[2]):
#   img.imsave(f'/home/ahalya/local_burstnet/testimg/test{n}.png',burst_stack[:,:,n]) (import matplotlib.image as img)
# """
