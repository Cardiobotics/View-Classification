import numpy as np
from skimage.transform import resize


class CustomTransforms:
    def __init__(self, target_length, target_height, target_width):
        self.target_length = target_length
        self.target_height = target_height
        self.target_width = target_width

        self.black_val = -1.0
        self.white_val = 1.0

    def apply_transforms(self, img):
        if len(img.shape) == 4:
            img = self.t_grayscale_mean(img)
            img = np.squeeze(img)
        if img.shape[1] != self.target_height or img.shape[2] != self.target_width:
            img = self.t_resize(img)
        img = self.t_normalize_signed(img)
        return img

    def t_grayscale_mean(self, img):
        '''
        Converts a RBG video/image into a grayscale one by taking the mean over all channels.
        :param img: The input image, shape assumed to be (L,H,W,C)
        :return: The transformed image in shape (L,H,W,1)
        '''
        img = np.average(img, axis=-1)
        return img.astype(np.uint8)


    def t_resize(self, img):
        '''
        Resizes the input image to the supplied length, height and width using interpolation.
        The number of channels is preserved.
        :param img: The input image, shape is assumed to be (L,H,W,C)
        :param target_length: The new length the input image will be resized into
        :param target_height: The new height the input image will be resized into
        :param target_width: The new width the input image will be resized into
        :return: The resized image with shape (target_length, target_height, target_width, C)
        '''
        if self.target_length is None:
            target_length = img.shape[0]
        else:
            target_length = self.target_length
        return resize(img, (target_length, self.target_height, self.target_width), mode='constant',
                      cval=self.black_val, preserve_range=True, anti_aliasing=True)

    def t_normalize_signed(self, img):
        '''
        Normalizes an image with values ranging from 0-255 into the range -1:1.
        :param img: The input image to be transformed.
        :return: The image with values normalized to the new range in float format.
        '''
        return ((img / 255.) * 2 - 1).astype(np.float32)