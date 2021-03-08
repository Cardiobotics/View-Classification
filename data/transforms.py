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
        img = self.t_normalize_signed(img)
        if img.shape[1] != self.target_height or img.shape[2] != self.target_width:
            img = self.t_resize(img)
        return img

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
        return resize(img, (self.target_length, self.target_height, self.target_width), mode='constant',
                      cval=self.black_val, preserve_range=True, anti_aliasing=True)

    def t_normalize_signed(self, img):
        '''
        Normalizes an image with values ranging from 0-255 into the range -1:1.
        :param img: The input image to be transformed.
        :return: The image with values normalized to the new range in float format.
        '''
        return ((img / 255.) * 2 - 1).astype(np.float32)