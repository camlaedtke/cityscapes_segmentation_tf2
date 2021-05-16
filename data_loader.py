import tensorflow as tf
import numpy as np


class DataLoader():
    
    # TODO: Explore additional augmentations, see if random crop is being implemented correctly

    def __init__(self, img_height, img_width, n_classes, sparse=False):

        self.n_classes = n_classes
        self.img_height = img_height
        self.img_width = img_width
        self.MEAN = np.array([0.485, 0.456, 0.406])
        self.STD = np.array([0.229, 0.224, 0.225])
        self.sparse = sparse


    @tf.function
    def random_crop(self, image, mask):
        """
        Inputs: full resolution image and mask
        A scale between 0.5 and 1.0 is randomly chosen. 
        Then, we multiply original height and width by the scale, 
        and randomly crop to the scaled height and width.
        """
        scales = tf.convert_to_tensor(np.array(
            [0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]))
        scale = scales[tf.random.uniform(shape=[], minval=0, maxval=9, dtype=tf.int32)]
        scale = tf.cast(scale, tf.float32)

        shape = tf.cast(tf.shape(image), tf.float32)
        h = tf.cast(shape[0] * scale, tf.int32)
        w = tf.cast(shape[1] * scale, tf.int32)
        combined_tensor = tf.concat([image, mask], axis=2)
        combined_tensor = tf.image.random_crop(combined_tensor, size=[h, w, 4])
        return combined_tensor[:,:,0:3], combined_tensor[:,:,-1]
    

    @tf.function
    def mask_to_categorical(self, image, mask):
        mask = tf.squeeze(mask)
        if self.sparse == False:
            mask = tf.one_hot(tf.cast(mask, tf.int32), self.n_classes)
        mask = tf.cast(mask, tf.int32)
        return image, mask


    @tf.function
    def normalize(self, image, mask):
        image = image / 255.0
        image = image - self.MEAN
        image = image / self.STD
        return image, mask


    @tf.function
    def load_image_train(self, input_image, input_mask):

        image = tf.cast(input_image, tf.uint8)
        mask = tf.cast(input_mask, tf.uint8)

        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)


        image, mask = self.random_crop(image, mask)
        mask = tf.expand_dims(mask, axis=-1)

        image = tf.image.resize(image, (self.img_height, self.img_width))
        mask = tf.image.resize(mask, (self.img_height, self.img_width))

        image, mask = self.normalize(tf.cast(image, tf.float32), mask)

        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_brightness(image, 0.05)
            image = tf.image.random_saturation(image, 0.6, 1.6)
            image = tf.image.random_contrast(image, 0.7, 1.3)
            image = tf.image.random_hue(image, 0.05)

        image, mask = self.mask_to_categorical(image, mask)
        mask = tf.squeeze(mask)

        return image, mask
   

    def load_image_test(self, input_image, input_mask):
        image = tf.image.resize(input_image, (self.img_height, self.img_width))
        mask = tf.image.resize(input_mask, (self.img_height, self.img_width))
        image, mask = self.normalize(tf.cast(image, tf.float32), mask)
        image, mask = self.mask_to_categorical(image, mask)
        mask = tf.squeeze(mask)
        return image, mask


    def load_image_eval(self, input_image, input_mask):
        image = tf.image.resize(input_image, (self.img_height, self.img_width))
        image, mask = self.normalize(tf.cast(image, tf.float32), input_mask)
        image, mask = self.mask_to_categorical(image, mask)
        mask = tf.squeeze(mask)
        return image, mask
