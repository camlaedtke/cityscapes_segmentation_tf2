# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import tensorflow as tf
import numpy as np
import glob
import os


class CityscapesLoader():
    
    def __init__(self, img_height, img_width, n_classes):

        self.n_classes = n_classes
        self.img_height = img_height
        self.img_width = img_width
        self.MEAN = np.array([0.485, 0.456, 0.406])
        self.STD = np.array([0.229, 0.224, 0.225])
        

    @tf.function
    def random_crop(self, img, seg):
        """
        Inputs: full resolution image and mask
        A scale between 0.5 and 1.0 is randomly chosen. 
        Then, we multiply original height and width by the scale, 
        and randomly crop to the scaled height and width.
        [0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]
        
        """
        scales = tf.convert_to_tensor(np.array(
        [0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0], 
            dtype=np.float32))
        scale = scales[tf.random.uniform(shape=[], minval=0, maxval=13, dtype=tf.int32)]

        shape = tf.cast(tf.shape(img), tf.float32)
        h = tf.cast(shape[0] * scale, tf.int32)
        w = tf.cast(shape[1] * scale, tf.int32)
        combined_tensor = tf.concat([img, seg], axis=2)
        combined_tensor = tf.image.random_crop(combined_tensor, size=[h, w, 4])
        return combined_tensor[:,:,0:3], combined_tensor[:,:,-1]


    @tf.function
    def normalize(self, img):
        img = img / 255.0
        img = img - self.MEAN
        img = img / self.STD
        return img

    
    @tf.function
    def load_image_train(self, input_image, input_seg):
        img = tf.cast(input_image, tf.uint8)
        seg = tf.cast(input_seg, tf.uint8)
        
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            seg = tf.image.flip_left_right(seg)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.random_brightness(img, 0.1)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.random_saturation(img, 0.7, 1.3)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.random_contrast(img, 0.7, 1.3)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.random_hue(img, 0.05)
            
        img, seg = self.random_crop(img, seg)
        seg = tf.expand_dims(seg, axis=-1)
        
        img = tf.image.resize(img, (self.img_height, self.img_width), method='bilinear')
        seg = tf.image.resize(seg, (self.img_height, self.img_width), method='nearest')
        img = self.normalize(tf.cast(img, tf.float32))
        
        seg = tf.squeeze(tf.cast(seg, tf.int32))
        return img, seg
    
    
    @tf.function
    def load_image_test(self, input_image, input_seg):
        img = tf.cast(input_image, tf.uint8)
        seg = tf.cast(input_seg, tf.uint8)
        
        # img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, (self.img_height, self.img_width), method='bilinear')
        seg = tf.image.resize(seg, (self.img_height, self.img_width), method='nearest')
        img = self.normalize(tf.cast(img, tf.float32))
        
        seg = tf.squeeze(tf.cast(seg, tf.int32))
        return img, seg
    
    
    @tf.function
    def load_image_eval(self, input_image, input_seg):
        img = tf.cast(input_image, tf.uint8)
        seg = tf.cast(input_seg, tf.uint8)
        
        seg = tf.expand_dims(seg, axis=-1)
        img = tf.image.resize(img, (self.img_height, self.img_width), method='bilinear')
        img = self.normalize(tf.cast(img, tf.float32))
        
        seg = tf.squeeze(tf.cast(seg, tf.int32))
        return img, seg


class CityscapesLoaderTFDS():
    
    def __init__(self, img_height, img_width, n_classes):

        self.n_classes = n_classes
        self.img_height = img_height
        self.img_width = img_width
        self.MEAN = np.array([0.485, 0.456, 0.406])
        self.STD = np.array([0.229, 0.224, 0.225])
        self.id2label = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3,  
                                    4, 5, 0, 0, 0, 6, 0, 7, 8, 9, 10, 11, 
                                    12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0], tf.int32)
        

    @tf.function
    def random_crop(self, img, seg):
        """
        Inputs: full resolution image and mask
        A scale between 0.5 and 1.0 is randomly chosen. 
        Then, we multiply original height and width by the scale, 
        and randomly crop to the scaled height and width.
        [0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]
        
        """
        scales = tf.convert_to_tensor(np.array(
        [0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0], 
            dtype=np.float32))
        scale = scales[tf.random.uniform(shape=[], minval=0, maxval=9, dtype=tf.int32)]

        shape = tf.cast(tf.shape(img), tf.float32)
        h = tf.cast(shape[0] * scale, tf.int32)
        w = tf.cast(shape[1] * scale, tf.int32)
        combined_tensor = tf.concat([img, seg], axis=2)
        combined_tensor = tf.image.random_crop(combined_tensor, size=[h, w, 4])
        return combined_tensor[:,:,0:3], combined_tensor[:,:,-1]


    @tf.function
    def normalize(self, img):
        img = img / 255.0
        img = img - self.MEAN
        img = img / self.STD
        return img

    
    @tf.function
    def load_image_train(self, datapoint):
        img = datapoint['image_left']
        seg = datapoint['segmentation_label']
        
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            seg = tf.image.flip_left_right(seg)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.random_brightness(img, 0.1)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.random_saturation(img, 0.7, 1.3)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.random_contrast(img, 0.7, 1.3)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.random_hue(img, 0.05)
            
        img, seg = self.random_crop(img, seg)
        seg = tf.expand_dims(seg, axis=-1)
        
        img = tf.image.resize(img, (self.img_height, self.img_width), method='bilinear')
        seg = tf.image.resize(seg, (self.img_height, self.img_width), method='nearest')
        img = self.normalize(tf.cast(img, tf.float32))
        
        seg = tf.squeeze(seg)
        seg = tf.gather(self.id2label, tf.cast(seg, tf.int32))
        
        return img, seg
    
    
    @tf.function
    def load_image_test(self, datapoint):
        img = datapoint['image_left']
        seg = datapoint['segmentation_label']
        
        # img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, (self.img_height, self.img_width), method='bilinear')
        seg = tf.image.resize(seg, (self.img_height, self.img_width), method='nearest')
        img = self.normalize(tf.cast(img, tf.float32))
        
        seg = tf.squeeze(seg, axis=-1)
        seg = tf.gather(self.id2label, tf.cast(seg, tf.int32))
        
        return img, seg
    
    
    @tf.function
    def load_image_eval(self, datapoint):
        img = datapoint['image_left']
        seg = datapoint['segmentation_label']
        seg = tf.expand_dims(seg, axis=-1)
        img = tf.image.resize(img, (self.img_height, self.img_width), method='bilinear')
        img = self.normalize(tf.cast(img, tf.float32))
        seg = tf.squeeze(seg)
        seg = tf.gather(self.id2label, tf.cast(seg, tf.int32))
        return img, seg


class SunLoader():
    
    def __init__(self, data_dir, bad_imgs_file, img_height, img_width, n_classes, img_pattern='*.jpg'):
        self.data_dir = data_dir
        self.bad_imgs_file = bad_imgs_file
        self.n_classes = n_classes
        self.img_height = img_height
        self.img_width = img_width
        self.img_pattern = img_pattern
        self.MEAN = np.array([0.4706051, 0.4594826, 0.4198055])
        self.STD = np.array([0.26260594, 0.26121938, 0.28267792])


    def get_image_list(self):
        letter = '*'
        category = '*'
        search_image_files1 = os.path.join(self.data_dir, letter,category, self.img_pattern)
        search_image_files2 = os.path.join(self.data_dir, letter,category,'*', self.img_pattern)
        image_list1 = glob.glob(search_image_files1)
        image_list2 = glob.glob(search_image_files2)
        print("Found {} files with search pattern {}".format(len(image_list1), search_image_files1))
        print("Found {} files with search pattern {}".format(len(image_list2), search_image_files2))
        image_list = image_list1 + image_list2

        with open(self.bad_imgs_file) as f:
            for line in f:
                image_list.remove(line.rstrip('\n'))
        f.close()

        np.random.shuffle(image_list)

        print("Filtered images: {}".format(len(image_list)))
        return image_list

    def parse_label(self, x):
        """only works with my specific file system setup"""
        x = x.split('/')[6]
        return x

    def get_label_list(self, img_list):
        label_list = []
        for i in range(0, len(img_list)):
            label_list.append(self.parse_label(img_list[i]))
        print("Total: {} labels with {} categories".format(len(label_list), len(set(label_list))))
        label_dict = dict(zip(set(label_list), [i for i in range(0, len(label_list))]))
        label_encodings = []
        for i in range(0, len(label_list)):
            label_encodings.append(label_dict[label_list[i]])
        return label_encodings

    @tf.function
    def random_crop(self, image):
        scales = tf.convert_to_tensor(np.array([0.75, 0.8125, 0.875, 0.9375, 1.0]))
        scale = scales[tf.random.uniform(shape=[], minval=0, maxval=5, dtype=tf.int32)]
        scale = tf.cast(scale, tf.float32)

        shape = tf.cast(tf.shape(image), tf.float32)
        h = tf.cast(shape[0] * scale, tf.int32)
        w = tf.cast(shape[1] * scale, tf.int32)
        image = tf.image.random_crop(image, size=[h, w, 3])
        return image

    @tf.function
    def normalize(self, image, label):
        image = image / 255.0
        image = image - self.MEAN
        image = image / self.STD
        return image, label
    
    
    @tf.function
    def load_image_train(self, image_path, label):

        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.cast(img, dtype=tf.float32)
        if tf.random.uniform(()) > 0.5:
            img = self.random_crop(img)
        img = tf.image.resize(images=img, size=[self.img_height, self.img_width])
        img = tf.image.random_brightness(img, 0.05)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.image.random_hue(img, 0.05)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, 0, 255)

        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)

        img, label = self.normalize(img, label)

        label = tf.one_hot(tf.cast(label, tf.int32), self.n_classes)

        return img, label


    def load_image_test(self, image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.image.resize(images=img, size=[self.img_height, self.img_width])
        img = tf.clip_by_value(img, 0, 255)
        img, label = self.normalize(img, label)
        label = tf.one_hot(tf.cast(label, tf.int32), self.n_classes)

        return img, label


class ImageNetLoader():
    
    def __init__(self, img_height, img_width, n_classes, sparse=False):
        self.n_classes = n_classes
        self.img_height = img_height
        self.img_width = img_width
        self.MEAN = np.array([0.485, 0.456, 0.406])
        self.STD = np.array([0.229, 0.224, 0.225])
        
    
    @tf.function
    def random_crop(self, image, label):

        scales = tf.convert_to_tensor(np.array([0.75, 0.8125, 0.875, 0.9375, 1.0]))
        scale = scales[tf.random.uniform(shape=[], minval=0, maxval=5, dtype=tf.int32)]
        scale = tf.cast(scale, tf.float32)

        shape = tf.cast(tf.shape(image), tf.float32)
        h = tf.cast(shape[0] * scale, tf.int32)
        w = tf.cast(shape[1] * scale, tf.int32)
        image = tf.image.random_crop(image, size=[h, w, 3])
        return image, label

    @tf.function
    def normalize(self, image, label):
        image = image / 255.0
        image = image - self.MEAN
        image = image / self.STD
        return image, label
    
    
    @tf.function
    def load_image_train(self, datapoint):

        image = datapoint['image']
        label = datapoint['label']
        label = tf.one_hot(tf.cast(label, tf.int32), self.n_classes)

        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)

        image, label = self.random_crop(image, label)
        image = tf.image.resize(image, (self.img_height, self.img_width))
        image, label = self.normalize(tf.cast(image, tf.float32), label)

        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_brightness(image, 0.05)
            image = tf.image.random_saturation(image, 0.6, 1.6)
            image = tf.image.random_contrast(image, 0.7, 1.3)
            image = tf.image.random_hue(image, 0.05)

        return image, label
   

    def load_image_test(self, datapoint):
        image = datapoint['image']
        label = datapoint['label']
        label = tf.one_hot(tf.cast(label, tf.int32), self.n_classes)
        image = tf.image.resize(image, (self.img_height, self.img_width))
        image, label = self.normalize(tf.cast(image, tf.float32), label)
        return image, label


def get_image_stats(X_train):
    
    R_MEAN = np.mean(X_train[:,:,:,0]) 
    G_MEAN = np.mean(X_train[:,:,:,1]) 
    B_MEAN = np.mean(X_train[:,:,:,2])

    print("Mean value of first channel: {}".format(R_MEAN))
    print("Mean value of second channel: {}".format(G_MEAN))
    print("Mean value of third channel: {}".format(B_MEAN))

    R_STD = np.std(X_train[:,:,:,0]) 
    G_STD = np.std(X_train[:,:,:,1]) 
    B_STD = np.std(X_train[:,:,:,2]) 

    print("Std of first channel: {}".format(R_STD))
    print("Std of second channel: {}".format(G_STD))
    print("Std of third channel: {}".format(B_STD))

    RGB_MEAN = np.array([R_MEAN, G_MEAN, B_MEAN])
    RGB_STD = np.array([R_STD, G_STD, B_STD])

    return RGB_MEAN, RGB_STD
