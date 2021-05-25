import os
import sys
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import PolynomialDecay, PiecewiseConstantDecay
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from models.clf.hrnet_clf_accumilate import HRNet_CLF

K.clear_session()
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def enable_amp():
    mixed_precision.set_global_policy("mixed_float16")

print("Tensorflow version: ", tf.__version__)
print(physical_devices,"\n")
enable_amp() 


class ImageNetLoader():
    
    def __init__(self, img_height, img_width, n_classes):
        self.n_classes = n_classes
        self.img_height = img_height
        self.img_width = img_width
        self.MEAN = np.array([0.485, 0.456, 0.406])
        self.STD = np.array([0.229, 0.224, 0.225])
        
    
    @tf.function
    def random_crop(self, image):

        scales = tf.convert_to_tensor(np.array([0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]))
        scale = scales[tf.random.uniform(shape=[], minval=0, maxval=7, dtype=tf.int32)]
        scale = tf.cast(scale, tf.float32)

        shape = tf.cast(tf.shape(image), tf.float32)
        h = tf.cast(shape[0] * scale, tf.int32)
        w = tf.cast(shape[1] * scale, tf.int32)
        image = tf.image.random_crop(image, size=[h, w, 3])
        return image

    @tf.function
    def normalize(self, image):
        image = image / 255.0
        image = image - self.MEAN
        image = image / self.STD
        return image
    
    
    @tf.function
    def load_image_train(self, datapoint):

        img = datapoint['image']
        label = datapoint['label']
        label = tf.one_hot(tf.cast(label, tf.int32), self.n_classes)

        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)

        img = self.random_crop(img)
        img = tf.image.resize(img, (self.img_height, self.img_width))
        img = self.normalize(tf.cast(img, tf.float32))

        if tf.random.uniform(()) > 0.5:
            img = tf.image.random_brightness(img, 0.05)
            img = tf.image.random_saturation(img, 0.6, 1.6)
            img = tf.image.random_contrast(img, 0.7, 1.3)
            img = tf.image.random_hue(img, 0.05)

        return img, label
   

    def load_image_test(self, datapoint):
        img = datapoint['image']
        label = datapoint['label']
        label = tf.one_hot(tf.cast(label, tf.int32), self.n_classes)
        img = tf.image.resize(img, (self.img_height, self.img_width))
        img = self.normalize(tf.cast(img, tf.float32))
        return img, label
    
    
img_height = 224
img_width = 224
n_classes = 1000

pipeline = ImageNetLoader(
    n_classes = n_classes,
    img_height = img_height,
    img_width = img_width,
)



dataset, info = tfds.load(
    'imagenet2012:5.0.0', 
    data_dir='/workspace/tensorflow_datasets/', 
    with_info=True, 
    shuffle_files=True
)


train = dataset['train'].map(pipeline.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
valid = dataset['validation'].map(pipeline.load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

TRAIN_LENGTH = info.splits['train'].num_examples
VALID_LENGTH = info.splits['validation'].num_examples


BATCH_SIZE = 128
ACCUM_STEPS = 2
BUFFER_SIZE = 8192
ADJ_BATCH_SIZE = BATCH_SIZE * ACCUM_STEPS
print("Effective batch size: {}".format(ADJ_BATCH_SIZE))


train_dataset = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
valid_dataset = valid.batch(BATCH_SIZE)


model = HRNet_CLF(
    stage1_cfg = {'NUM_MODULES': 1,'NUM_BRANCHES': 1,'BLOCK': 'BOTTLENECK','NUM_BLOCKS': [4]}, 
    stage2_cfg = {'NUM_MODULES': 1,'NUM_BRANCHES': 2,'BLOCK': 'BASIC',     'NUM_BLOCKS': [4, 4]},
    stage3_cfg = {'NUM_MODULES': 4,'NUM_BRANCHES': 3,'BLOCK': 'BASIC',     'NUM_BLOCKS': [4, 4, 4]},
    stage4_cfg = {'NUM_MODULES': 3,'NUM_BRANCHES': 4,'BLOCK': 'BASIC',     'NUM_BLOCKS': [4, 4, 4, 4]},
    input_height = img_height, 
    input_width = img_width, 
    n_classes = n_classes, 
    W = 48,
    ACCUM_STEPS=ACCUM_STEPS
)


MODEL_PATH = "weights/"+model.name+".h5"

model.load_weights(MODEL_PATH)

print(model.summary())


CURR_EPOCH = 15
EPOCHS = 100 - CURR_EPOCH

STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = VALID_LENGTH // BATCH_SIZE
DECAY_STEPS = (STEPS_PER_EPOCH * EPOCHS) // ACCUM_STEPS


E1 = 30 - CURR_EPOCH
E2 = 60 - CURR_EPOCH
E3 = 90 - CURR_EPOCH

S1 = (STEPS_PER_EPOCH * E1) // ACCUM_STEPS
S2 = (STEPS_PER_EPOCH * E2) // ACCUM_STEPS
S3 = (STEPS_PER_EPOCH * E3) // ACCUM_STEPS

print("--- LR decay --- \nstep {}: {} \nstep {}: {} \nstep {}: {}".format(S1, 1e-2, S2, 1e-3, S3, 1e-4))


learning_rate_fn = PiecewiseConstantDecay(
    boundaries = [S1, S2, S3], 
    values = [0.1, 0.01, 0.001, 0.0001]
)


model.compile(
    optimizer = SGD(learning_rate=learning_rate_fn, momentum=0.9, decay=0.0001),
    loss=CategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)


callbacks = [
    ModelCheckpoint(
        MODEL_PATH, 
        monitor='val_accuracy', 
        mode='max', 
        verbose=2, 
        save_best_only=True, 
        save_weights_only=True
    )    
]


results = model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    callbacks=callbacks,
    verbose=1
)






























































































