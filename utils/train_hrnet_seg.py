import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision

from utils.data_utils import get_labels
# from models.seg.hrnet_accumilate import HRNet
from models.seg.hrnet import HRNet
from data_loaders import CityscapesLoader


K.clear_session()
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def enable_amp():
    mixed_precision.set_global_policy("mixed_float16")

print("Tensorflow version: ", tf.__version__)
print(physical_devices,"\n")
enable_amp() 


fine = True

n_classes = 20
img_height = 512
img_width = 1024

BATCH_SIZE = 12
BUFFER_SIZE = 512

labels = get_labels()
trainid2label = { label.trainId : label for label in labels }
catid2label = { label.categoryId : label for label in labels }


pipeline = CityscapesLoader(
    img_height=img_height, 
    img_width=img_width, 
    n_classes=n_classes
)


dataset, info = tfds.load(
    name = 'cityscapes/semantic_segmentation', 
    data_dir = '/workspace/tensorflow_datasets/', 
    with_info = True,
    shuffle_files=True
)


train = dataset['train'].map(pipeline.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
valid = dataset['validation'].map(pipeline.load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

TRAIN_LENGTH = info.splits['train'].num_examples
VALID_LENGTH = info.splits['validation'].num_examples

train_dataset = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
valid_dataset = valid.batch(BATCH_SIZE)


model = HRNet(
    stage1_cfg = {'NUM_MODULES': 1,'NUM_BRANCHES': 1,'BLOCK': 'BOTTLENECK','NUM_BLOCKS': [4]}, 
    stage2_cfg = {'NUM_MODULES': 1,'NUM_BRANCHES': 2,'BLOCK': 'BASIC',     'NUM_BLOCKS': [4, 4]},
    stage3_cfg = {'NUM_MODULES': 4,'NUM_BRANCHES': 3,'BLOCK': 'BASIC',     'NUM_BLOCKS': [4, 4, 4]},
    stage4_cfg = {'NUM_MODULES': 3,'NUM_BRANCHES': 4,'BLOCK': 'BASIC',     'NUM_BLOCKS': [4, 4, 4, 4]},
    input_height = img_height, 
    input_width = img_width, 
    n_classes = n_classes, 
    W = 48,
    conv_upsample=True,
)


print(model.summary())

MODEL_PATH = "weights/"+model.name+".h5"

model.load_weights("weights/HRNet_CLF_W48.h5", by_name=True)


def iou_coef(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)
    y_pred = tf.math.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=n_classes)
    smooth = 1
    iou_total = 0
    for i in range(1, n_classes):
        intersection = tf.math.reduce_sum(y_true[:,:,:,i] * y_pred[:,:,:,i], axis=(1,2))
        union = tf.math.reduce_sum(y_true[:,:,:,i] + y_pred[:,:,:,i], axis=(1,2)) 
        iou = tf.math.reduce_mean(tf.math.divide_no_nan(2.*intersection+smooth, union+smooth), axis=0)
        iou_total += iou
    return iou_total/(n_classes-1)


def weighted_cross_entropy_loss(y_true_labels, y_pred_logits):
    c_weights = [0.0,    2.602,  6.707,  3.522,  9.877, 9.685,  9.398,  10.288, 9.969,  4.336, 
                 9.454,  7.617,  9.405,  10.359, 6.373, 10.231, 10.262, 10.264, 10.394, 10.094] 
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_labels, logits=y_pred_logits)  
    weights = tf.gather(c_weights, y_true_labels)  
    losses = tf.multiply(losses, weights)
    return tf.math.reduce_mean(losses)


EPOCHS = 350
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE 
VALIDATION_STEPS = VALID_LENGTH // BATCH_SIZE 
DECAY_STEPS = (STEPS_PER_EPOCH * EPOCHS) # // ACCUM_STEPS
print("Decay steps: {}".format(DECAY_STEPS))


learning_rate_fn = PolynomialDecay(
    initial_learning_rate = 1e-2,
    decay_steps = DECAY_STEPS,
    end_learning_rate = 1e-5,
    power = 0.9
)


model.compile(
    optimizer = SGD(learning_rate=learning_rate_fn, momentum=0.9, decay=0.0005),
    loss = weighted_cross_entropy_loss,
    metrics = ['accuracy', iou_coef]
)


callbacks = [
    # ReduceLROnPlateau(monitor='val_iou_coef', mode='max', patience=10, factor=0.2, min_lr=1e-5, verbose=2),
    ModelCheckpoint(MODEL_PATH, monitor='val_iou_coef', mode='max', verbose=2, save_best_only=True, save_weights_only=True)    
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






