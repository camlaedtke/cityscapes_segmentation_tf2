"""
The functions in this script are my attempt at manually calculating mean IoU 
in a way that matches the Cityscapes dataset implementation.

Variations of mean IoU score 
  micro:    True positives, false positives, and false negatives are computed globally
  macro:    True positives, false positives, and false negatives are computed for each class
            and their unweighted mean is returned
  weighted: Metrics are computed for each class and returns the mean weighted by the 
            number of true instances in each class 
            
After comparing my calculations with tf.keras.metrics.MeanIoU(), they didn't quite match.

So for evaluation calculations I am using tf.keras.metrics.MeanIoU() instead 

"""


import numpy as np
import tensorflow as tf

n_classes = 20


# https://github.com/pikabite/segmentations_tf2/blob/master/configs/cityscape_hrnet.yaml
# do tf.constant instead of np.array when using inside loss function
# Not sure how these class weights are calculated
class_weights = np.array([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 
                 1.0865, 1.1529, 1.0507])


def get_mean_iou(y_true, y_pred):
    """IOU = TP / (TP + FN + FP)"""
   
    threshold = tf.math.reduce_max(y_pred, axis=-1, keepdims=True)
    # make sure [0, 0, 0] doesn't become [1, 1, 1]
    # Use abs(x) > eps, instead of x != 0 to check for zero
    y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    TP = tf.math.reduce_sum(y_pred[:,:,:,1:] * y_true[:,:,:,1:])
    FN = tf.math.reduce_sum(y_true[:,:,:,1:] * (1 - y_pred[:,:,:,1:])) 
    FP = tf.math.reduce_sum(y_pred[:,:,:,1:] * (1 - y_true[:,:,:,1:]))
    
    mean_iou_micro =  tf.math.divide_no_nan(TP, TP + FN + FP)
    
    iou_class = np.zeros((n_classes-1,))
    iou_class_weighted = np.zeros((n_classes-1,))
    
    for i in range(1, n_classes):
        tp = tf.math.reduce_sum(y_pred[:,:,:,i] * y_true[:,:,:,i])
        fn = tf.math.reduce_sum(y_true[:,:,:,i] * (1 - y_pred[:,:,:,i])) 
        fp = tf.math.reduce_sum(y_pred[:,:,:,i] * (1 - y_true[:,:,:,i])) 
        denominator = tp+fn+fp
        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class. 
        if denominator != 0:
            iou = tf.math.divide_no_nan(tp, denominator)
            iou_class[i-1] = iou
            iou_class_weighted[i-1] = iou * class_weights[i-1]
        else:
            # We need to differentiate between the class scores that are
            # zero because the IoU is zero, and the class scores that 
            # we are ignoring
            iou_class[i-1] = -1
            
    mean_iou_macro = np.mean(iou_class[iou_class != -1])
    mean_iou_weighted = np.mean(iou_class_weighted[iou_class_weighted != -1])
    
    return iou_class, mean_iou_micro.numpy(), mean_iou_macro, mean_iou_weighted


def evaluate_iou(model, dataset, n_samples):
    
    iou_class_scores = np.zeros((n_samples, n_classes-1))
    iou_micro_scores = np.zeros((n_samples,))
    iou_macro_scores = np.zeros((n_samples,))
    iou_weighted_scores = np.zeros((n_samples,))
    
    inf_times = np.zeros((n_samples, ))
    
    for idx, (image, mask) in enumerate(dataset):
        print("\r Predicting {} \ {} ".format(idx+1, n_samples), end='')
        
        X = np.expand_dims(image.numpy(), axis=0)
        y_true = np.expand_dims(mask.numpy(), axis=0)
        
        t_start = time.time()
        y_pred = model.predict(X)
        
        t_end = time.time()
        t_inf = t_end-t_start
        
        inf_times[idx] = t_inf
        
        if model.name == "u2net":
            y_pred = y_pred[0]
        y_pred = tf.image.resize(y_pred, (1024, 2048))
        
        iou_class, iou_micro, iou_macro, iou_weighed = get_mean_iou(y_true, y_pred)
        iou_class_scores[idx] = iou_class
        iou_micro_scores[idx] = iou_micro
        iou_macro_scores[idx] = iou_macro
        iou_weighted_scores[idx] = iou_weighed
        
        if idx == (n_samples-1):
            break
    
    print("Average inference time: {:.2f}s".format(np.mean(inf_times)))
            
    return iou_class_scores, np.mean(iou_micro_scores), iou_macro_scores, iou_weighted_scores