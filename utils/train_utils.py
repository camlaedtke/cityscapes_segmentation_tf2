import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm


class TrainAccumilator:
    
    # TODO: Change hard-coded metrics and callbacks
    
    def __init__(self, accum_steps, model, optimizer, loss_fn, n_classes, reduce_lr_on_plateau=None):
        self.accum_steps = accum_steps
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_miou_metric = tf.keras.metrics.MeanIoU(num_classes=n_classes)
        
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_miou_metric = tf.keras.metrics.MeanIoU(num_classes=n_classes)
        
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        
        self.history = {
            "loss": [], 
            "accuracy": [], 
            "miou": [], 
            "val_loss": [], 
            "val_accuracy": [],
            "val_miou": [], 
            "lr": []
        }
        
        
    def log_metrics(self, train_loss, train_acc, train_miou, val_loss, val_acc, val_miou, current_lr):
        
        self.history['loss'].append(train_loss)
        self.history['accuracy'].append(train_acc)
        self.history['miou'].append(train_miou)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_acc)
        self.history['val_miou'].append(val_miou)
        self.history['lr'].append(current_lr)
        
        self.train_acc_metric.reset_states()
        self.train_miou_metric.reset_states()
        self.val_acc_metric.reset_states()
        self.val_miou_metric.reset_states()
        
        
    def do_callbacks(self):
        if self.reduce_lr_on_plateau is not None:
            self.optimizer.inner_optimizer = self.reduce_lr_on_plateau.update(self.history, self.optimizer)
        
        
    @tf.function
    def accumilate_train_step(self, x, y, accum_gradient, train_vars):
        
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True) 
            loss_value = self.loss_fn(y, logits)
            scaled_loss_value = self.optimizer.get_scaled_loss(loss_value)

        scaled_gradients = tape.gradient(scaled_loss_value, train_vars)
        
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        
        accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, gradients)]
        
        self.train_acc_metric.update_state(y, logits)
        self.train_miou_metric.update_state(y, tf.argmax(logits, axis=-1))
        
        return loss_value, accum_gradient


    @tf.function
    def test_step(self, x, y):
        
        logits = self.model(x, training=False)
        loss_value = self.loss_fn(y, logits)
        
        self.val_acc_metric.update_state(y, logits)
        self.val_miou_metric.update_state(y, tf.argmax(logits, axis=-1))
        return loss_value
    
    
    def train_model(self, train_dataset):
        batch_loss = 0
        train_batch_losses = []
        
        # get trainable variables
        train_vars = self.model.trainable_variables 
        # Create empty gradient list (not a tf.Variable list)
        accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]

        for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_dataset)):

            loss_value, accum_gradient = self.accumilate_train_step(
                x_batch_train, y_batch_train, accum_gradient, train_vars)

            batch_loss += loss_value

            # Now, after executing all the tapes we need, we apply the optimization step
            if (step > 0) and (step % self.accum_steps == 0):

                self.optimizer.apply_gradients(zip(accum_gradient, train_vars))
                
                train_batch_losses.append(batch_loss.numpy()/self.accum_steps)

                # reset batch loss, trainable variables, accum gradients
                batch_loss = 0
                train_vars = self.model.trainable_variables 
                accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
                
        return train_batch_losses
    
    
    def validate_model(self, test_dataset):
        test_batch_losses = []
        
        for x_batch_val, y_batch_val in test_dataset:
            loss_value = self.test_step(x_batch_val, y_batch_val)
            test_batch_losses.append(loss_value.numpy())
            
        return test_batch_losses
        
        
    def fit(self, epochs, train_dataset, test_dataset, weights_path):
       
        for epoch in range(epochs):
        
            train_batch_losses = self.train_model(train_dataset)
                    
            test_batch_losses = self.validate_model(test_dataset)
                
            # Get all metrics
            train_loss = np.mean(train_batch_losses)
            train_acc = self.train_acc_metric.result().numpy()
            train_miou = self.train_miou_metric.result().numpy()
            
            val_loss = np.mean(test_batch_losses)
            val_acc = self.val_acc_metric.result().numpy()
            val_miou = self.val_miou_metric.result().numpy()
            
            # curr_lr = self.optimizer.learning_rate
            curr_lr = self.optimizer.inner_optimizer._decayed_lr(tf.float32)
            curr_lr = curr_lr.numpy()
            
            self.log_metrics(train_loss, train_acc, train_miou, val_loss, val_acc, val_miou, curr_lr)
            
            self.do_callbacks()
            
            print("\nEpoch {} - loss: {:.4f} , accuracy: {:.4f}, miou: {:.4f},"\
                  " val_loss: {:.4f}, val_accuracy: {:.4f}, val_miou: {:.4f}, lr: {:.10f}".format(epoch+1, 
                   train_loss, train_acc, train_miou, val_loss, val_acc, val_miou, curr_lr))
            
            self.model.save_weights(weights_path)
            
        print("Training finished")
        return self.history

            

            

        


