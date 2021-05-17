import tensorflow as tf


class ReduceLROnPlateau:
    
    # TODO: Find out how to set the optimizer learning rate manually during training
    
    def __init__(self, monitor, mode, patience, factor, min_lr):

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.factor = factor 
        self.min_lr = min_lr
        
        self.wait = 0
        
        if mode == "max":
            self.best = 0
        elif mode == "min":
            self.best = 1e12


    def update(self, metric_logs, optimizer):
        
        current = metric_logs[self.monitor][-1]
        
        if self.mode == "max":
            if current > self.best:
                self.best = current
                self.wait = 0
            else:
                self.wait = self.wait + 1
        elif self.mode == "min":
            if current < self.best:
                self.best = current
                self.wait = 0
            else:
                self.wait = self.wait + 1
                
        if self.wait > self.patience:
            reduced_lr = optimizer.inner_optimizer.lr.read_value().numpy() * self.factor
            if reduced_lr > self.min_lr:
                optimizer.inner_optimizer.lr.assign(reduced_lr)
                print(" \n Learning rate decreased to {}".format(
                    optimizer.inner_optimizer.lr.read_value().numpy()))
            self.wait = 0
            
        return optimizer
