import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, UpSampling2D, Lambda, Conv2DTranspose
from tensorflow.keras.models import Sequential


BN_MOMENTUM = 0.1


def conv3x3(out_planes, stride=1):
    """3 x 3 convolution with padding"""
    return Conv2D(filters=out_planes, kernel_size=3, strides=stride, padding="same", use_bias=False)


class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, _in_channel, output_dim, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.downsample = downsample
        self.conv1 = conv3x3(output_dim, stride)
        self.bn1 = BatchNormalization(momentum=BN_MOMENTUM)
        self.relu = ReLU()
        self.conv2 = conv3x3(output_dim)
        self.bn2 = BatchNormalization(momentum=BN_MOMENTUM)

    def call(self, inputs, training=None, mask=None):
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, _in_channel, output_dim, stride=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.downsample = downsample
        self.conv1 = Conv2D(filters=output_dim, kernel_size=1, padding="same", use_bias=False)
        self.bn1 = BatchNormalization(momentum=BN_MOMENTUM)
        self.conv2 = Conv2D(filters=output_dim, kernel_size=3, strides=stride, padding="same", use_bias=False)
        self.bn2 = BatchNormalization(momentum=BN_MOMENTUM)
        self.conv3 = Conv2D(filters=output_dim*self.expansion, kernel_size=1, padding="same", use_bias=False)
        self.bn3 = BatchNormalization(momentum=BN_MOMENTUM)
        self.relu = ReLU()

    def call(self, inputs, training=None, mask=None):
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(tf.keras.layers.Layer):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels,
                 multi_scale_output=True, **kwargs):
        super(HighResolutionModule, self).__init__(**kwargs)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.blocks = blocks
        self.num_blocks = num_blocks
        self.num_channels = num_channels

        self.branches = self._make_branches(self.num_branches, self.blocks, self.num_blocks, self.num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = ReLU()

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = Sequential(
                Conv2D(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, strides=stride, bias=False
                ),
                BatchNormalization(momentum=BN_MOMENTUM),
            )
            
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        
        return Sequential(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return branches

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(Sequential([
                        Conv2D(num_inchannels[i], kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False),
                        BatchNormalization(momentum=BN_MOMENTUM),
                        UpSampling2D(size=(2 ** (j - i), 2 ** (j - i)), interpolation='bilinear')
                    ]))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(Sequential([
                                Conv2D(num_outchannels_conv3x3,kernel_size=(3,3),strides=(2,2),padding="same",use_bias=False),
                                BatchNormalization(momentum=BN_MOMENTUM)
                            ]))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(Sequential([
                                Conv2D(num_outchannels_conv3x3,kernel_size=(3,3),strides=(2,2),padding="same",use_bias=False),
                                BatchNormalization(momentum=BN_MOMENTUM),
                                ReLU()
                            ]))
                    fuse_layer.append(Sequential(conv3x3s))
            fuse_layers.append(fuse_layer)

        return fuse_layers

    def get_num_inchannels(self):
        return self.num_inchannels

    def call(self, inputs, training=None, mask=None):
        if self.num_branches == 1:
            return [self.branches[0](inputs[0])]

        for i in range(self.num_branches):
            inputs[i] = self.branches[i](inputs[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = inputs[0] if i == 0 else self.fuse_layers[i][0](inputs[0])
            # We cast to tf.float32 to ensure there is no mismatch between data types
            # when using automatic mixed precision
            y = tf.cast(y, tf.float32)
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + tf.cast(inputs[j], tf.float32)
                elif j > 1:
                    xx = self.fuse_layers[i][j](inputs[j])
                    xx = tf.cast(xx, tf.float32)
                    width_output = inputs[i].get_shape()[-2]
                    height_output = inputs[i].get_shape()[-3]
                    y = y + tf.image.resize(xx, size=(height_output, width_output))
                else:
                    xx = self.fuse_layers[i][j](inputs[j])
                    xx = tf.cast(xx, tf.float32)
                    y = y + xx
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {
    'BASIC'     : BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRNet(tf.keras.models.Model):
    def __init__(self, 
                 stage1_cfg, 
                 stage2_cfg, 
                 stage3_cfg, 
                 stage4_cfg, 
                 input_height, 
                 input_width, 
                 n_classes, 
                 W,
                 ACCUM_STEPS,
                 conv_upsample=False):
        super(HRNet, self).__init__()

        C, C2, C4, C8 = W, int(W*2), int(W*4), int(W*8)
        
        stage1_cfg['NUM_CHANNELS'] = [C]
        stage2_cfg['NUM_CHANNELS'] = [C, C2]
        stage3_cfg['NUM_CHANNELS'] = [C, C2, C4]
        stage4_cfg['NUM_CHANNELS'] = [C, C2, C4, C8]
        
        self.stage1_cfg = stage1_cfg
        self.stage2_cfg = stage2_cfg
        self.stage3_cfg = stage3_cfg
        self.stage4_cfg = stage4_cfg
        self.NUM_CLASSES = n_classes
        self.ACCUM_STEPS = ACCUM_STEPS
        self.inplanes = 64
        self.input_height = input_height
        self.input_width = input_width
        self.W = W

        # stem net
        self.conv1 = Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn1 = BatchNormalization(momentum=BN_MOMENTUM)
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn2 = BatchNormalization(momentum=BN_MOMENTUM)
        self.relu = ReLU()

        # STAGE 1
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # STAGE 2
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        # STAGE 3
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        # STAGE 4
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels)

        last_inp_channels = np.int(np.sum(pre_stage_channels))
        
        # Last layer
        if conv_upsample:
            self.last_layer = Sequential([
                Conv2D(
                    filters=last_inp_channels, 
                    kernel_size=1, 
                    strides=1, 
                    padding="same", 
                    use_bias=False),
                BatchNormalization(momentum=BN_MOMENTUM),
                ReLU(),
                Conv2DTranspose(
                    filters=self.W, 
                    kernel_size=1, 
                    strides=4, 
                    padding="same", 
                    use_bias=False),
                Conv2D(filters=self.NUM_CLASSES, 
                    kernel_size=1, 
                    strides=1, 
                    padding="same", 
                    use_bias=False, 
                    dtype="float32")
            ])
        else:
            self.last_layer = Sequential([
                Conv2D(
                    filters=last_inp_channels, 
                    kernel_size=1, 
                    strides=1, 
                    padding="same", 
                    use_bias=False),
                BatchNormalization(momentum=BN_MOMENTUM),
                ReLU(),
                Conv2D(
                    filters=self.NUM_CLASSES, 
                    kernel_size=1, 
                    strides=1, 
                    padding="same", 
                    dtype="float32"),
                Lambda(lambda x: tf.image.resize(x, size=(input_height, input_width), method='bilinear')),
            ])
        
        self.build_model()
        

    @staticmethod
    def _make_transition_layer(num_channels_pre_layer, num_channels_cur_layer):
        num_branches_pre = len(num_channels_pre_layer)
        num_branches_cur = len(num_channels_cur_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(Sequential([
                        Conv2D(
                            filters=num_channels_cur_layer[i],
                            kernel_size=(3,3),
                            strides=(1,1),
                            padding="same",
                            use_bias=False),
                        BatchNormalization(momentum=BN_MOMENTUM),
                        ReLU()
                    ]))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(Sequential([
                        Conv2D(
                            filters=outchannels, 
                            kernel_size=(3,3), 
                            strides=(2,2), 
                            padding="same", 
                            use_bias=False),
                        BatchNormalization(),
                        ReLU()
                    ]))
                transition_layers.append(Sequential(conv3x3s))

        return transition_layers

    @staticmethod
    def _make_stage(layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_channels = layer_config['NUM_CHANNELS']
        num_blocks = layer_config['NUM_BLOCKS']
        block = blocks_dict[layer_config['BLOCK']]
        
        modules = []
        for i in range(num_modules):

            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches=num_branches,
                    blocks=block,
                    num_blocks=num_blocks,
                    num_inchannels=num_inchannels,
                    num_channels=num_channels,
                    multi_scale_output=reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return modules, num_inchannels

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential([
                Conv2D(
                    planes * block.expansion, 
                    kernel_size=(1,1), 
                    strides=(stride,stride), 
                    padding="same",use_bias=False
                ),
                BatchNormalization(momentum=BN_MOMENTUM)
            ])

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(layers)

    @staticmethod
    def _forward_stage(stage, xs):
        ys = xs
        for module in stage:
            ys = module(ys)
            if not isinstance(ys, list):
                ys = [ys]
        return ys

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # STAGE 1
        x = self.layer1(x)

        # STAGE 2
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self._forward_stage(self.stage2, x_list)

        # STAGE 3
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self._forward_stage(self.stage3, x_list)

        # STAGE 4
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self._forward_stage(self.stage4, x_list)

        # Upsampling
        x0_h, x0_w = x[0].get_shape()[1], x[0].get_shape()[2]
        x1 = tf.image.resize(x[1], size=(x0_h, x0_w))
        x2 = tf.image.resize(x[2], size=(x0_h, x0_w))
        x3 = tf.image.resize(x[3], size=(x0_h, x0_w))

        # Concat
        x0 = tf.cast(x[0], tf.float32)
        x = tf.concat([x0, x1, x2, x3], axis=3)

        x = self.last_layer(x)
        
        x = tf.cast(x, tf.float32)

        return x
        
        
    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # scaled_loss = self.optimizer.get_scaled_loss(loss)
            
        # Calculate batch gradients
        # scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        # gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        gradients = tape.gradient(loss, self.trainable_variables)
            
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
 
        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables 
        # otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    
    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))
            
            
    def build_model(self):
            
        # Initialize weights of the network
        inp_test = tf.random.normal(shape=(1, self.input_height, self.input_width, 3))
        out_test = self(inp_test)

        self._name = "HRNet_W{}".format(self.W)
        
         # Gradient accumilation
        self.n_gradients = tf.constant(self.ACCUM_STEPS, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), 
                                                  trainable=False) for v in self.trainable_variables]
