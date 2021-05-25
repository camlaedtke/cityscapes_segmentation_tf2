import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from einops.layers.tensorflow import Rearrange


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    


class SelfAttention(tf.keras.layers.Layer):

    def __init__(self, dimension, heads=8, dropout_rate=0.0):
        super(SelfAttention, self).__init__()

        self.heads = heads
        self.scale = dimension ** -0.5

        self.mlp_in = tf.keras.layers.Dense(dimension * 3, use_bias=False)
        self.mlp_out = tf.keras.layers.Dense(dimension)

        self.rearrange_attention = Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        self.rearrange_output = Rearrange('b h n d -> b n (h d)')
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs):
        qkv = self.mlp_in(inputs)
        qkv = self.rearrange_attention(qkv)

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        dot_product = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attention = tf.nn.softmax(dot_product, axis=-1)

        out = tf.einsum('bhij,bhjd->bhid', attention, v)
        out = self.rearrange_output(out)
        out = self.mlp_out(out)
        out = self.dropout(out)
        return out


class Residual(tf.keras.layers.Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def call(self, x):
        return self.fn(x) + x


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()
        self.norm = LayerNormalization(epsilon=1e-6)
        self.fn = fn

    def call(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(tf.keras.layers.Layer):
    def __init__(self, dropout_rate, fn):
        super(PreNormDrop, self).__init__()
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout_rate)
        self.fn = fn

    def call(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(tf.keras.layers.Layer):
    
    def __init__(self, dim, hidden_dim, dropout_rate):
        super(FeedForward, self).__init__()
        self.net = Sequential([
            Dense(hidden_dim),
            Activation(tf.nn.gelu),
            Dropout(dropout_rate),
            Dense(dim),
            Dropout(dropout_rate),
        ])
        
    def call(self, x):
        return self.net(x)


def _forward_stage(stage, xs):
    ys = xs

    for module in stage:
        ys = module(ys)
        if not isinstance(ys, list):
            ys = [ys]
    return ys


class TransformerModel(tf.keras.models.Model):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = Sequential(layers)
        # print(self.net.layers)

    def call(self, x):
        return self.net(x)


class setr_pup(tf.keras.models.Model):
    
    def __init__(
        self,
        img_size, 
        n_classes,
        num_patches, 
        patch_size,
        transformer_layers,
        dim,
        heads,
        mlp_dim,
        ACCUM_STEPS,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        
        super().__init__()
        
        self._name = "SETR_PUP"
        self.img_size = img_size
        self.ACCUM_STEPS = ACCUM_STEPS
        
        self.patch_creator = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, dim)

        self.transformer = TransformerModel(
            dim=dim,
            depth=transformer_layers,
            heads=heads,
            mlp_dim=dim,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )


        self.last_layer = Sequential([
            Reshape(target_shape=(int(img_size//16), int(img_size//16), dim)),

            UpSampling2D(size=(2,2), interpolation='bilinear'),
            Conv2D(dim, kernel_size=(1,1), strides=(1,1), padding='same'),

            UpSampling2D(size=(2,2), interpolation='bilinear'),
            Conv2D(256, kernel_size=(1,1), strides=(1,1), padding='same'),

            UpSampling2D(size=(2,2), interpolation='bilinear'),
            Conv2D(256, kernel_size=(1,1), strides=(1,1), padding='same'),

            UpSampling2D(size=(2,2), interpolation='bilinear'),
            Conv2D(256, kernel_size=(1,1), strides=(1,1), padding='same'),
            Conv2D(n_classes, kernel_size=(1,1), strides=(1,1), padding='same', dtype='float32'),
        ])
        
        self.build_model()
    
    
    def call(self, inputs):
        
        x = self.patch_creator(inputs)
        
        x = self.patch_encoder(x)
        
        x = self.transformer(x)
        
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
            scaled_loss = self.optimizer.get_scaled_loss(loss)
            
        # Calculate batch gradients
        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        # gradients = tape.gradient(loss, self.trainable_variables)
            
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
        inp_test = tf.random.normal(shape=(1, self.img_size, self.img_size, 3))
        out_test = self(inp_test)
        
         # Gradient accumilation
        self.n_gradients = tf.constant(self.ACCUM_STEPS, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), 
                                                  trainable=False) for v in self.trainable_variables]

        

        

        

        

        

        

        

        

        

        

        

        

