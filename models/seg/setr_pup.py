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

        self.qkv = Dense(dimension * 3, use_bias=False)
        self.rearrange_attention = Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        self.attn_dropout = Dropout(dropout_rate)
        
        self.rearrange_output = Rearrange('b h n d -> b n (h d)')
        self.proj = Dense(dimension)
        self.proj_dropout = Dropout(dropout_rate)

    def call(self, inputs):
        qkv = self.qkv(inputs)
        qkv = self.rearrange_attention(qkv)

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        dot_product = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attention = tf.nn.softmax(dot_product, axis=-1)
        attention = self.attn_dropout(attention)

        x = tf.einsum('bhij,bhjd->bhid', attention, v)
        x = self.rearrange_output(x)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


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
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(
                            FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = Sequential(layers)

    def call(self, x):
        return self.net(x)


class SETR_PUP(tf.keras.models.Model):
    
    def __init__(
        self,
        layers,
        hidden_dim,
        heads,
        img_size, 
        n_classes,
        patch_size,
        num_patches, 
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        
        super(SETR_PUP, self).__init__()
        
        self._name = "SETR_PUP"
        self.img_size = img_size
        self.mlp_dim = int(hidden_dim*4)
        
        self.patch_creator = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, hidden_dim)

        self.transformer = TransformerModel(
            dim=hidden_dim,
            depth=layers,
            heads=heads,
            mlp_dim=self.mlp_dim,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )

        self.last_layer = Sequential([
            Reshape(target_shape=(int(img_size//16), int(img_size//16), hidden_dim)),

            UpSampling2D(size=(2,2), interpolation='bilinear'),
            Conv2D(hidden_dim, kernel_size=(3,3), strides=(1,1), padding='same'),

            UpSampling2D(size=(2,2), interpolation='bilinear'),
            Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'),

            UpSampling2D(size=(2,2), interpolation='bilinear'),
            Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'),

            UpSampling2D(size=(2,2), interpolation='bilinear'),
            Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'),
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
        
            
    def build_model(self):
            
        # Initialize weights of the network
        inp_test = tf.random.normal(shape=(1, self.img_size, self.img_size, 3))
        out_test = self(inp_test)
        

        

        

        

        

        

        

        

        

        

        

        

        

