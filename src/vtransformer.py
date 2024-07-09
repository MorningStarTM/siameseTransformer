import tensorflow as tf
from keras import layers
from keras.models import Model
import numpy as np


hp = {}
hp['image_size'] = 256
hp['num_channels'] = 3
hp['patch_size'] = 32
hp['num_patches'] = (hp['image_size']**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp['patch_size']*hp['patch_size']*hp["num_channels"])
hp['batch_size'] = 2
hp['lr'] = 1e-5
hp["num_epochs"] = 30
hp['num_classes'] = 3
hp["num_layers"] = 3
hp["hidden_dim"] = 256
hp["mlp_dim"] = 256
hp['num_heads'] = 3
hp['dropout_rate'] = 0.1
hp['class_names'] = ["s1", "s2", "s3"]



class ClassToken(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        #initial values for the weight
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32), 
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        #reshape
        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        #change data type
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls   
    

def mlp(x, cf):
    x = layers.Dense(cf['mlp_dim'], activation='gelu')(x)
    x = layers.Dropout(cf['dropout_rate'])(x)
    x = layers.Dense(cf['hidden_dim'])(x)
    x = layers.Dropout(cf['dropout_rate'])(x)
    return x


def transformer_encoder(x, cf):
    skip_1 = x
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(num_heads=cf['num_heads'], key_dim=cf['hidden_dim'])(x,x)
    x = layers.Add()([x, skip_1])
    
    skip_2 = x
    x = layers.LayerNormalization()(x)
    x = mlp(x, cf)
    x = layers.Add()([x, skip_2])
    
    return x



def CNN_ViT(hp):
    input_shape = (hp['image_size'], hp['image_size'], hp['num_channels'])
    inputs = layers.Input(input_shape)
    print(inputs.shape)
    output = build_resnet(inputs)
    print(output.shape)

    patch_embed = layers.Conv2D(hp['hidden_dim'], kernel_size=(hp['patch_size']), padding='same')(output)
    print(patch_embed.shape)
    _, h, w, f = output.shape
    patch_embed = layers.Reshape((h*w,f))(output)

    #Position Embedding
    positions = tf.range(start=0, limit=hp['num_patches'], delta=1)
    pos_embed = layers.Embedding(input_dim=hp['num_patches'], output_dim=hp['hidden_dim'])(positions)

    print(f"patch embedding : {patch_embed.shape}")
    print(f"position embeding : {pos_embed.shape}")
    #Patch + Position Embedding
    embed = patch_embed + pos_embed
    
    #Token
    token = ClassToken()(embed)
    x = layers.Concatenate(axis=1)([token, embed]) #(None, 257, 256)
    
    #Transformer encoder
    for _ in range(hp['num_layers']):
        x = transformer_encoder(x, hp)
    
    
    x = layers.LayerNormalization()(x)
    x = x[:, 0, :]
    x = layers.Dense(hp['num_classes'], activation='softmax')(x)