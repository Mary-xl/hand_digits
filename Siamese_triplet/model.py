
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Layer, Lambda

def build_basemodel(input_shape, embeddingsize):

    input_img = Input(shape=input_shape)
    x=Conv2D(128, (7,7),activation='relu',kernel_initializer='he_uniform', name='conv_1')(input_img)
    x=MaxPooling2D((2,2))(x)
    x=Conv2D(128,(3,3), activation='relu',kernel_initializer='he_uniform')(x)
    x=MaxPooling2D((2,2))(x)
    x=Conv2D(256,(3,3),activation='relu',kernel_initializer='he_uniform')(x)
    x=Flatten()(x)
    x=Dense(4096, activation='relu',kernel_initializer='he_uniform')(x)
    x=Dense(embeddingsize,activation=None,kernel_initializer='he_uniform')(x)
    #Lambda(lambda x: K.l2_normalize(x, axis=-1))
    l2_norm_output = Lambda(lambda x: K.l2_normalize(x, axis=1), name='embedding_layer')(x)
    base=Model(inputs=input_img, outputs=l2_norm_output)
    return base

# triplet loss: L=max(d(A,P)−d(A,N)+margin,0)
class TripletLossLayer(Layer):
    def __init__(self, **kwargs):
        #self.alpha=alpha
        super(TripletLossLayer,self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, postive, negative=inputs
        p_dist=K.sum(K.square(anchor-postive), axis=-1)
        n_dist=K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist-n_dist+0.2, 0), axis=0)

    def call(self,inputs):
        loss=self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def build_model(input_shape, base, margin=0.2):

    anchor_input=base.input
    positive_input=Input(input_shape,name='positive_input')
    negative_input=Input(input_shape,name='negative_input')

    feature_a=base(anchor_input)
    feature_p=base(positive_input)
    feature_n=base(negative_input)

    loss_layer=TripletLossLayer(name='TripletLossLayer')([feature_a,feature_p,feature_n])

    triplet_model=Model(inputs=[anchor_input,positive_input,negative_input], outputs=loss_layer)

    return triplet_model