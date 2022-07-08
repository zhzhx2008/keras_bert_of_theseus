import os
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model, BERT
from bert4keras.optimizers import AdaFactor
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.models import build_transformer_model

config_path = '/data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/data0/nfs_data/zhaoxi9/pretrained_language_model/Chinese-BERT-wwm/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

expert_model = build_transformer_model(
    config_path, checkpoint_path, prefix='Expert-'
)

class LST_BERT(BERT):
    """定义新的BERT类
    """
    def __init__(self, expert_model=None, **kwargs):
        super(LST_BERT, self).__init__(**kwargs)
        self.expert_model = expert_model
        self.expert_layers = 0
        while True:
            try:
                n = 'Expert-Transformer-%d-FeedForward-Norm' % self.expert_layers
                expert_model.get_layer(n)
                self.expert_layers += 1
            except:
                break
        self.skip = self.expert_layers // self.num_hidden_layers

    def get_inputs(self):
        return self.expert_model.inputs

    def apply_embeddings(self, inputs):
        x = self.expert_model.get_layer('Expert-Embedding-Dropout').output
        x = keras.layers.Dense(self.hidden_size, use_bias=False)(x)
        return x

    def apply_main_layers(self, inputs, index):
        x = super(LST_BERT, self).apply_main_layers(inputs, index)
        n = (index + 1) * self.skip - 1
        n = 'Expert-Transformer-%d-FeedForward-Norm' % n
        y = self.expert_model.get_layer(n).output
        y = keras.layers.Dense(self.hidden_size, use_bias=False)(y)
        return keras.layers.Add()([x, y])

    def initializer(self, shape, dtype=None, order=2, gain=1.0):
        """使用DeepNorm的思想初始化
        """
        if shape[0] > 10000 or shape[0] < 10:
            hidden_size = shape[1]
        else:
            hidden_size = shape[0]
        gain *= self.num_hidden_layers**(-1. / order)
        stddev = 1.13684723 / hidden_size**0.5 * gain
        return K.truncated_normal(shape, stddev=stddev)


for layer in expert_model.layers:
    layer.trainable = False
expert_model = keras.models.Model(expert_model.inputs, expert_model.outputs)
base = build_transformer_model(
    config_path,
    model=LST_BERT,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    expert_model=expert_model,
    return_keras_model=False
)
print(base.model.summary())

from keras.utils.vis_utils import plot_model
plot_model(base.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Model: "model_3"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# Expert-Input-Token (InputLayer) (None, None)         0
# __________________________________________________________________________________________________
# Expert-Input-Segment (InputLaye (None, None)         0
# __________________________________________________________________________________________________
# Expert-Embedding-Token (Embeddi (None, None, 768)    16226304    Expert-Input-Token[0][0]
# __________________________________________________________________________________________________
# Expert-Embedding-Segment (Embed (None, None, 768)    1536        Expert-Input-Segment[0][0]
# __________________________________________________________________________________________________
# Expert-Embedding-Token-Segment  (None, None, 768)    0           Expert-Embedding-Token[0][0]
#                                                                  Expert-Embedding-Segment[0][0]
# __________________________________________________________________________________________________
# Expert-Embedding-Position (Posi (None, None, 768)    393216      Expert-Embedding-Token-Segment[0]
# __________________________________________________________________________________________________
# Expert-Embedding-Norm (LayerNor (None, None, 768)    1536        Expert-Embedding-Position[0][0]
# __________________________________________________________________________________________________
# Expert-Embedding-Dropout (Dropo (None, None, 768)    0           Expert-Embedding-Norm[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-0-MultiHeadS (None, None, 768)    2362368     Expert-Embedding-Dropout[0][0]
#                                                                  Expert-Embedding-Dropout[0][0]
#                                                                  Expert-Embedding-Dropout[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-0-MultiHeadS (None, None, 768)    0           Expert-Transformer-0-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-0-MultiHeadS (None, None, 768)    0           Expert-Embedding-Dropout[0][0]
#                                                                  Expert-Transformer-0-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-0-MultiHeadS (None, None, 768)    1536        Expert-Transformer-0-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-0-FeedForwar (None, None, 768)    4722432     Expert-Transformer-0-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-0-FeedForwar (None, None, 768)    0           Expert-Transformer-0-FeedForward[
# __________________________________________________________________________________________________
# Expert-Transformer-0-FeedForwar (None, None, 768)    0           Expert-Transformer-0-MultiHeadSel
#                                                                  Expert-Transformer-0-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-0-FeedForwar (None, None, 768)    1536        Expert-Transformer-0-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-1-MultiHeadS (None, None, 768)    2362368     Expert-Transformer-0-FeedForward-
#                                                                  Expert-Transformer-0-FeedForward-
#                                                                  Expert-Transformer-0-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-1-MultiHeadS (None, None, 768)    0           Expert-Transformer-1-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-1-MultiHeadS (None, None, 768)    0           Expert-Transformer-0-FeedForward-
#                                                                  Expert-Transformer-1-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-1-MultiHeadS (None, None, 768)    1536        Expert-Transformer-1-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-1-FeedForwar (None, None, 768)    4722432     Expert-Transformer-1-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-1-FeedForwar (None, None, 768)    0           Expert-Transformer-1-FeedForward[
# __________________________________________________________________________________________________
# Expert-Transformer-1-FeedForwar (None, None, 768)    0           Expert-Transformer-1-MultiHeadSel
#                                                                  Expert-Transformer-1-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-1-FeedForwar (None, None, 768)    1536        Expert-Transformer-1-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-2-MultiHeadS (None, None, 768)    2362368     Expert-Transformer-1-FeedForward-
#                                                                  Expert-Transformer-1-FeedForward-
#                                                                  Expert-Transformer-1-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-2-MultiHeadS (None, None, 768)    0           Expert-Transformer-2-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-2-MultiHeadS (None, None, 768)    0           Expert-Transformer-1-FeedForward-
#                                                                  Expert-Transformer-2-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-2-MultiHeadS (None, None, 768)    1536        Expert-Transformer-2-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-2-FeedForwar (None, None, 768)    4722432     Expert-Transformer-2-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-2-FeedForwar (None, None, 768)    0           Expert-Transformer-2-FeedForward[
# __________________________________________________________________________________________________
# Expert-Transformer-2-FeedForwar (None, None, 768)    0           Expert-Transformer-2-MultiHeadSel
#                                                                  Expert-Transformer-2-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-2-FeedForwar (None, None, 768)    1536        Expert-Transformer-2-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-3-MultiHeadS (None, None, 768)    2362368     Expert-Transformer-2-FeedForward-
#                                                                  Expert-Transformer-2-FeedForward-
#                                                                  Expert-Transformer-2-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-3-MultiHeadS (None, None, 768)    0           Expert-Transformer-3-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-3-MultiHeadS (None, None, 768)    0           Expert-Transformer-2-FeedForward-
#                                                                  Expert-Transformer-3-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-3-MultiHeadS (None, None, 768)    1536        Expert-Transformer-3-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-3-FeedForwar (None, None, 768)    4722432     Expert-Transformer-3-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-3-FeedForwar (None, None, 768)    0           Expert-Transformer-3-FeedForward[
# __________________________________________________________________________________________________
# Expert-Transformer-3-FeedForwar (None, None, 768)    0           Expert-Transformer-3-MultiHeadSel
#                                                                  Expert-Transformer-3-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-3-FeedForwar (None, None, 768)    1536        Expert-Transformer-3-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-4-MultiHeadS (None, None, 768)    2362368     Expert-Transformer-3-FeedForward-
#                                                                  Expert-Transformer-3-FeedForward-
#                                                                  Expert-Transformer-3-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-4-MultiHeadS (None, None, 768)    0           Expert-Transformer-4-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-4-MultiHeadS (None, None, 768)    0           Expert-Transformer-3-FeedForward-
#                                                                  Expert-Transformer-4-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-4-MultiHeadS (None, None, 768)    1536        Expert-Transformer-4-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-4-FeedForwar (None, None, 768)    4722432     Expert-Transformer-4-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-4-FeedForwar (None, None, 768)    0           Expert-Transformer-4-FeedForward[
# __________________________________________________________________________________________________
# Expert-Transformer-4-FeedForwar (None, None, 768)    0           Expert-Transformer-4-MultiHeadSel
#                                                                  Expert-Transformer-4-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-4-FeedForwar (None, None, 768)    1536        Expert-Transformer-4-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-5-MultiHeadS (None, None, 768)    2362368     Expert-Transformer-4-FeedForward-
#                                                                  Expert-Transformer-4-FeedForward-
#                                                                  Expert-Transformer-4-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-5-MultiHeadS (None, None, 768)    0           Expert-Transformer-5-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-5-MultiHeadS (None, None, 768)    0           Expert-Transformer-4-FeedForward-
#                                                                  Expert-Transformer-5-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-5-MultiHeadS (None, None, 768)    1536        Expert-Transformer-5-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-5-FeedForwar (None, None, 768)    4722432     Expert-Transformer-5-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-5-FeedForwar (None, None, 768)    0           Expert-Transformer-5-FeedForward[
# __________________________________________________________________________________________________
# Expert-Transformer-5-FeedForwar (None, None, 768)    0           Expert-Transformer-5-MultiHeadSel
#                                                                  Expert-Transformer-5-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-5-FeedForwar (None, None, 768)    1536        Expert-Transformer-5-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-6-MultiHeadS (None, None, 768)    2362368     Expert-Transformer-5-FeedForward-
#                                                                  Expert-Transformer-5-FeedForward-
#                                                                  Expert-Transformer-5-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-6-MultiHeadS (None, None, 768)    0           Expert-Transformer-6-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-6-MultiHeadS (None, None, 768)    0           Expert-Transformer-5-FeedForward-
#                                                                  Expert-Transformer-6-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-6-MultiHeadS (None, None, 768)    1536        Expert-Transformer-6-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-6-FeedForwar (None, None, 768)    4722432     Expert-Transformer-6-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-6-FeedForwar (None, None, 768)    0           Expert-Transformer-6-FeedForward[
# __________________________________________________________________________________________________
# Expert-Transformer-6-FeedForwar (None, None, 768)    0           Expert-Transformer-6-MultiHeadSel
#                                                                  Expert-Transformer-6-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-6-FeedForwar (None, None, 768)    1536        Expert-Transformer-6-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-7-MultiHeadS (None, None, 768)    2362368     Expert-Transformer-6-FeedForward-
#                                                                  Expert-Transformer-6-FeedForward-
#                                                                  Expert-Transformer-6-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-7-MultiHeadS (None, None, 768)    0           Expert-Transformer-7-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-7-MultiHeadS (None, None, 768)    0           Expert-Transformer-6-FeedForward-
#                                                                  Expert-Transformer-7-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-7-MultiHeadS (None, None, 768)    1536        Expert-Transformer-7-MultiHeadSel
# __________________________________________________________________________________________________
# Expert-Transformer-7-FeedForwar (None, None, 768)    4722432     Expert-Transformer-7-MultiHeadSel
# __________________________________________________________________________________________________
# dense_73 (Dense)                (None, None, 128)    98304       Expert-Embedding-Dropout[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-7-FeedForwar (None, None, 768)    0           Expert-Transformer-7-FeedForward[
# __________________________________________________________________________________________________
# Transformer-0-MultiHeadSelfAtte (None, None, 128)    66048       dense_73[0][0]
#                                                                  dense_73[0][0]
#                                                                  dense_73[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-7-FeedForwar (None, None, 768)    0           Expert-Transformer-7-MultiHeadSel
#                                                                  Expert-Transformer-7-FeedForward-
# __________________________________________________________________________________________________
# Transformer-0-MultiHeadSelfAtte (None, None, 128)    0           Transformer-0-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-7-FeedForwar (None, None, 768)    1536        Expert-Transformer-7-FeedForward-
# __________________________________________________________________________________________________
# Transformer-0-MultiHeadSelfAtte (None, None, 128)    0           dense_73[0][0]
#                                                                  Transformer-0-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-8-MultiHeadS (None, None, 768)    2362368     Expert-Transformer-7-FeedForward-
#                                                                  Expert-Transformer-7-FeedForward-
#                                                                  Expert-Transformer-7-FeedForward-
# __________________________________________________________________________________________________
# Transformer-0-MultiHeadSelfAtte (None, None, 128)    256         Transformer-0-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-8-MultiHeadS (None, None, 768)    0           Expert-Transformer-8-MultiHeadSel
# __________________________________________________________________________________________________
# Transformer-0-FeedForward (Feed (None, None, 128)    131712      Transformer-0-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-8-MultiHeadS (None, None, 768)    0           Expert-Transformer-7-FeedForward-
#                                                                  Expert-Transformer-8-MultiHeadSel
# __________________________________________________________________________________________________
# Transformer-0-FeedForward-Dropo (None, None, 128)    0           Transformer-0-FeedForward[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-8-MultiHeadS (None, None, 768)    1536        Expert-Transformer-8-MultiHeadSel
# __________________________________________________________________________________________________
# Transformer-0-FeedForward-Add ( (None, None, 128)    0           Transformer-0-MultiHeadSelfAttent
#                                                                  Transformer-0-FeedForward-Dropout
# __________________________________________________________________________________________________
# Expert-Transformer-8-FeedForwar (None, None, 768)    4722432     Expert-Transformer-8-MultiHeadSel
# __________________________________________________________________________________________________
# Transformer-0-FeedForward-Norm  (None, None, 128)    256         Transformer-0-FeedForward-Add[0][
# __________________________________________________________________________________________________
# dense_80 (Dense)                (None, None, 128)    98304       Expert-Transformer-2-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-8-FeedForwar (None, None, 768)    0           Expert-Transformer-8-FeedForward[
# __________________________________________________________________________________________________
# add_1 (Add)                     (None, None, 128)    0           Transformer-0-FeedForward-Norm[0]
#                                                                  dense_80[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-8-FeedForwar (None, None, 768)    0           Expert-Transformer-8-MultiHeadSel
#                                                                  Expert-Transformer-8-FeedForward-
# __________________________________________________________________________________________________
# Transformer-1-MultiHeadSelfAtte (None, None, 128)    66048       add_1[0][0]
#                                                                  add_1[0][0]
#                                                                  add_1[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-8-FeedForwar (None, None, 768)    1536        Expert-Transformer-8-FeedForward-
# __________________________________________________________________________________________________
# Transformer-1-MultiHeadSelfAtte (None, None, 128)    0           Transformer-1-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-9-MultiHeadS (None, None, 768)    2362368     Expert-Transformer-8-FeedForward-
#                                                                  Expert-Transformer-8-FeedForward-
#                                                                  Expert-Transformer-8-FeedForward-
# __________________________________________________________________________________________________
# Transformer-1-MultiHeadSelfAtte (None, None, 128)    0           add_1[0][0]
#                                                                  Transformer-1-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-9-MultiHeadS (None, None, 768)    0           Expert-Transformer-9-MultiHeadSel
# __________________________________________________________________________________________________
# Transformer-1-MultiHeadSelfAtte (None, None, 128)    256         Transformer-1-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-9-MultiHeadS (None, None, 768)    0           Expert-Transformer-8-FeedForward-
#                                                                  Expert-Transformer-9-MultiHeadSel
# __________________________________________________________________________________________________
# Transformer-1-FeedForward (Feed (None, None, 128)    131712      Transformer-1-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-9-MultiHeadS (None, None, 768)    1536        Expert-Transformer-9-MultiHeadSel
# __________________________________________________________________________________________________
# Transformer-1-FeedForward-Dropo (None, None, 128)    0           Transformer-1-FeedForward[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-9-FeedForwar (None, None, 768)    4722432     Expert-Transformer-9-MultiHeadSel
# __________________________________________________________________________________________________
# Transformer-1-FeedForward-Add ( (None, None, 128)    0           Transformer-1-MultiHeadSelfAttent
#                                                                  Transformer-1-FeedForward-Dropout
# __________________________________________________________________________________________________
# Expert-Transformer-9-FeedForwar (None, None, 768)    0           Expert-Transformer-9-FeedForward[
# __________________________________________________________________________________________________
# Transformer-1-FeedForward-Norm  (None, None, 128)    256         Transformer-1-FeedForward-Add[0][
# __________________________________________________________________________________________________
# dense_87 (Dense)                (None, None, 128)    98304       Expert-Transformer-5-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-9-FeedForwar (None, None, 768)    0           Expert-Transformer-9-MultiHeadSel
#                                                                  Expert-Transformer-9-FeedForward-
# __________________________________________________________________________________________________
# add_2 (Add)                     (None, None, 128)    0           Transformer-1-FeedForward-Norm[0]
#                                                                  dense_87[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-9-FeedForwar (None, None, 768)    1536        Expert-Transformer-9-FeedForward-
# __________________________________________________________________________________________________
# Transformer-2-MultiHeadSelfAtte (None, None, 128)    66048       add_2[0][0]
#                                                                  add_2[0][0]
#                                                                  add_2[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-10-MultiHead (None, None, 768)    2362368     Expert-Transformer-9-FeedForward-
#                                                                  Expert-Transformer-9-FeedForward-
#                                                                  Expert-Transformer-9-FeedForward-
# __________________________________________________________________________________________________
# Transformer-2-MultiHeadSelfAtte (None, None, 128)    0           Transformer-2-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-10-MultiHead (None, None, 768)    0           Expert-Transformer-10-MultiHeadSe
# __________________________________________________________________________________________________
# Transformer-2-MultiHeadSelfAtte (None, None, 128)    0           add_2[0][0]
#                                                                  Transformer-2-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-10-MultiHead (None, None, 768)    0           Expert-Transformer-9-FeedForward-
#                                                                  Expert-Transformer-10-MultiHeadSe
# __________________________________________________________________________________________________
# Transformer-2-MultiHeadSelfAtte (None, None, 128)    256         Transformer-2-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-10-MultiHead (None, None, 768)    1536        Expert-Transformer-10-MultiHeadSe
# __________________________________________________________________________________________________
# Transformer-2-FeedForward (Feed (None, None, 128)    131712      Transformer-2-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-10-FeedForwa (None, None, 768)    4722432     Expert-Transformer-10-MultiHeadSe
# __________________________________________________________________________________________________
# Transformer-2-FeedForward-Dropo (None, None, 128)    0           Transformer-2-FeedForward[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-10-FeedForwa (None, None, 768)    0           Expert-Transformer-10-FeedForward
# __________________________________________________________________________________________________
# Transformer-2-FeedForward-Add ( (None, None, 128)    0           Transformer-2-MultiHeadSelfAttent
#                                                                  Transformer-2-FeedForward-Dropout
# __________________________________________________________________________________________________
# Expert-Transformer-10-FeedForwa (None, None, 768)    0           Expert-Transformer-10-MultiHeadSe
#                                                                  Expert-Transformer-10-FeedForward
# __________________________________________________________________________________________________
# Transformer-2-FeedForward-Norm  (None, None, 128)    256         Transformer-2-FeedForward-Add[0][
# __________________________________________________________________________________________________
# dense_94 (Dense)                (None, None, 128)    98304       Expert-Transformer-8-FeedForward-
# __________________________________________________________________________________________________
# Expert-Transformer-10-FeedForwa (None, None, 768)    1536        Expert-Transformer-10-FeedForward
# __________________________________________________________________________________________________
# add_3 (Add)                     (None, None, 128)    0           Transformer-2-FeedForward-Norm[0]
#                                                                  dense_94[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-11-MultiHead (None, None, 768)    2362368     Expert-Transformer-10-FeedForward
#                                                                  Expert-Transformer-10-FeedForward
#                                                                  Expert-Transformer-10-FeedForward
# __________________________________________________________________________________________________
# Transformer-3-MultiHeadSelfAtte (None, None, 128)    66048       add_3[0][0]
#                                                                  add_3[0][0]
#                                                                  add_3[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-11-MultiHead (None, None, 768)    0           Expert-Transformer-11-MultiHeadSe
# __________________________________________________________________________________________________
# Transformer-3-MultiHeadSelfAtte (None, None, 128)    0           Transformer-3-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-11-MultiHead (None, None, 768)    0           Expert-Transformer-10-FeedForward
#                                                                  Expert-Transformer-11-MultiHeadSe
# __________________________________________________________________________________________________
# Transformer-3-MultiHeadSelfAtte (None, None, 128)    0           add_3[0][0]
#                                                                  Transformer-3-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-11-MultiHead (None, None, 768)    1536        Expert-Transformer-11-MultiHeadSe
# __________________________________________________________________________________________________
# Transformer-3-MultiHeadSelfAtte (None, None, 128)    256         Transformer-3-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-11-FeedForwa (None, None, 768)    4722432     Expert-Transformer-11-MultiHeadSe
# __________________________________________________________________________________________________
# Transformer-3-FeedForward (Feed (None, None, 128)    131712      Transformer-3-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Expert-Transformer-11-FeedForwa (None, None, 768)    0           Expert-Transformer-11-FeedForward
# __________________________________________________________________________________________________
# Transformer-3-FeedForward-Dropo (None, None, 128)    0           Transformer-3-FeedForward[0][0]
# __________________________________________________________________________________________________
# Expert-Transformer-11-FeedForwa (None, None, 768)    0           Expert-Transformer-11-MultiHeadSe
#                                                                  Expert-Transformer-11-FeedForward
# __________________________________________________________________________________________________
# Transformer-3-FeedForward-Add ( (None, None, 128)    0           Transformer-3-MultiHeadSelfAttent
#                                                                  Transformer-3-FeedForward-Dropout
# __________________________________________________________________________________________________
# Expert-Transformer-11-FeedForwa (None, None, 768)    1536        Expert-Transformer-11-FeedForward
# __________________________________________________________________________________________________
# Transformer-3-FeedForward-Norm  (None, None, 128)    256         Transformer-3-FeedForward-Add[0][
# __________________________________________________________________________________________________
# dense_101 (Dense)               (None, None, 128)    98304       Expert-Transformer-11-FeedForward
# __________________________________________________________________________________________________
# add_4 (Add)                     (None, None, 128)    0           Transformer-3-FeedForward-Norm[0]
#                                                                  dense_101[0][0]
# ==================================================================================================
# Total params: 102,961,664
# Trainable params: 1,284,608
# Non-trainable params: 101,677,056
# __________________________________________________________________________________________________
# None


# for layer in base.model.layers:
#     print(layer.name)
# Expert-Input-Token
# Expert-Input-Segment
# Expert-Embedding-Token
# Expert-Embedding-Segment
# Expert-Embedding-Token-Segment
# Expert-Embedding-Position
# Expert-Embedding-Norm
# Expert-Embedding-Dropout
# Expert-Transformer-0-MultiHeadSelfAttention
# Expert-Transformer-0-MultiHeadSelfAttention-Dropout
# Expert-Transformer-0-MultiHeadSelfAttention-Add
# Expert-Transformer-0-MultiHeadSelfAttention-Norm
# Expert-Transformer-0-FeedForward
# Expert-Transformer-0-FeedForward-Dropout
# Expert-Transformer-0-FeedForward-Add
# Expert-Transformer-0-FeedForward-Norm
# Expert-Transformer-1-MultiHeadSelfAttention
# Expert-Transformer-1-MultiHeadSelfAttention-Dropout
# Expert-Transformer-1-MultiHeadSelfAttention-Add
# Expert-Transformer-1-MultiHeadSelfAttention-Norm
# Expert-Transformer-1-FeedForward
# Expert-Transformer-1-FeedForward-Dropout
# Expert-Transformer-1-FeedForward-Add
# Expert-Transformer-1-FeedForward-Norm
# Expert-Transformer-2-MultiHeadSelfAttention
# Expert-Transformer-2-MultiHeadSelfAttention-Dropout
# Expert-Transformer-2-MultiHeadSelfAttention-Add
# Expert-Transformer-2-MultiHeadSelfAttention-Norm
# Expert-Transformer-2-FeedForward
# Expert-Transformer-2-FeedForward-Dropout
# Expert-Transformer-2-FeedForward-Add
# Expert-Transformer-2-FeedForward-Norm
# Expert-Transformer-3-MultiHeadSelfAttention
# Expert-Transformer-3-MultiHeadSelfAttention-Dropout
# Expert-Transformer-3-MultiHeadSelfAttention-Add
# Expert-Transformer-3-MultiHeadSelfAttention-Norm
# Expert-Transformer-3-FeedForward
# Expert-Transformer-3-FeedForward-Dropout
# Expert-Transformer-3-FeedForward-Add
# Expert-Transformer-3-FeedForward-Norm
# Expert-Transformer-4-MultiHeadSelfAttention
# Expert-Transformer-4-MultiHeadSelfAttention-Dropout
# Expert-Transformer-4-MultiHeadSelfAttention-Add
# Expert-Transformer-4-MultiHeadSelfAttention-Norm
# Expert-Transformer-4-FeedForward
# Expert-Transformer-4-FeedForward-Dropout
# Expert-Transformer-4-FeedForward-Add
# Expert-Transformer-4-FeedForward-Norm
# Expert-Transformer-5-MultiHeadSelfAttention
# Expert-Transformer-5-MultiHeadSelfAttention-Dropout
# Expert-Transformer-5-MultiHeadSelfAttention-Add
# Expert-Transformer-5-MultiHeadSelfAttention-Norm
# Expert-Transformer-5-FeedForward
# Expert-Transformer-5-FeedForward-Dropout
# Expert-Transformer-5-FeedForward-Add
# Expert-Transformer-5-FeedForward-Norm
# Expert-Transformer-6-MultiHeadSelfAttention
# Expert-Transformer-6-MultiHeadSelfAttention-Dropout
# Expert-Transformer-6-MultiHeadSelfAttention-Add
# Expert-Transformer-6-MultiHeadSelfAttention-Norm
# Expert-Transformer-6-FeedForward
# Expert-Transformer-6-FeedForward-Dropout
# Expert-Transformer-6-FeedForward-Add
# Expert-Transformer-6-FeedForward-Norm
# Expert-Transformer-7-MultiHeadSelfAttention
# Expert-Transformer-7-MultiHeadSelfAttention-Dropout
# Expert-Transformer-7-MultiHeadSelfAttention-Add
# Expert-Transformer-7-MultiHeadSelfAttention-Norm
# Expert-Transformer-7-FeedForward
# dense_73
# Expert-Transformer-7-FeedForward-Dropout
# Transformer-0-MultiHeadSelfAttention
# Expert-Transformer-7-FeedForward-Add
# Transformer-0-MultiHeadSelfAttention-Dropout
# Expert-Transformer-7-FeedForward-Norm
# Transformer-0-MultiHeadSelfAttention-Add
# Expert-Transformer-8-MultiHeadSelfAttention
# Transformer-0-MultiHeadSelfAttention-Norm
# Expert-Transformer-8-MultiHeadSelfAttention-Dropout
# Transformer-0-FeedForward
# Expert-Transformer-8-MultiHeadSelfAttention-Add
# Transformer-0-FeedForward-Dropout
# Expert-Transformer-8-MultiHeadSelfAttention-Norm
# Transformer-0-FeedForward-Add
# Expert-Transformer-8-FeedForward
# Transformer-0-FeedForward-Norm
# dense_80
# Expert-Transformer-8-FeedForward-Dropout
# add_1
# Expert-Transformer-8-FeedForward-Add
# Transformer-1-MultiHeadSelfAttention
# Expert-Transformer-8-FeedForward-Norm
# Transformer-1-MultiHeadSelfAttention-Dropout
# Expert-Transformer-9-MultiHeadSelfAttention
# Transformer-1-MultiHeadSelfAttention-Add
# Expert-Transformer-9-MultiHeadSelfAttention-Dropout
# Transformer-1-MultiHeadSelfAttention-Norm
# Expert-Transformer-9-MultiHeadSelfAttention-Add
# Transformer-1-FeedForward
# Expert-Transformer-9-MultiHeadSelfAttention-Norm
# Transformer-1-FeedForward-Dropout
# Expert-Transformer-9-FeedForward
# Transformer-1-FeedForward-Add
# Expert-Transformer-9-FeedForward-Dropout
# Transformer-1-FeedForward-Norm
# dense_87
# Expert-Transformer-9-FeedForward-Add
# add_2
# Expert-Transformer-9-FeedForward-Norm
# Transformer-2-MultiHeadSelfAttention
# Expert-Transformer-10-MultiHeadSelfAttention
# Transformer-2-MultiHeadSelfAttention-Dropout
# Expert-Transformer-10-MultiHeadSelfAttention-Dropout
# Transformer-2-MultiHeadSelfAttention-Add
# Expert-Transformer-10-MultiHeadSelfAttention-Add
# Transformer-2-MultiHeadSelfAttention-Norm
# Expert-Transformer-10-MultiHeadSelfAttention-Norm
# Transformer-2-FeedForward
# Expert-Transformer-10-FeedForward
# Transformer-2-FeedForward-Dropout
# Expert-Transformer-10-FeedForward-Dropout
# Transformer-2-FeedForward-Add
# Expert-Transformer-10-FeedForward-Add
# Transformer-2-FeedForward-Norm
# dense_94
# Expert-Transformer-10-FeedForward-Norm
# add_3
# Expert-Transformer-11-MultiHeadSelfAttention
# Transformer-3-MultiHeadSelfAttention
# Expert-Transformer-11-MultiHeadSelfAttention-Dropout
# Transformer-3-MultiHeadSelfAttention-Dropout
# Expert-Transformer-11-MultiHeadSelfAttention-Add
# Transformer-3-MultiHeadSelfAttention-Add
# Expert-Transformer-11-MultiHeadSelfAttention-Norm
# Transformer-3-MultiHeadSelfAttention-Norm
# Expert-Transformer-11-FeedForward
# Transformer-3-FeedForward
# Expert-Transformer-11-FeedForward-Dropout
# Transformer-3-FeedForward-Dropout
# Expert-Transformer-11-FeedForward-Add
# Transformer-3-FeedForward-Add
# Expert-Transformer-11-FeedForward-Norm
# Transformer-3-FeedForward-Norm
# dense_101
# add_4


# for layer in expert_model.layers:
#     print(layer.name)
# Input-Token
# Input-Segment
# Embedding-Token
# Embedding-Segment
# Embedding-Token-Segment
# Embedding-Position
# Embedding-Norm
# Embedding-Dropout
# Transformer-0-MultiHeadSelfAttention
# Transformer-0-MultiHeadSelfAttention-Dropout
# Transformer-0-MultiHeadSelfAttention-Add
# Transformer-0-MultiHeadSelfAttention-Norm
# Transformer-0-FeedForward
# Transformer-0-FeedForward-Dropout
# Transformer-0-FeedForward-Add
# Transformer-0-FeedForward-Norm
# Transformer-1-MultiHeadSelfAttention
# Transformer-1-MultiHeadSelfAttention-Dropout
# Transformer-1-MultiHeadSelfAttention-Add
# Transformer-1-MultiHeadSelfAttention-Norm
# Transformer-1-FeedForward
# Transformer-1-FeedForward-Dropout
# Transformer-1-FeedForward-Add
# Transformer-1-FeedForward-Norm
# Transformer-2-MultiHeadSelfAttention
# Transformer-2-MultiHeadSelfAttention-Dropout
# Transformer-2-MultiHeadSelfAttention-Add
# Transformer-2-MultiHeadSelfAttention-Norm
# Transformer-2-FeedForward
# Transformer-2-FeedForward-Dropout
# Transformer-2-FeedForward-Add
# Transformer-2-FeedForward-Norm
# Transformer-3-MultiHeadSelfAttention
# Transformer-3-MultiHeadSelfAttention-Dropout
# Transformer-3-MultiHeadSelfAttention-Add
# Transformer-3-MultiHeadSelfAttention-Norm
# Transformer-3-FeedForward
# Transformer-3-FeedForward-Dropout
# Transformer-3-FeedForward-Add
# Transformer-3-FeedForward-Norm
# Transformer-4-MultiHeadSelfAttention
# Transformer-4-MultiHeadSelfAttention-Dropout
# Transformer-4-MultiHeadSelfAttention-Add
# Transformer-4-MultiHeadSelfAttention-Norm
# Transformer-4-FeedForward
# Transformer-4-FeedForward-Dropout
# Transformer-4-FeedForward-Add
# Transformer-4-FeedForward-Norm
# Transformer-5-MultiHeadSelfAttention
# Transformer-5-MultiHeadSelfAttention-Dropout
# Transformer-5-MultiHeadSelfAttention-Add
# Transformer-5-MultiHeadSelfAttention-Norm
# Transformer-5-FeedForward
# Transformer-5-FeedForward-Dropout
# Transformer-5-FeedForward-Add
# Transformer-5-FeedForward-Norm
# Transformer-6-MultiHeadSelfAttention
# Transformer-6-MultiHeadSelfAttention-Dropout
# Transformer-6-MultiHeadSelfAttention-Add
# Transformer-6-MultiHeadSelfAttention-Norm
# Transformer-6-FeedForward
# Transformer-6-FeedForward-Dropout
# Transformer-6-FeedForward-Add
# Transformer-6-FeedForward-Norm
# Transformer-7-MultiHeadSelfAttention
# Transformer-7-MultiHeadSelfAttention-Dropout
# Transformer-7-MultiHeadSelfAttention-Add
# Transformer-7-MultiHeadSelfAttention-Norm
# Transformer-7-FeedForward
# Transformer-7-FeedForward-Dropout
# Transformer-7-FeedForward-Add
# Transformer-7-FeedForward-Norm
# Transformer-8-MultiHeadSelfAttention
# Transformer-8-MultiHeadSelfAttention-Dropout
# Transformer-8-MultiHeadSelfAttention-Add
# Transformer-8-MultiHeadSelfAttention-Norm
# Transformer-8-FeedForward
# Transformer-8-FeedForward-Dropout
# Transformer-8-FeedForward-Add
# Transformer-8-FeedForward-Norm
# Transformer-9-MultiHeadSelfAttention
# Transformer-9-MultiHeadSelfAttention-Dropout
# Transformer-9-MultiHeadSelfAttention-Add
# Transformer-9-MultiHeadSelfAttention-Norm
# Transformer-9-FeedForward
# Transformer-9-FeedForward-Dropout
# Transformer-9-FeedForward-Add
# Transformer-9-FeedForward-Norm
# Transformer-10-MultiHeadSelfAttention
# Transformer-10-MultiHeadSelfAttention-Dropout
# Transformer-10-MultiHeadSelfAttention-Add
# Transformer-10-MultiHeadSelfAttention-Norm
# Transformer-10-FeedForward
# Transformer-10-FeedForward-Dropout
# Transformer-10-FeedForward-Add
# Transformer-10-FeedForward-Norm
# Transformer-11-MultiHeadSelfAttention
# Transformer-11-MultiHeadSelfAttention-Dropout
# Transformer-11-MultiHeadSelfAttention-Add
# Transformer-11-MultiHeadSelfAttention-Norm
# Transformer-11-FeedForward
# Transformer-11-FeedForward-Dropout
# Transformer-11-FeedForward-Add
# Transformer-11-FeedForward-Norm


# print(expert_model.summary())
# Model: "model_1"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# Input-Token (InputLayer)        (None, None)         0
# __________________________________________________________________________________________________
# Input-Segment (InputLayer)      (None, None)         0
# __________________________________________________________________________________________________
# Embedding-Token (Embedding)     (None, None, 768)    16226304    Input-Token[0][0]
# __________________________________________________________________________________________________
# Embedding-Segment (Embedding)   (None, None, 768)    1536        Input-Segment[0][0]
# __________________________________________________________________________________________________
# Embedding-Token-Segment (Add)   (None, None, 768)    0           Embedding-Token[0][0]
#                                                                  Embedding-Segment[0][0]
# __________________________________________________________________________________________________
# Embedding-Position (PositionEmb (None, None, 768)    393216      Embedding-Token-Segment[0][0]
# __________________________________________________________________________________________________
# Embedding-Norm (LayerNormalizat (None, None, 768)    1536        Embedding-Position[0][0]
# __________________________________________________________________________________________________
# Embedding-Dropout (Dropout)     (None, None, 768)    0           Embedding-Norm[0][0]
# __________________________________________________________________________________________________
# Transformer-0-MultiHeadSelfAtte (None, None, 768)    2362368     Embedding-Dropout[0][0]
#                                                                  Embedding-Dropout[0][0]
#                                                                  Embedding-Dropout[0][0]
# __________________________________________________________________________________________________
# Transformer-0-MultiHeadSelfAtte (None, None, 768)    0           Transformer-0-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-0-MultiHeadSelfAtte (None, None, 768)    0           Embedding-Dropout[0][0]
#                                                                  Transformer-0-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-0-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-0-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-0-FeedForward (Feed (None, None, 768)    4722432     Transformer-0-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-0-FeedForward-Dropo (None, None, 768)    0           Transformer-0-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-0-FeedForward-Add ( (None, None, 768)    0           Transformer-0-MultiHeadSelfAttent
#                                                                  Transformer-0-FeedForward-Dropout
# __________________________________________________________________________________________________
# Transformer-0-FeedForward-Norm  (None, None, 768)    1536        Transformer-0-FeedForward-Add[0][
# __________________________________________________________________________________________________
# Transformer-1-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-0-FeedForward-Norm[0]
#                                                                  Transformer-0-FeedForward-Norm[0]
#                                                                  Transformer-0-FeedForward-Norm[0]
# __________________________________________________________________________________________________
# Transformer-1-MultiHeadSelfAtte (None, None, 768)    0           Transformer-1-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-1-MultiHeadSelfAtte (None, None, 768)    0           Transformer-0-FeedForward-Norm[0]
#                                                                  Transformer-1-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-1-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-1-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-1-FeedForward (Feed (None, None, 768)    4722432     Transformer-1-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-1-FeedForward-Dropo (None, None, 768)    0           Transformer-1-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-1-FeedForward-Add ( (None, None, 768)    0           Transformer-1-MultiHeadSelfAttent
#                                                                  Transformer-1-FeedForward-Dropout
# __________________________________________________________________________________________________
# Transformer-1-FeedForward-Norm  (None, None, 768)    1536        Transformer-1-FeedForward-Add[0][
# __________________________________________________________________________________________________
# Transformer-2-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-1-FeedForward-Norm[0]
#                                                                  Transformer-1-FeedForward-Norm[0]
#                                                                  Transformer-1-FeedForward-Norm[0]
# __________________________________________________________________________________________________
# Transformer-2-MultiHeadSelfAtte (None, None, 768)    0           Transformer-2-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-2-MultiHeadSelfAtte (None, None, 768)    0           Transformer-1-FeedForward-Norm[0]
#                                                                  Transformer-2-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-2-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-2-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-2-FeedForward (Feed (None, None, 768)    4722432     Transformer-2-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-2-FeedForward-Dropo (None, None, 768)    0           Transformer-2-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-2-FeedForward-Add ( (None, None, 768)    0           Transformer-2-MultiHeadSelfAttent
#                                                                  Transformer-2-FeedForward-Dropout
# __________________________________________________________________________________________________
# Transformer-2-FeedForward-Norm  (None, None, 768)    1536        Transformer-2-FeedForward-Add[0][
# __________________________________________________________________________________________________
# Transformer-3-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-2-FeedForward-Norm[0]
#                                                                  Transformer-2-FeedForward-Norm[0]
#                                                                  Transformer-2-FeedForward-Norm[0]
# __________________________________________________________________________________________________
# Transformer-3-MultiHeadSelfAtte (None, None, 768)    0           Transformer-3-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-3-MultiHeadSelfAtte (None, None, 768)    0           Transformer-2-FeedForward-Norm[0]
#                                                                  Transformer-3-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-3-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-3-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-3-FeedForward (Feed (None, None, 768)    4722432     Transformer-3-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-3-FeedForward-Dropo (None, None, 768)    0           Transformer-3-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-3-FeedForward-Add ( (None, None, 768)    0           Transformer-3-MultiHeadSelfAttent
#                                                                  Transformer-3-FeedForward-Dropout
# __________________________________________________________________________________________________
# Transformer-3-FeedForward-Norm  (None, None, 768)    1536        Transformer-3-FeedForward-Add[0][
# __________________________________________________________________________________________________
# Transformer-4-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-3-FeedForward-Norm[0]
#                                                                  Transformer-3-FeedForward-Norm[0]
#                                                                  Transformer-3-FeedForward-Norm[0]
# __________________________________________________________________________________________________
# Transformer-4-MultiHeadSelfAtte (None, None, 768)    0           Transformer-4-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-4-MultiHeadSelfAtte (None, None, 768)    0           Transformer-3-FeedForward-Norm[0]
#                                                                  Transformer-4-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-4-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-4-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-4-FeedForward (Feed (None, None, 768)    4722432     Transformer-4-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-4-FeedForward-Dropo (None, None, 768)    0           Transformer-4-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-4-FeedForward-Add ( (None, None, 768)    0           Transformer-4-MultiHeadSelfAttent
#                                                                  Transformer-4-FeedForward-Dropout
# __________________________________________________________________________________________________
# Transformer-4-FeedForward-Norm  (None, None, 768)    1536        Transformer-4-FeedForward-Add[0][
# __________________________________________________________________________________________________
# Transformer-5-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-4-FeedForward-Norm[0]
#                                                                  Transformer-4-FeedForward-Norm[0]
#                                                                  Transformer-4-FeedForward-Norm[0]
# __________________________________________________________________________________________________
# Transformer-5-MultiHeadSelfAtte (None, None, 768)    0           Transformer-5-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-5-MultiHeadSelfAtte (None, None, 768)    0           Transformer-4-FeedForward-Norm[0]
#                                                                  Transformer-5-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-5-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-5-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-5-FeedForward (Feed (None, None, 768)    4722432     Transformer-5-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-5-FeedForward-Dropo (None, None, 768)    0           Transformer-5-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-5-FeedForward-Add ( (None, None, 768)    0           Transformer-5-MultiHeadSelfAttent
#                                                                  Transformer-5-FeedForward-Dropout
# __________________________________________________________________________________________________
# Transformer-5-FeedForward-Norm  (None, None, 768)    1536        Transformer-5-FeedForward-Add[0][
# __________________________________________________________________________________________________
# Transformer-6-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-5-FeedForward-Norm[0]
#                                                                  Transformer-5-FeedForward-Norm[0]
#                                                                  Transformer-5-FeedForward-Norm[0]
# __________________________________________________________________________________________________
# Transformer-6-MultiHeadSelfAtte (None, None, 768)    0           Transformer-6-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-6-MultiHeadSelfAtte (None, None, 768)    0           Transformer-5-FeedForward-Norm[0]
#                                                                  Transformer-6-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-6-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-6-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-6-FeedForward (Feed (None, None, 768)    4722432     Transformer-6-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-6-FeedForward-Dropo (None, None, 768)    0           Transformer-6-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-6-FeedForward-Add ( (None, None, 768)    0           Transformer-6-MultiHeadSelfAttent
#                                                                  Transformer-6-FeedForward-Dropout
# __________________________________________________________________________________________________
# Transformer-6-FeedForward-Norm  (None, None, 768)    1536        Transformer-6-FeedForward-Add[0][
# __________________________________________________________________________________________________
# Transformer-7-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-6-FeedForward-Norm[0]
#                                                                  Transformer-6-FeedForward-Norm[0]
#                                                                  Transformer-6-FeedForward-Norm[0]
# __________________________________________________________________________________________________
# Transformer-7-MultiHeadSelfAtte (None, None, 768)    0           Transformer-7-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-7-MultiHeadSelfAtte (None, None, 768)    0           Transformer-6-FeedForward-Norm[0]
#                                                                  Transformer-7-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-7-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-7-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-7-FeedForward (Feed (None, None, 768)    4722432     Transformer-7-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-7-FeedForward-Dropo (None, None, 768)    0           Transformer-7-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-7-FeedForward-Add ( (None, None, 768)    0           Transformer-7-MultiHeadSelfAttent
#                                                                  Transformer-7-FeedForward-Dropout
# __________________________________________________________________________________________________
# Transformer-7-FeedForward-Norm  (None, None, 768)    1536        Transformer-7-FeedForward-Add[0][
# __________________________________________________________________________________________________
# Transformer-8-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-7-FeedForward-Norm[0]
#                                                                  Transformer-7-FeedForward-Norm[0]
#                                                                  Transformer-7-FeedForward-Norm[0]
# __________________________________________________________________________________________________
# Transformer-8-MultiHeadSelfAtte (None, None, 768)    0           Transformer-8-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-8-MultiHeadSelfAtte (None, None, 768)    0           Transformer-7-FeedForward-Norm[0]
#                                                                  Transformer-8-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-8-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-8-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-8-FeedForward (Feed (None, None, 768)    4722432     Transformer-8-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-8-FeedForward-Dropo (None, None, 768)    0           Transformer-8-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-8-FeedForward-Add ( (None, None, 768)    0           Transformer-8-MultiHeadSelfAttent
#                                                                  Transformer-8-FeedForward-Dropout
# __________________________________________________________________________________________________
# Transformer-8-FeedForward-Norm  (None, None, 768)    1536        Transformer-8-FeedForward-Add[0][
# __________________________________________________________________________________________________
# Transformer-9-MultiHeadSelfAtte (None, None, 768)    2362368     Transformer-8-FeedForward-Norm[0]
#                                                                  Transformer-8-FeedForward-Norm[0]
#                                                                  Transformer-8-FeedForward-Norm[0]
# __________________________________________________________________________________________________
# Transformer-9-MultiHeadSelfAtte (None, None, 768)    0           Transformer-9-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-9-MultiHeadSelfAtte (None, None, 768)    0           Transformer-8-FeedForward-Norm[0]
#                                                                  Transformer-9-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-9-MultiHeadSelfAtte (None, None, 768)    1536        Transformer-9-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-9-FeedForward (Feed (None, None, 768)    4722432     Transformer-9-MultiHeadSelfAttent
# __________________________________________________________________________________________________
# Transformer-9-FeedForward-Dropo (None, None, 768)    0           Transformer-9-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-9-FeedForward-Add ( (None, None, 768)    0           Transformer-9-MultiHeadSelfAttent
#                                                                  Transformer-9-FeedForward-Dropout
# __________________________________________________________________________________________________
# Transformer-9-FeedForward-Norm  (None, None, 768)    1536        Transformer-9-FeedForward-Add[0][
# __________________________________________________________________________________________________
# Transformer-10-MultiHeadSelfAtt (None, None, 768)    2362368     Transformer-9-FeedForward-Norm[0]
#                                                                  Transformer-9-FeedForward-Norm[0]
#                                                                  Transformer-9-FeedForward-Norm[0]
# __________________________________________________________________________________________________
# Transformer-10-MultiHeadSelfAtt (None, None, 768)    0           Transformer-10-MultiHeadSelfAtten
# __________________________________________________________________________________________________
# Transformer-10-MultiHeadSelfAtt (None, None, 768)    0           Transformer-9-FeedForward-Norm[0]
#                                                                  Transformer-10-MultiHeadSelfAtten
# __________________________________________________________________________________________________
# Transformer-10-MultiHeadSelfAtt (None, None, 768)    1536        Transformer-10-MultiHeadSelfAtten
# __________________________________________________________________________________________________
# Transformer-10-FeedForward (Fee (None, None, 768)    4722432     Transformer-10-MultiHeadSelfAtten
# __________________________________________________________________________________________________
# Transformer-10-FeedForward-Drop (None, None, 768)    0           Transformer-10-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-10-FeedForward-Add  (None, None, 768)    0           Transformer-10-MultiHeadSelfAtten
#                                                                  Transformer-10-FeedForward-Dropou
# __________________________________________________________________________________________________
# Transformer-10-FeedForward-Norm (None, None, 768)    1536        Transformer-10-FeedForward-Add[0]
# __________________________________________________________________________________________________
# Transformer-11-MultiHeadSelfAtt (None, None, 768)    2362368     Transformer-10-FeedForward-Norm[0
#                                                                  Transformer-10-FeedForward-Norm[0
#                                                                  Transformer-10-FeedForward-Norm[0
# __________________________________________________________________________________________________
# Transformer-11-MultiHeadSelfAtt (None, None, 768)    0           Transformer-11-MultiHeadSelfAtten
# __________________________________________________________________________________________________
# Transformer-11-MultiHeadSelfAtt (None, None, 768)    0           Transformer-10-FeedForward-Norm[0
#                                                                  Transformer-11-MultiHeadSelfAtten
# __________________________________________________________________________________________________
# Transformer-11-MultiHeadSelfAtt (None, None, 768)    1536        Transformer-11-MultiHeadSelfAtten
# __________________________________________________________________________________________________
# Transformer-11-FeedForward (Fee (None, None, 768)    4722432     Transformer-11-MultiHeadSelfAtten
# __________________________________________________________________________________________________
# Transformer-11-FeedForward-Drop (None, None, 768)    0           Transformer-11-FeedForward[0][0]
# __________________________________________________________________________________________________
# Transformer-11-FeedForward-Add  (None, None, 768)    0           Transformer-11-MultiHeadSelfAtten
#                                                                  Transformer-11-FeedForward-Dropou
# __________________________________________________________________________________________________
# Transformer-11-FeedForward-Norm (None, None, 768)    1536        Transformer-11-FeedForward-Add[0]
# ==================================================================================================
# Total params: 101,677,056
# Trainable params: 101,677,056
# Non-trainable params: 0
# __________________________________________________________________________________________________
# None
