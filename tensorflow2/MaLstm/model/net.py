import tensorflow as tf
from tensorflow.python.keras import backend as K


class MaLSTM(tf.keras.Model):
    def __init__(self, vocab_size, dim, len):
        super(MaLSTM, self).__init__()
        self._len = len
        self._dim = dim
        self._vocab_size = vocab_size

        self._embedding = tf.keras.layers.Embedding(self._vocab_size, self._dim, input_length=self._len,
                                                    trainable=False)
        self._lstm = tf.keras.layers.LSTM(self._len)

        self._malstm_dist = ManDist()

    def call(self, item):
        x = item[0]
        y = item[1]

        q1 = self._embedding(x)
        q2 = self._embedding(y)

        q1_lstm = self._lstm(q1)
        q2_lstm = self._lstm(q2)

        print('q1_lstm')
        print(q1_lstm)
        print('q2_lstm')
        print(q2_lstm)

        malstm_dist = self._malstm_dist([q1_lstm, q2_lstm])
        print("malstm_dist")
        print(malstm_dist)
        return malstm_dist


class ManDist(tf.keras.Model):
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        print("self.result")
        print(self.result)
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
