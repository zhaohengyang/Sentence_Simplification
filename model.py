from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
import random, sys
import numpy as np
from headline_generator.attention_layer import SimpleContextLayer
from keras.optimizers import Adam, RMSprop  # usually I prefer Adam but article used rmsprop
import tensorflow as tf



def model1(embedding, config):
    # Load configuration
    seed = config.seed
    weight_decay = config.weight_decay
    maxlen = config.maxlen
    p_emb = config.p_emb
    rnn_layers = config.rnn_layers
    rnn_size = config.rnn_size
    p_dense = config.p_dense
    activation_rnn_size = config.activation_rnn_size
    p_W = config.p_W
    p_U = config.p_U
    optimizer = config.optimizer
    vocab_size, embedding_size = embedding.shape
    print "model1", embedding.shape
    # seed weight initialization
    random.seed(seed)
    np.random.seed(seed)
    regularizer = l2(weight_decay) if weight_decay else None
    # construct model
    model = Sequential()
    # Use embedding layer to embed input one-hot vectors
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=maxlen,
                        W_regularizer=regularizer, dropout=p_emb, weights=[embedding], mask_zero=True))
    for i in range(rnn_layers):
        lstm = LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                       W_regularizer=regularizer, U_regularizer=regularizer,
                       b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U,
                      )
        model.add(lstm)
        model.add(Dropout(p_dense))

    if activation_rnn_size:
        model.add(SimpleContextLayer(config))
    model.add(TimeDistributed(Dense(vocab_size, W_regularizer=regularizer, b_regularizer=regularizer)))

    # tf.nn.softmax(x) # only take weight as input
    # Dense(1, input_dim=2, activation=tf.nn.sampled_softmax_loss())
    model.add(Activation('softmax'))
    # loss function loss(y_predict, y_true)


    # opt = Adam(lr=LR)  # keep calm and reduce learning rate
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model
