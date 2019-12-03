import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re

import sys
import os
import tensorflow as tf
import tensorflow.keras as keras
#import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Embedding, Concatenate, Add, Activation,Dot
from tensorflow.keras.layers import Dense, Input, Flatten,Reshape, MaxPooling2D,MaxPooling3D
from tensorflow.keras.layers import Conv3D, MaxPooling1D, Embedding, Dropout, AdditiveAttention, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt


#EMBEDDING_DIM = 100
#user_review_num = 2000
#item_review_num = 2000



drop_rate = 0.2


def make_emb_matrix():

    embeddings_index = {}
    f = open('glove.6B.100d.txt',encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))
    return embeddings_index

def get_vector_matrix(word_index,EMBEDDING_DIM,embeddings_index):

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
"""
item_embedding_matrix = np.random.random((len(item_word_index) + 1, EMBEDDING_DIM))
for word, i in item_word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        item_embedding_matrix[i] = embedding_vector
"""

def get_model(user_embedding_matrix,item_embedding_matrix,user_word_index,item_word_index,USER_SEQUENCE_LENGTH,ITEM_SEQUENCE_LENGTH,user_review_num,item_review_num,conv_filters=128,attention_units=32,embedding_id=32,filter_sizes = [3,4,5],EMBEDDING_DIM=100):
    user_embedding_layer = Embedding(len(user_word_index) + 1,
                            EMBEDDING_DIM,weights=[user_embedding_matrix],
                            input_length=USER_SEQUENCE_LENGTH,trainable=True)
    item_embedding_layer = Embedding(len(item_word_index) + 1,
                            EMBEDDING_DIM,weights=[item_embedding_matrix],
                            input_length=ITEM_SEQUENCE_LENGTH,trainable=True)

    user_sequence_input = Input(shape=(USER_SEQUENCE_LENGTH,user_review_num), dtype='int32')
    print(user_sequence_input.shape)
    user_embedded_reviews = user_embedding_layer(user_sequence_input)
    print(user_embedded_reviews.shape)
    user_embedded_reviews_flat = Reshape((user_review_num,USER_SEQUENCE_LENGTH,EMBEDDING_DIM,1))(user_embedded_reviews)
    print(user_embedded_reviews_flat.shape)
    conv_out = []
    for f_size in filter_sizes:
        l_cov1= Conv3D(conv_filters, (1,f_size,EMBEDDING_DIM), activation='relu',padding='valid')(user_embedded_reviews_flat)
        print(l_cov1.shape)
        l_pool1 = MaxPooling3D(pool_size =(1,USER_SEQUENCE_LENGTH-f_size+1,1),padding='valid')(l_cov1)
        print(l_pool1.shape)
        l_flat = Flatten()(l_pool1)
        print(l_flat.shape)
        conv_out.append(l_flat)
    conv_joined = Concatenate()(conv_out)
    print(conv_joined.shape)    


    item_sequence_input = Input(shape=(ITEM_SEQUENCE_LENGTH,item_review_num), dtype='int32')
    item_embedded_reviews = item_embedding_layer(item_sequence_input)
    item_embedded_reviews_flat = Reshape((item_review_num,ITEM_SEQUENCE_LENGTH,EMBEDDING_DIM,1))(item_embedded_reviews)
    print(item_embedded_reviews_flat.shape)
    item_conv_out = []
    for f_size in filter_sizes:
        l_cov1= Conv3D(conv_filters, (1,f_size,EMBEDDING_DIM), activation='relu',padding='valid')(item_embedded_reviews_flat)
        print(l_cov1.shape)
        l_pool1 = MaxPooling3D(pool_size =(1,ITEM_SEQUENCE_LENGTH-f_size+1,1),padding='valid')(l_cov1)
        print(l_pool1.shape)
        l_flat = Flatten()(l_pool1)
        print(l_flat.shape)
        item_conv_out.append(l_flat)
    item_conv_joined = Concatenate()(item_conv_out)
    print(item_conv_joined.shape)    



    user_flat = Reshape((user_review_num,conv_filters*len(filter_sizes)))(conv_joined)
    print(user_flat.shape)
    #user_drop = Dropout(1.0)(user_flat)

    total_item = 1000000
    u_iid = Input(shape=(user_review_num), dtype='int32')
    item_id_embedding = Embedding(total_item + 2,
                                embedding_id,
                                input_length=1,trainable=True)
    item_embs = item_id_embedding(u_iid)
    item_embs = Activation('relu')(item_embs)
    print(item_embs.shape)
    ##

    user_atten = Dense(attention_units,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(user_flat)
    print(user_atten.shape) 
    item_id_atten = Dense(attention_units,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(item_embs)
    print(item_id_atten.shape)
    added = Add()([user_atten,item_id_atten])
    print(added.shape)
    added = Activation('relu')(added)
    user_a = Dense(1)(added)
    print(user_a.shape)
    user_a = tf.keras.activations.softmax(user_a)
    #user_a= AdditiveAttention()([user_flat,item_embs])
    u_feas = Multiply()([user_flat,user_a])
    print(u_feas.shape)
    u_feas  = tf.keras.backend.sum(u_feas,axis = 1)
    print(u_feas.shape)
    u_feas = Dropout(drop_rate)(u_feas)


    item_flat = Reshape((item_review_num,conv_filters*len(filter_sizes)))(item_conv_joined)
    #item_drop = Dropout(1.0)(item_flat)
    total_users = 200000
    i_uid = Input(shape=(item_review_num,), dtype='int32')
    user_id_embedding = Embedding(total_users + 2,
                                embedding_id,
                                input_length=1,trainable=True)
    user_embs = user_id_embedding(i_uid)
    user_embs = Activation('relu')(user_embs)

    item_atten = Dense(attention_units,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(item_flat)
    print(item_atten.shape) 
    user_id_atten = Dense(attention_units,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(user_embs)
    print(user_id_atten.shape)
    item_added = Add()([item_atten,user_id_atten])
    print(item_added.shape)
    item_added = Activation('relu')(item_added)
    item_a = Dense(1)(item_added)
    print(item_a.shape)
    item_a = tf.keras.activations.softmax(item_a,axis=1)
    #user_a= AdditiveAttention()([user_flat,item_embs])
    i_feas = Multiply()([item_flat,item_a])
    print(i_feas.shape)
    i_feas  = tf.keras.backend.sum(i_feas,axis = 1)
    print(i_feas.shape)
    i_feas = Dropout(drop_rate)(i_feas)


    uid = Input(shape=(1), dtype='int32')
    iid = Input(shape=(1,), dtype='int32')
    item_id_embedding = Embedding(total_item + 2,
                                embedding_id,
                                input_length=1,trainable=True)
    item_id_emb = item_id_embedding(iid)
    print(item_id_emb.shape)
    #item_embs = Activation('relu')(item_embs)
    user_id_embedding = Embedding(total_users + 2,
                                embedding_id,
                                input_length=1,trainable=True)
    user_id_emb = user_id_embedding(uid)
    print(user_id_emb.shape)
    #user_embs = Activation('relu')(user_embs)
    u_feas_latent = Dense(embedding_id)(u_feas)
    print(user_id_atten.shape)
    u_feas = Add()([u_feas_latent,user_id_emb])

    i_feas_latent = Dense(embedding_id)(i_feas)
    i_feas = Add()([i_feas_latent,item_id_emb])
    print(i_feas.shape,u_feas.shape)
    u_feas =tf.keras.backend.squeeze(u_feas,axis=1)
    i_feas =tf.keras.backend.squeeze(i_feas,axis=1)
    print(u_feas.shape)
    preds = Dot(axes=-1)([u_feas ,i_feas])
    #preds =tf.keras.backend.squeeze(preds,axis=1)
    print(preds.shape)

    model = Model(inputs=[user_sequence_input,item_sequence_input,u_iid,i_uid,uid,iid], outputs=preds)
    model.compile(optimizer='adam',
                loss='mse',
                metrics=['accuracy'])
    return model            