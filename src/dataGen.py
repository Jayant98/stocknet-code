
import nltk
nltk.download('punkt')
from DataPipe import DataPipe
import numpy as np
import tensorflow as tf
from keras.models import Model

pipe = DataPipe()

train_batch_gen = pipe.batch_gen(phase='train')

for train_batch_dict in train_batch_gen:
  x = train_batch_dict['word_batch'].reshape(8,11,1,768)
  print(train_batch_dict)
  break


'''RRE_TRAINED_MODEL_NAME = 'roberta-base' 

input_ids_layer = tf.keras.layers.Input(shape=512, name="input_ids", dtype='int32')
attn_mask_layer = tf.keras.layers.Input(shape=512, name="attn_mask", dtype='int32')

lang_model = TFRobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

out = lang_model(input_ids =input_ids_layer, attention_mask = attn_mask_layer).last_hidden_state
model=Model([input_ids_layer ,attn_mask_layer], out)
train_batch_gen = pipe.batch_gen_bert(phase='train')  # a new gen for a new epoch
model.compile()
                
for train_batch_dict in train_batch_gen:

    to_add = np.zeros((8,11,1,512,768))
    
    for ind in range(8):
        ids = train_batch_dict['word_batch_id'][ind]
        attn = train_batch_dict['word_batch_attn'][ind]
        for j in range(11):
            print(ids.shape, ids[j].shape)
            embeddings = model([ids[j],attn[j]])
            embeddings = np.array(embeddings)

            to_add[ind][j] = embeddings

    feed_dict = {"training": True,
                  "batch_size": np.array(8),
                  "T": train_batch_dict['T_batch'],
                  "n_words": train_batch_dict['n_words_batch'].reshape((8,11,1)),
                  "n_msg": train_batch_dict['n_msgs_batch'].reshape(8,11),
                  "y": train_batch_dict['y_batch'],
                  "ss_ind": train_batch_dict['ss_index_batch'],
                  "embeddings": to_add,
                  }
'''