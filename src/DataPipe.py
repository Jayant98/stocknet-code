# encoding=utf-8
#!/usr/local/bin/python
import os
import io
import json
import numpy as np
from datetime import datetime, timedelta
import random

from numpy.core.fromnumeric import argpartition
from ConfigLoader import logger, path_parser, config_model, dates, stock_symbols, vocab, vocab_size

import  time
import pandas as pd

from nltk.tokenize import word_tokenize

from keras.preprocessing.sequence import pad_sequences

class DataPipe:

    def __init__(self):
        # load path
        self.pretrained_model = 'roberta-base'
        
        # load model config
        self.batch_size = config_model['batch_size']
        self.shuffle = config_model['shuffle']

        self.max_n_days = config_model['max_n_days']
        self.max_n_words = config_model['max_n_words']
        self.max_n_msgs = config_model['max_n_msgs']
        self.emb_path = config_model['emb_path']

        self.word_embed_size = config_model['word_embed_size']
        self.y_size = config_model['y_size']

        self.movement_path = path_parser.movement
        self.tweet_path = path_parser.preprocessed
        self.vocab_path = path_parser.vocab
        self.glove_path = path_parser.glove

        self.word_embed_type = config_model['word_embed_type']
        self.word_embed_size = config_model['word_embed_size']

        
        self.parlvote_path = config_model['parlvote_path']
        self.data = self.init_df()
        self.PRE_TRAINED_MODEL_NAME = 'roberta-base' 
        

        
    
    
    def _get_batch_size(self, phase):
        """
            phase: train, dev, test, unit_test
        """
        if phase in ['train',"dev","test"]:
            return self.batch_size
        elif phase == 'unit_test':
            return 5
        else:
            return 1

    @staticmethod
    def _convert_token_to_id(token, token_id_dict):
        if token not in token_id_dict:
            token = 'UNK'
        return token_id_dict[token]

    def index_token(self, token_list, key='id', type='word'):
        assert key in ('id', 'token')
        assert type in ('word', 'stock')
        indexed_token_dict = dict()

        if type == 'word':
            token_list_cp = list(token_list)  # un-change the original input
            token_list_cp.insert(0, 'UNK')  # for unknown tokens
            token_list_cp.insert(1, 'PAD')  # for PADDING
        else:
            token_list_cp = token_list

        if key == 'id':
            for id in range(len(token_list_cp)):
                indexed_token_dict[id] = token_list_cp[id]
        else:
            for id in range(len(token_list_cp)):
                indexed_token_dict[token_list_cp[id]] = id

        # id_token_dict = dict(zip(token_id_dict.values(), token_id_dict.keys()))
        return indexed_token_dict
        
    def _convert_words_to_ids(self, words, vocab_id_dict):
        """
            Replace each word in the data set with its index in the dictionary
        :param words: words in tweet
        :param vocab_id_dict: dict, vocab-id
        :return:
        """
        return [self._convert_token_to_id(w, vocab_id_dict) for w in words]

    
    def init_df(self):
        df = pd.read_csv(self.parlvote_path)
        all_ids = np.unique(df['speaker_id'])
        

        speaker_map = {}
        speaker_party = {}
        for ind, row in df.iterrows():
          if row['speaker_id'] in list(speaker_map.keys()):
            speaker_map[row['speaker_id']].append([ind, row['vote']])
          else:
            speaker_map[row['speaker_id']] = [[ind, row['vote']]]
          
          if row['speaker_id'] in list(speaker_party.keys()):
            speaker_party[row['speaker_id']].append(row['party'])
          else:
            speaker_party[row['speaker_id']] = [row['party']]

        def windower(speaker_map, speaker_party):
          
          windowed_speaker_data = {}
          window=10

          for key in list(speaker_map.keys()):
            data = speaker_map[key]
            if (len(data)>window+1):
              for i in range(len(data)-window):
                start = i
                end = i+window
                to_train = data[start:end]
                to_pred = data[end]

                if (key in list(windowed_speaker_data.keys())):
                  windowed_speaker_data[key].append([to_train,to_pred])
                else:
                  windowed_speaker_data[key] = [[to_train,to_pred]]

          return windowed_speaker_data

        res = windower(speaker_map, speaker_party)

        return res
    
    

    def sample_gen_from_one_stock(self, speaker, phase, vocab_id_dict):
        """
            generate samples for the given stock.

            => tuple, (x:dict, y_:int, price_seq: list of floats, prediction_date_str:str)
        """

        min_ind = 0;
        max_ind = 0;

        speaker_data = self.data[speaker]

        if phase == 'train':
          min_ind = 0;
          max_ind = int(0.8*len(self.data[speaker]))
        else:
          min_ind = int(0.8*len(self.data[speaker]))
          max_ind = len(self.data[speaker])
        nday_tokenized = np.zeros((self.max_n_days,self.max_n_msgs,self.max_n_words, self.word_embed_size))
        nday_labs = np.zeros((self.max_n_days,2))
        ss_index = np.zeros((self.max_n_days,self.max_n_msgs))
        n_word = np.zeros((self.max_n_days,self.max_n_msgs))
        for speech_ind in range(min_ind, max_ind):
            sample = speaker_data[speech_ind]
            train_sample = sample[0]
            eval_sample = sample[1]

            for sub_sample_ind in range(len(train_sample)):
              current_sample = train_sample[sub_sample_ind]
              
              file_id = current_sample[0]
              embds = np.load((self.emb_path + str(file_id)+".npy"))
              if (current_sample[1] == 1):
                nday_labs[sub_sample_ind] = np.array([0.0, 1.0])
              else:
                nday_labs[sub_sample_ind]  = np.array([1.0, 0.0])
              
              ss_index[sub_sample_ind][0] = 1 - 1 # might require change
              n_word[sub_sample_ind][0] = 1 # might require change
              
              nday_tokenized[sub_sample_ind][0] = embds
            
            if (eval_sample[1] == 1):
              e_label = np.array([0.0, 1.0]) 
            else:
              e_label = np.array([1.0, 0.0] ) 
              
            eval_embds = np.load((self.emb_path + str(eval_sample[0])+".npy"))

            ss_index[-1][0] = 1-1
            
            n_word[-1][0] = 0
            
              
            nday_tokenized[-1][0] = eval_embds
            nday_labs[-1] = (e_label)
              
            
            sample_dict = {
                'T': self.max_n_days, 

                'ys': nday_labs, 
                'ss_index': ss_index,
                'msgs': nday_tokenized,
                'n_words': n_word, 
                'n_msgs': 1,
            }

            yield sample_dict


    def batch_gen(self, phase):
        batch_size = self._get_batch_size(phase="train")
        vocab_id_dict = self.index_token(vocab, key='token')

        all_speakers = (list(self.data.keys()))
        all_speakers.sort()
        
        generators = [self.sample_gen_from_one_stock(s, phase, vocab_id_dict) for s in all_speakers]
        # logger.info('{0} Generators prepared...'.format(len(generators)))

        while True:
            #start_time = time.time()


            T_batch = np.zeros([batch_size, ], dtype=np.int32) #?
            y_batch = np.zeros([batch_size, self.max_n_days, self.y_size], dtype=np.float32) #?

            word_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs, self.max_n_words, self.word_embed_size], dtype=np.float32)
            n_msgs_batch = [[[1] for i in range(self.max_n_days)] for j in range(batch_size)]
            n_words_batch = np.zeros([batch_size, self.max_n_days,self.max_n_msgs], dtype=np.int32)
            ss_index_batch = np.zeros([batch_size, self.max_n_days, self.max_n_msgs], dtype=np.int32)
            sample_id = 0
            try:
                while sample_id < batch_size:
                    gen_id = random.randint(0, len(generators)-1)
                    try:
                        sample_dict = next(generators[gen_id])
                        T = sample_dict['T']

                        T_batch[sample_id] = T
                        # target
                        y_batch[sample_id] = sample_dict['ys']
                        ss_index_batch[sample_id] = sample_dict['ss_index']
                        word_batch[sample_id] = sample_dict['msgs']
                        n_words_batch[sample_id] = sample_dict['n_words']

                        sample_id += 1
                    except StopIteration:
                        
                        del generators[gen_id]
                        if generators:
                            continue
                        else:
                            raise StopIteration

                batch_dict = {
                    # meta
                    'batch_size': sample_id,
                    'T_batch': T_batch,
                    # target
                    'ss_index_batch': ss_index_batch,
                    'y_batch': y_batch, # label: up, down or close?
                    'word_batch': word_batch,
                    'n_msgs_batch': np.array(n_msgs_batch),
                    'n_words_batch': n_words_batch,
                }

                yield batch_dict

            except StopIteration:
              print('All generators exhausted')
              break

    