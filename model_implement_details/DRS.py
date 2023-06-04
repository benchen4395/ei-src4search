# coding=utf-8

"""
    Collection of custom layer implementations.
    Here we provide the dynamic-length representation scheme (DRS) with the original BERT model.
    Our main references is:
        -- Google BERT codes: https://github.com/google-research/bert/blob/master/modeling.py
    It is only shown for the rough implementation of DRS. But `No guarantee that code can be run directly`,
    When you want to use this solution, you must modify it.
    Warning:
    Your use of any information or materials on these codes is entirely at your own risk, for which we shall not be liable.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


class BERT(Layer):
    """
    BERT layer.
    """
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=False,
                 args=None,
                 pretrain_label_size=2,
                 l2_reg=0.001,
                 seed=None
                 scope=None):
        """
        BERT layer. This is an implementation of tf-BERT model.
        - config: `BertConfig` instance.
        - is_training: bool. true for training model, false for eval model. 
            Controls whether dropout will be applied.
        - use_one_hot_embeddings: (optional) bool. Whether to use one-hot word embeddings 
            or tf.embedding_lookup() for the word embeddings.
        - args: model input tensor, contain: wpe_ids, ner_ids, mask, seg_ids, and labels.
        - seed: parameter inilization seed.
        - scope: (optional) variable scope. Defaults to "bert".
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0 
        pass    # you can reference the official BERT code in (https://github.com/google-research/bert)

    # shorten the tokens length by batch dropping zero padding columns.
    def omit_value_columns(self, batch_embed, omit_value, name):
        if self.bool_mask_tag is None:
            temp_sum = tf.reduce_sum(batch_embed, axis=0)
            self.bool_mask_tag = tf.not_equal(temp_sum, omit_value)
        tensor_wo_zero_cols = tf.boolean_mask(batch_embed, self.bool_mask_tag, name=name+"_wo_zero_cols", axis=1)
        # print('**the omit_value_columns tensor is: \n', tensor_wo_zero_cols)
        return tensor_wo_zero_cols
    
    def tf_apply(self, ids_dict, label, update=True, is_training=False):
        self.update = update
        self.ids_dict = ids_dict
        self.label = label
        if self.update == 'drs':
            input_mask_ids = self.omit_value_columns(ids_dict["mask_ids"], 0, "mask_ids")
            input_wpe_ids = self.omit_value_columns(ids_dict["wpe_ids"], 0, "wpe_ids")
            input_ner_ids = self.omit_value_columns(ids_dict["ner_ids"], 0, "ner_ids")
            token_type_ids = self.omit_value_columns(ids_dict["seg_ids"], 0, "seg_ids")
        else:
            input_mask_ids = self.omit_value_columns(ids_dict["mask_ids"], -1, "mask_ids")
            input_wpe_ids = self.omit_value_columns(ids_dict["wpe_ids"], -1, "wpe_ids")
            input_ner_ids = self.omit_value_columns(ids_dict["ner_ids"], -1, "ner_ids")
            token_type_ids = self.omit_value_columns(ids_dict["seg_ids"], -1, "seg_ids")
        pass    # you can reference the official BERT code in (https://github.com/google-research/bert)