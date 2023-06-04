# coding=utf-8

"""
    Collection of custom layer implementations.
    Here we provide the contrastive adversarial training (CAT) with the original BERT model.
    Our main references is:
        -- Google BERT codes: https://github.com/google-research/bert/blob/master/modeling.py#L375
        -- adversarial text classification: https://github.com/tensorflow/models/tree/master/research/adversarial_text
        -- r-drop codes: https://github.com/dropreg/R-Drop
    It is only shown for the rough implementation of CAT. But `No guarantee that code can be run directly`,
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

class BertModel(object):
  """
  BERT model ("Bidirectional Encoder Representations from Transformers").
  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               labels
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None,
               **kwargs):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    self.input_ids = input_ids
    self.input_mask = input_mask
    self.config = config
    self.labels = labels

    self.norm_length = 5.0  # normal_length for adversarial training
    self.loss_weights = [1, 0, 0]   # loss weighting values for loss_bce, ate, adv
    self.r_drop_rate = 0.02 # r-drop initial rate value
    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)
    
      # compute the original loss and logit. 
      origin_loss, origin_logit = self.cl_loss_from_embedding(return_logit=True)
      new_loss = adversarial_loss(self.embedding_output, origin_loss, origin_logit, self.cl_loss_from_embedding, self.norm_length, self.loss_weights)
      self.output = {"logit": origin_logit, "loss": new_loss}

  def cl_loss_from_embedding(self, return_logit=False):
      logit_old = self.cl_logit_from_embedding(self.config, self.embedding)
      y_true = tf.cast(tf.reshape(self.labels, (-1,)), tf.float32)
      logits=tf.reshape(logit_old, (-1,))
      loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_true)
      if return_logit:
        return loss, logits
      else:
        return loss

  def adv_output(self):
      return self.output

  def cl_logit_from_embedding(self, config, embedding_output):
      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            self.input_ids, self.input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        self.first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        print('**the self.first_token_tensor is: \n',self.first_token_tensor)
        self.pooled_output = tf.layers.dense(
            self.first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))
     # post process
     # to_do

def adversarial_loss(embedding, old_loss, old_logit, loss_fn, norm_length, loss_weights):
    l_bce_weight, l_ate_weight, l_adv_weight = loss_weights  # loss weighting for loss_bce, ate, adv
    """Adds gradient to embedding and recomputes loss for adversarial training."""
    with tf.variable_scope("adversarial_loss"):
      grad, = tf.gradients(
          old_loss,
          embedding,
          aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
      grad = tf.stop_gradient(grad) # decide whether you use loss_bce, and you can also implement it with other methods
      perturb = _scale_l2(grad, norm_length)
    adv_loss, adv_logit = loss_fn(embedding + perturb, return_logit=True)
    tf.summary.scalar("total_loss/ate_loss", tf.reduce_mean(adv_loss, axis=0))
    # KL diverage Loss for old_logit and adv_logit
    KLD_loss = 0.5*tf.keras.losses.KLD(old_logit,adv_logit) + 0.5*tf.keras.losses.KLD(adv_logit,old_logit)
    new_loss = l_bce_weight*old_loss + l_ate_weight * adv_loss
    tf.summary.scalar("total_loss/adv_loss", tf.reduce_mean(KLD_loss))
    total_loss = new_loss + l_adv_weight*KLD_loss
    return total_loss

def _scale_l2(x, norm_length):
    alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit

def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """
  activation_value = None
  pass  # you can reference the official BERT code in (https://github.com/google-research/bert)
  return activation_value

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)

def transformer_model(**kwargs):
    """
    Multi-headed, multi-layer Transformer from "Attention is All You Need".
    This is almost an exact implementation of the original Transformer encoder.
    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """
    final_output = None
    pass    # you can reference the official BERT code in (https://github.com/google-research/bert)
    return final_output

def embedding_postprocessor(**kwargs):
    """
    Performs various post-processing on a word embedding tensor.
    Args:
        **args**
    Returns:
        float tensor with same shape as `input_tensor`.

    Raises:
        ValueError: One of the tensor shapes or input values is invalid.
  """
    output = None 
    pass    # you can reference the official BERT code in (https://github.com/google-research/bert)
    return output

def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))