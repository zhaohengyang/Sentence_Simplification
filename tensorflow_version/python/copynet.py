from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import numerics
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.util.all_util import make_all

# Bring more nn-associated functionality into this package.
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.nn_ops import *
from tensorflow.python.ops.candidate_sampling_ops import *
from tensorflow.python.ops.embedding_ops import *
from tensorflow.python.ops.rnn import *
import tensorflow as tf


def copynet_sampled_softmax_loss(weights, biases, inputs, labels, num_samples,
                                 num_classes,
                                 copynet_W,
                                 copynet_biases,
                                 oov_size,
                                 memory,
                                 encoder_size,
                                 encoder_inputs,
                                 num_true=1,
                                 sampled_values=None,
                                 remove_accidental_hits=True,
                                 partition_strategy="mod",
                                 name="sampled_softmax_loss",
                                 activation=tf.nn.tanh):
    """Computes and returns the sampled softmax training loss.

    This is a faster way to train a softmax classifier over a huge number of
    classes.

    This operation is for training only.  It is generally an underestimate of
    the full softmax loss.

    At inference time, you can compute full softmax probabilities with the
    expression `tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)`.

    See our [Candidate Sampling Algorithms Reference]
    (../../extras/candidate_sampling.pdf)

    Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
    ([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.

    Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-sharded) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    copynet_W: A `Tensor` of type `float32` and shape `[dim, dim]`.
    copynet_biases: A `Tensor` of type `float32` and shape `[dim]`.
    oov_size: An `int`. The number of oov placeholder for unknow words in the sequence inputs.
        oov are located after the vocab
    memory: A `Tensor` of shape `[encoder_size, dim]`. The encoder output
        of the input sequences.
    encoder_size: An `int`. The size of encoder input sequences.
    encoder_inputs: A `Tensor` of shape `[batch_size, encoder_size]`. A list of word ids.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        True.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).
    activation: activation function used in calcualte_copynet_score_from_labels()
      and calcualte_copynet_score_from_samples()

    Returns:
    A `batch_size` 1-D tensor of per-example sampled softmax losses.

    """
    logits, labels = _copynet_compute_sampled_logits(
        weights, biases, inputs, labels, num_samples,
        num_classes,
        copynet_W,
        copynet_biases,
        oov_size,
        memory,
        encoder_size,
        encoder_inputs,
        num_true=num_true,
        sampled_values=sampled_values,
        subtract_log_q=True,
        remove_accidental_hits=remove_accidental_hits,
        partition_strategy=partition_strategy,
        name=name,
        activation=activation)
    sampled_losses = nn_ops.softmax_cross_entropy_with_logits(logits, labels)
    # sampled_losses is a [batch_size] tensor.
    return sampled_losses





def _sum_rows(x):
    """Returns a vector summing up each row of the matrix x."""
    # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
    # a matrix.  The gradient of _sum_rows(x) is more efficient than
    # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
    # we use _sum_rows(x) in the nce_loss() computation since the loss
    # is mostly used for training.
    cols = array_ops.shape(x)[1]
    ones_shape = array_ops.pack([cols, 1])
    ones = array_ops.ones(ones_shape, x.dtype)
    return array_ops.reshape(math_ops.matmul(x, ones), [-1])


def _copynet_compute_sampled_logits(weights, biases, inputs, labels, num_sampled,
                                        num_classes,
                                        copynet_W,
                                        copynet_biases,
                                        oov_size,
                                        memory,
                                        encoder_size,
                                        encoder_inputs,
                                        num_true=1,
                                        sampled_values=None,
                                        subtract_log_q=True,
                                        remove_accidental_hits=False,
                                        partition_strategy="mod",
                                        name=None,
                                        activation=tf.nn.tanh):


    """Helper function for nce_loss and sampled_softmax_loss functions.

    Computes sampled output training logits and labels suitable for implementing
    e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
    sampled_softmax_loss).

    Note: In the case where num_true > 1, we assign to each target class
    the target probability 1 / num_true so that the target probabilities
    sum to 1 per-example.

    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.

      biases: A `Tensor` of shape `[num_classes]`.  The class biases.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.  Note that this format differs from
          the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
      num_sampled: An `int`.  The number of classes to randomly sample per batch.
      num_classes: An `int`. The number of possible classes.
      copynet_W: A `Tensor` of type `float32` and shape `[dim, dim]`.
      copynet_biases: A `Tensor` of type `float32` and shape `[dim]`.
      oov_size: An `int`. The number of oov placeholder for unknow words in the sequence inputs.
       oov are located after the vocab
      memory: A `Tensor` of shape `[encoder_size, dim]`. The encoder output
            of the input sequences.
      encoder_size: An `int`. The size of encoder input sequences.
      encoder_inputs: A `Tensor` of shape `[batch_size, encoder_size]`. A list of word ids.
                      Or a list of `Tensor` with  shape [encoder_size].
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      subtract_log_q: A `bool`.  whether to subtract the log expected count of
          the labels in the sample to get the logits of the true labels.
          Default is True.  Turn off for Negative Sampling.
      remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  Default is
          False.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).
      activation: activation function used in calcualte_copynet_score_from_labels()
      and calcualte_copynet_score_from_samples()
    Returns:
      out_logits, out_labels: `Tensor` objects each with shape
          `[batch_size, num_true + num_sampled]`, for passing to either
          `nn.sigmoid_cross_entropy_with_logits` (NCE) or
          `nn.softmax_cross_entropy_with_logits` (sampled softmax).
    """
    if not isinstance(weights, list):
        weights = [weights]

    with ops.op_scope(
                    weights + [biases, inputs, labels], name, "compute_sampled_logits"):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)
        labels_flat = array_ops.reshape(labels, [-1])

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape = [batch_size, 1] tensor
        #   sampled_expected_count shape = [num_sampled] tensor
        if sampled_values is None:
            sampled_values = candidate_sampling_ops.uniform_candidate_sampler(
                true_classes=labels,
                num_true=num_true,
                num_sampled=num_sampled,
                unique=True,
                range_max=num_classes)
        # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        # pylint: disable=unpacking-non-sequence
        sampled, true_expected_count, sampled_expected_count = sampled_values

        # labels_flat is a [batch_size * num_true] tensor
        # sampled is a [num_sampled] int tensor
        all_ids = array_ops.concat(0, [labels_flat, sampled])

        # weights shape is [num_classes, dim]
        dim = array_ops.shape(inputs)[1:2]
        oov_filter = array_ops.concat(0, [array_ops.ones([num_classes - oov_size, dim[0]]),
                                          array_ops.zeros([oov_size, dim[0]])])

        weights_vocab = math_ops.mul(weights[0], oov_filter)
        all_w = embedding_ops.embedding_lookup(
            [weights_vocab], all_ids, partition_strategy=partition_strategy)

        oov_filter = array_ops.concat(0, [array_ops.ones([num_classes - oov_size]), array_ops.zeros([oov_size])])
        biases_vocab = math_ops.mul(biases, oov_filter)
        all_b = embedding_ops.embedding_lookup(biases_vocab, all_ids)
        # true_w shape is [batch_size * num_true, dim]
        # true_b is a [batch_size * num_true] tensor
        true_w = array_ops.slice(
            all_w, [0, 0], array_ops.pack([array_ops.shape(labels_flat)[0], -1]))
        true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))

        # inputs shape is [batch_size, dim]
        # true_w shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        new_true_w_shape = array_ops.concat(0, [[-1, num_true], dim])
        row_wise_dots = math_ops.mul(
            array_ops.expand_dims(inputs, 1),
            array_ops.reshape(true_w, new_true_w_shape))
        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = array_ops.reshape(row_wise_dots,
                                           array_ops.concat(0, [[-1], dim]))
        true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
        true_b = array_ops.reshape(true_b, [-1, num_true])
        true_logits += true_b

        # copynet_label_scores [batch_size, num_true]
        copynet_true_logits = calcualte_copynet_score_from_labels(copynet_W=copynet_W,
                                                                  copynet_biases=copynet_biases,
                                                                  labels=labels, memory=memory,
                                                                  encoder_size=encoder_size,
                                                                  encoder_inputs=encoder_inputs,
                                                                  r_t=inputs,
                                                                  activation=activation)
        true_logits += copynet_true_logits

        # Lookup weights and biases for sampled labels.
        #   sampled_w shape is [num_sampled, dim]
        #   sampled_b is a [num_sampled] float tensor
        sampled_w = array_ops.slice(
            all_w, array_ops.pack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
        sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

        # inputs has shape [batch_size, dim]
        # sampled_w has shape [num_sampled, dim]
        # sampled_b has shape [num_sampled]
        # Apply X*W'+B, which yields [batch_size, num_sampled]
        sampled_logits = math_ops.matmul(inputs,
                                         sampled_w,
                                         transpose_b=True) + sampled_b

        copynet_sampled_logits = calcualte_copynet_score_from_samples(copynet_W=copynet_W,
                                                                      copynet_biases=copynet_biases,
                                                                      sample_ids=sampled, memory=memory,
                                                                      encoder_size=encoder_size,
                                                                      encoder_inputs=encoder_inputs,
                                                                      r_t=inputs,
                                                                      activation=activation)
        sampled_logits += copynet_sampled_logits

        if remove_accidental_hits:
            acc_hits = candidate_sampling_ops.compute_accidental_hits(
                labels, sampled, num_true=num_true)
            acc_indices, acc_ids, acc_weights = acc_hits

            # This is how SparseToDense expects the indices.
            acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
            acc_ids_2d_int32 = array_ops.reshape(math_ops.cast(
                acc_ids, dtypes.int32), [-1, 1])
            sparse_indices = array_ops.concat(
                1, [acc_indices_2d, acc_ids_2d_int32], "sparse_indices")
            # Create sampled_logits_shape = [batch_size, num_sampled]
            sampled_logits_shape = array_ops.concat(
                0,
                [array_ops.shape(labels)[:1], array_ops.expand_dims(num_sampled, 0)])
            if sampled_logits.dtype != acc_weights.dtype:
                acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
            acc_hit_matix = sparse_ops.sparse_to_dense(
                sparse_indices, sampled_logits_shape, acc_weights,
                default_value=0.0, validate_indices=False)
            sampled_logits += acc_hit_matix

        if subtract_log_q:
            # Subtract log of Q(l), prior probability that l appears in sampled.
            true_logits -= math_ops.log(true_expected_count)
            sampled_logits -= math_ops.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = array_ops.concat(1, [true_logits, sampled_logits])
        # true_logits is a float tensor, ones_like(true_logits) is a float tensor
        # of ones. We then divide by num_true to ensure the per-example labels sum
        # to 1.0, i.e. form a proper probability distribution.
        out_labels = array_ops.concat(
            1, [array_ops.ones_like(true_logits) / num_true,
                array_ops.zeros_like(sampled_logits)])
    return out_logits, out_labels

def calcualte_copynet_score_from_labels(copynet_W, copynet_biases, labels, memory,
                                            encoder_size, encoder_inputs, r_t, activation=tf.nn.tanh):
    '''
    Args:
        copynet_W: A `Tensor` of type `float32` and shape `[dim, dim]`.
        copynet_biases: A `Tensor` of type `float32` and shape `[dim]`.
        labels: A `Tensor` shape `[batch_size, num_true]`. A list of word ids.
        memory: A `Tensor` shape `[batch_size, encoder_size, dim]`. The encoder output
            of the input sequences. Or a list (with size [of `Tensor` with  shape [batch_size, dim].
        encoder_size: An `int`. The size of encoder input sequences.
        encoder_inputs: A `Tensor` of shape `[batch_size, encoder_size]`.
                        Or a list of `Tensor` with  shape [encoder_size].

        r_t: A `Tensor` of shape [batch_size, dim]
    Returns:
        score: `Tensor` of shape `[batch_size, num_true]`
    '''

    if isinstance(encoder_inputs, list):
        encoder_inputs = tf.concat(1, [tf.reshape(i, [-1, 1]) for i in encoder_inputs])
    if labels.dtype != tf.int64:
        labels = tf.cast(labels, tf.int64)
    if encoder_inputs.dtype != tf.int64:
        encoder_inputs = tf.cast(encoder_inputs, tf.int64)
    if isinstance(memory, list):
        batch_size = tf.shape(labels)[0]
        memory = tf.reshape(tf.concat(1, memory), tf.concat(0, [[batch_size], [encoder_size], [-1]]))

    # labels_flat: [batch_size*num_true]
    # input_matrix, compare_result: [batch_size, num_true, encoder_size]
    # reduced_sum: [batch_size, num_true, 1]
    # normalized_compare_result: [batch_size, num_true, encoder_size]
    # memory_embedded_inputs: [batch_size, num_true, encoder_size] batch_matmul [batch_size, encoder_size, dim]
    # = [batch_size, num_true, dim]
    labels_flat = tf.reshape(labels, [-1])
    input_matrix = tf.concat(1, [tf.expand_dims(labels_flat, 1)] * encoder_size, name='concat')
    input_matrix = tf.reshape(input_matrix, tf.concat(0, [tf.shape(labels), [-1]]))
    input_matrix_shape = tf.shape(input_matrix)
    compare_result = tf.cast((tf.equal(input_matrix, tf.expand_dims(encoder_inputs, 1))), tf.float32)
    reduced_sum = tf.expand_dims(tf.reduce_sum(compare_result, [2]), 2)
    # to avoid 0/0
    reduced_sum = tf.maximum(reduced_sum, tf.constant(1.0))
    normalized_compare_result = tf.div(compare_result, reduced_sum)

    # r_t_expand: [batch_size, dim, 1]
    # copynet_scores: [batch_size, num_true]
    weighted_memory = activation(batch_vm(memory, copynet_W) + copynet_biases)
    memory_embedded_inputs = tf.batch_matmul(normalized_compare_result, weighted_memory)
    r_t_expand = tf.expand_dims(r_t, dim=2)
    copynet_scores = tf.batch_matmul(memory_embedded_inputs, r_t_expand)
    copynet_scores = tf.reshape(copynet_scores, tf.concat(0, [tf.shape(r_t)[0:1], [-1]]))
    #     copynet_scores_shape = tf.shape(copynet_scores)
    return copynet_scores



def calcualte_copynet_score_from_samples(copynet_W, copynet_biases, sample_ids, memory,
                                         encoder_size, encoder_inputs, r_t, activation=tf.nn.tanh):
    '''
    Args:
        copynet_W: A `Tensor` of type `float32` and shape `[dim, dim]`.
        copynet_biases: A `Tensor` of type `float32` and shape `[dim]`.
        sample_ids: A `Tensor` of type `int32` and shape `[num_sampled]`. A list of word ids.
        memory: A `Tensor` shape `[batch_size, encoder_size, dim]`. The encoder output of the
            input sequences. Or a list (with size [encoder_size]) of `Tensor` with  shape
            [batch_size, dim].
        encoder_size: An `int`. The size of encoder input sequences.
        encoder_inputs: A `Tensor` of shape `[batch_size, encoder_size]`.
                        Or a list (with size [encoder_size]) of `Tensor` with  shape [batch_size].
        r_t:  A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the decoder at time t.
    Returns:
        score: `Tensor` of shape `[batch_size, num_sampled]`
    '''

    if isinstance(encoder_inputs, list):
        encoder_inputs = tf.concat(1, [tf.reshape(i, [-1, 1]) for i in encoder_inputs])
    if sample_ids.dtype != tf.int64:
        sample_ids = tf.cast(sample_ids, tf.int64)
    if encoder_inputs.dtype != tf.int64:
        encoder_inputs = tf.cast(encoder_inputs, tf.int64)
    if isinstance(memory, list):
        batch_size = tf.shape(encoder_inputs)[0]
        memory = tf.reshape(tf.concat(1, memory), tf.concat(0, [[batch_size], [encoder_size], [-1]]))

    # encoder_inputs: [batch_size, encoder_size]
    # sample_matrix: [any, encoder_size]
    # compare_result, normalized_compare_result: [batch_size, any, encoder_size]
    # reduced_sum: [batch_size, any]
    # memory_embedded_inputs: [batch_size, any, encoder_size] batch_matmul [batch_size, encoder_size, dim]
    # = [batch_size, any, dim]
    sample_matrix = tf.concat(1, [tf.expand_dims(sample_ids, 1)] * encoder_size, name='concat')
    compare_result = tf.cast((tf.equal(sample_matrix, tf.expand_dims(encoder_inputs, 1))), tf.float32)
    reduced_sum = tf.expand_dims(tf.reduce_sum(compare_result, [2]), [2])
    # to avoid 0/0
    reduced_sum = tf.maximum(reduced_sum, tf.constant(1.0))
    normalized_compare_result = tf.div(compare_result, reduced_sum)

    weighted_memory = activation(batch_vm(memory, copynet_W) + copynet_biases)
    memory_embedded_inputs = tf.batch_matmul(normalized_compare_result, weighted_memory)
    # result: [batch_size, any, dim] batch_matmul [batch_size, dim, 1] = [batch_size, any, 1]
    # copynet_scores: [batch_size, any]
    copynet_scores = tf.batch_matmul(memory_embedded_inputs, tf.expand_dims(r_t, dim=2))
    copynet_scores = tf.squeeze(copynet_scores, [2])
    return copynet_scores

# check batch_vm
def batch_vm(v, m):
    '''
    Args:
        v: A `Tensor` of shape [batch_size, r_x, c_x]
        m: A `Tensor` of shape [r_y, c_y]
    Returns:
        vm: A `Tensor` of shape [batch_size, r_x, c_y]
    '''
    shape = tf.shape(v)
    rank = shape.get_shape()[0].value
    v = tf.expand_dims(v, rank)

    vm = tf.mul(v, m)
    return tf.reduce_sum(vm, rank-1)

