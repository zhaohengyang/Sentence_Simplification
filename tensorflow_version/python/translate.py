# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.
Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.
Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.
See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
# from tensorflow.models.rnn.translate import seq2seq_model
from headline_generator_tensorflow.python.preprocess import prepare_headline_generation_data, \
    get_glove_embedding_matrix, get_word_embedding_from_valcabulary, initialize_vocabulary, get_glove_embedding
from headline_generator_tensorflow.python import seq2seq_model
from headline_generator_tensorflow.python.preprocess import spacy_tokenizer
from headline_generator_tensorflow.python.experiment_logging import Logging, log

tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
# todo: check for oov_size
tf.app.flags.DEFINE_integer("oov_size", 50,
                            "Number of placeholders for unknown words during word to id processing in pre-processing")
tf.app.flags.DEFINE_integer("sentence_truncate", 3,
                            "Number of sentence select from the begining of the article during pre-processing")
# tf.app.flags.DEFINE_integer("epochs", 30,
#                             "Number of epochs until the training finishes")

tf.app.flags.DEFINE_integer("size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("source_vocab_size", 40000, "Article vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 40000, "headline vocabulary size.")
tf.app.flags.DEFINE_string("model_name", "headline-generator", "Model's name")
tf.app.flags.DEFINE_string("data_dir", "../sample_data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "../train", "Training directory.")
tf.app.flags.DEFINE_string('summaries_dir', '../tensorboard_logs', 'Summaries directory')
tf.app.flags.DEFINE_string("experiment_log_dir", "../experiment_log", "Log directory.")

tf.app.flags.DEFINE_string("glove_dir", "../../model/glove_word_embedding/glove.6B/glove.6B.100d.txt", "glove embedding file directory")
tf.app.flags.DEFINE_string("embedding_path", "../sample_data/init_word_embedding.pkl", "the file containing initialized word embeddings loaded from glove")

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_val_data_size", 1000,
                            "Limit on the size of validation data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 20,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("preprocessing", False,
                            "Set to True for interactive pre-processing.")

tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("embedding", True,
                            "Load word embedding from glove")

tf.app.flags.DEFINE_boolean("copynet", False,
                            "Use copynet")
FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(25, 10)]


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.
    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).
    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

def create_model(session, forward_only, embedding):
    """Create translation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.source_vocab_size, FLAGS.target_vocab_size, FLAGS.oov_size, _buckets,
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        copynet=FLAGS.copynet,
        embedding=embedding
    )
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

def train():
    """Set up log file path"""
    log_file = "{}/experiment.log".format(FLAGS.experiment_log_dir)
    cont_index = 0
    while (os.path.exists(log_file)):
        cont_index += 1
        log_file = "{}/experiment.log.cont-{}".format(FLAGS.experiment_log_dir, cont_index)

    Logging.set_log_file(log_file)
    Logging.print_header(FLAGS.model_name)

    # Prepare WMT data.
    print("Preparing WMT data in %s" % FLAGS.data_dir)
    train_content_ids_path, train_title_ids_path, evl_content_ids_path, evl_title_ids_path, vocab_path = prepare_headline_generation_data(
        FLAGS.data_dir, FLAGS.source_vocab_size, FLAGS.oov_size, sentence_truncate=FLAGS.sentence_truncate)
    _, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    # todo get glove embedding:
    # Load word embedding
    if FLAGS.embedding == True:
        # Use glove matrix to create a word embedding matrix of our vocabulary
        embedding = get_glove_embedding(vocab_path, FLAGS.embedding_path, FLAGS.glove_dir, FLAGS.size, FLAGS.source_vocab_size)
    else:
        embedding = None

    with tf.Session() as sess:
        # op to write logs to Tensorboard
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False, embedding)
        summary_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                                sess.graph)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)."
              % FLAGS.max_train_data_size)
        dev_set = read_data(evl_content_ids_path, evl_title_ids_path, FLAGS.max_val_data_size)
        train_set = read_data(train_content_ids_path, train_title_ids_path, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        log("Starting training", color="green")
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, summary, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            # todo add summary start
            log("step_loss: %d" %step_loss)
            perplexity_summary = tf.Summary()
            bucket_value = perplexity_summary.value.add()
            bucket_value.tag = "peplexity"
            bucket_value.simple_value = float(step_loss)
            summary_writer.add_summary(perplexity_summary, model.global_step.eval())
            summary_writer.add_summary(summary, model.global_step.eval())
            # todo add summary end

            # summary_writer.add_summary(summary, current_step)
            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                log("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity), color='yellow')
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                # todo add summary start
                perplexity_summary = tf.Summary()
                # todo add summary end

                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)

                    # This is a greedy decoder - outputs are just argmaxes of output_logits.
                    # print("debug train():")
                    # print("shape")
                    # print(output_logits[0].shape) # (64, 40000)

                    outputs = np.concatenate([np.argmax(logit, axis=1).reshape(-1, 1) for logit in output_logits], axis=1)
                    for output in outputs:
                        # If there is an EOS symbol in outputs, cut them at that point.
                        if data_utils.EOS_ID in output:
                            output = output[:list(output).index(data_utils.EOS_ID)]
                        # Print out French sentence corresponding to outputs.
                        print(" ".join([tf.compat.as_str(rev_vocab[word_id]) for word_id in output]))
                    sys.stdout.flush()

                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

                    # todo add summary
                    bucket_value = perplexity_summary.value.add()
                    bucket_value.tag = "peplexity_bucket)%d" % bucket_id
                    bucket_value.simple_value = eval_ppx
                    # todo add summary end


                # todo add summary start
                summary_writer.add_summary(perplexity_summary, model.global_step.eval())
                summary_writer.add_summary(summary, model.global_step.eval())
                summary_writer.add_summary(model.learning_rate.eval(), model.global_step.eval())
                print("model.global_step.eval():", model.global_step.eval())
                # todo add summary end
                sys.stdout.flush()

def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        summary_writer = tf.train.SummaryWriter(FLAGS.summaries_dir, graph=tf.get_default_graph())
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.en" % FLAGS.source_vocab_size)
        fr_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.fr" % FLAGS.target_vocab_size)
        en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
        _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
            # Which bucket does it belong to?
            bucket_id = min([b for b in xrange(len(_buckets))
                             if _buckets[b][0] > len(token_ids)])
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def self_test():
    """Test the translation model."""
    with tf.Session() as sess:
        print("Self-test for neural translation model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                           5.0, 32, 0.3, 0.99, num_samples=8)
        sess.run(tf.initialize_all_variables())

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                       bucket_id, False)

def main(_):
    if FLAGS.self_test:
        print("Self test start")
        self_test()
    elif FLAGS.decode:
        print("Text generation start")
        decode()
    elif FLAGS.preprocessing:
        print("Pre-processing start")
        prepare_headline_generation_data(FLAGS.data_dir, FLAGS.source_vocab_size, FLAGS.oov_size, sentence_truncate=FLAGS.sentence_truncate, tokenizer = spacy_tokenizer)
    else:
        print("Training start")
        train()

if __name__ == "__main__":
    tf.app.run()


