#!/usr/bin/python
# -*- coding:utf-8 -*-


# Portions of this code are adapted from the TensorFlow example materials, which
# retains its original Apache Licenese 2.0.  All other code was developed by
# David Jurgens and Yulia Tsvetkov and has the same license.
#
# Copyright 2017 David Jurgens and The TensorFlow Authors. All Rights Reserved.
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

"""
Equilid: Socially-Equitable Language Identification.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import os.path
import regex as re

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from glob import glob

from os.path import basename

from collections import defaultdict
from collections import Counter

# import seq2seq_model

from random import shuffle
import os.path

import string

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


from tensorflow.python.ops.rnn_cell import DropoutWrapper

from tensorflow.models.rnn.translate import data_utils


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")

tf.app.flags.DEFINE_integer("char_vocab_size", 40000, "Character vocabulary size.")
tf.app.flags.DEFINE_integer("lang_vocab_size", 40000, "Language vocabulary size.")

tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")

# Have the model be loaded from a path relative to where we currently are
cur_dir = os.path.dirname(os.path.abspath(__file__))
tf.app.flags.DEFINE_string("model_dir", cur_dir + "/../models/70lang",
                           "Location of trained classifier.")

tf.app.flags.DEFINE_string("predict_file", None, "File to predict.")
tf.app.flags.DEFINE_string("predict_output_file", None, "File to write predictions to.")

tf.app.flags.DEFINE_string("gpu", "/gpu:0", "GPU Device to use.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Limit on the size of test data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 10000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("use_lstm", False,
                            "Use an LSTM instead of a GRU.")

tf.app.flags.DEFINE_boolean("predict", True,
                            "Set to True for predicting on an unknown set.")

tf.app.flags.DEFINE_boolean("train", False,
                            "Train the model.")


tf.app.flags.DEFINE_string("train_mixed", "",
                            "Alternate between selected datasets and Full training sets.")

# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')



FLAGS = tf.app.flags.FLAGS

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = ['_PAD', '_GO', '_EOS', '_UNK']

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# We use a number of buckets and pad to the closest one for efficiency.
_buckets = [ (60, 11), (100, 26), (140, 36)]


def load_dataset(data_dir, data_type, max_size=0, select=None):
    """Loads the dataset from all sources"""

    # Get all the files of IDs
    files = [f for f in glob(data_dir + '/*' + data_type + '.ids')]    
    
    # The match the source and target files
    prefices = set()
    for f in files:
        prefices.add(basename(f).split(".")[0])
    paired_files = []
    for p in prefices:
        if (select is not None) and (not select in p):
            continue

        src = data_dir + '/' + p + '.source.' + data_type + '.ids'
        tgt = data_dir + '/' + p + '.target.' + data_type + '.ids'
        paired_files.append((src, tgt))
    
    # Everything gets stored here
    data_set = [[] for _ in _buckets]

    # Read in the files
    for (src, tgt) in paired_files:
        print("Loading data from %s" % (src))
        read_data_files(src, tgt, data_set, max_size=max_size)

    # Shuffle the data in each bucket
    for bucket in data_set:
        random.shuffle(bucket)

    size = 0
    for bucket in data_set:
        size += len(bucket)
    print('Loaded %d instances for %s' % (size, data_type))

    return data_set


def read_data_files(source_path, target_path, data_set, max_size=0):
    """Read data from source and target files and put into buckets.
    Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
    (source, target) pairs read from the provided data files that fit
    into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
    len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    counter = 0
    skipped = 0
    
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()

                # The target data sometimes comes with its origin affixed, which
                # we can safely skip 
                if '\t' in target:
                    target = target.split('\t')[0]
                if '\t' in source:
                    source = source.split('\t')[0]

                source_ids = [int(x) + 4 for x in source.split()]
                try:
                    target_ids = [int(x) + 4 for x in target.split()]
                except BaseException as e:
                    print(target)
                    raise e

        
                if len(source_ids) == 0:
                    pass

                target_ids.append(EOS_ID)
                is_placed = False
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size: #and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        is_placed = True
                        break
                if not is_placed:
                    # Just put all the ungainly long instances in the last
                    # bucket so we can handle them.  Ideally, the caller would
                    # have prefiltered their instances so this wouldn't be an
                    # issue, but just in case a few stragglers slipped in, might
                    # as well include them
                    data_set[-1].append([source_ids, target_ids])
                source, target = source_file.readline(), target_file.readline()


    print("Read %d lines, skipped %d instances" % (counter, skipped))
    return data_set


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float32
    model = Seq2SeqModel(
        FLAGS.char_vocab_size,
        FLAGS.lang_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        use_lstm=FLAGS.use_lstm,
        forward_only=forward_only,
        dtype=dtype)

    if FLAGS.train:
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
    else:
        if not tf.gfile.Exists(FLAGS.model_dir):
            print("No model file at %s .  Did you download a model yet?" \
                      % FLAGS.model_dir)
            sys.exit(1)
        print("loading model from %s" % (FLAGS.model_dir))
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.saver.restore(session, ckpt.model_checkpoint_path)


    return model



def train():
    """Train the Equilid Model from character to language-tagged-token data."""

    src_dev_files   = FLAGS.data_dir + '/source.ids.dev'
    src_train_files = FLAGS.data_dir + '/source.ids.train'
    tgt_dev_files   = FLAGS.data_dir + '/target.ids.dev'
    tgt_train_files = FLAGS.data_dir + '/target.ids.train'   

    # Ensure we have a directory to write to
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0, allow_growth=True)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, \
                                          log_device_placement=False, \
                                          device_count = {'GPU': 1}, \
                                          gpu_options=gpu_options)) as sess:

        dev_set = load_dataset(FLAGS.data_dir, 'dev')
        full_train_set = load_dataset(FLAGS.data_dir, 'train', \
                                          FLAGS.max_train_data_size)

        train_set_ = [ full_train_set, ]
        for  dataset_name in FLAGS.train_mixed.split(','):
            print('Loading specifc dataset: %s' % dataset_name)
            train_set_.append(load_dataset(FLAGS.data_dir, 'train', \
                                               FLAGS.max_train_data_size, select=dataset_name))


            num_datasets = len(train_set_)

        train_bucket_sizes_ = [ ]
        train_total_size_ = [ ]
        train_buckets_scale_ = [ ]

        for ii, train_set in enumerate(train_set_):
            train_bucket_sizes_.append([len(train_set[b]) for b in xrange(len(_buckets))])
            train_total_size_.append(float(sum(train_bucket_sizes_[ii])))

            # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
            # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
            # the size if i-th training bucket, as used later.

            train_buckets_scale_.append([sum(train_bucket_sizes_[ii][:i + 1]) / train_total_size_[ii]
                                         for i in xrange(len(train_bucket_sizes_[ii]))])

            

        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        print("Training model")

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        dataset_i = 0
        while True:
            dataset_i += 1

            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale_[dataset_i%num_datasets]))
                             if train_buckets_scale_[dataset_i%num_datasets][i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set_[dataset_i%num_datasets], bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f sec, perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.model_dir, "equilid.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def repair(tokens, predictions):
    """
    Repairs the language prediction sequence if the number of predictions did not
    match the input number of tokens and double-checks that punctuation in the
    input is aligned with the prediction's.  This function is necessary because
    of stochasticity in the LSTM output length and only has much effect
    for very short inputs or very long inputs.
    """

    # If we made a prediction for each token, return that.  NOTE: we could
    # probably do some double checking her to make sure
    # punctiation/hashtag/mention predictions are where they should be
    if len(tokens) == len(predictions):
        return predictions

    # If we only have words (no punctuation), then trunctate to the right number
    # of tokens
    if len(set(predictions)) == 1:
        return predictions[:len(tokens)]
    
    # See how many languages we estimated
    langs = set([x for x in predictions if len(x) == 3])

    # If we only ever saw one language (easy case), then we've just screwed up
    # the number of tokens so iterate over the tokens and fill in the blanks
    if len(langs) == 1:
        lang = list(langs)[0]

        # This is the output set of assignments, based on realignment
        repaired = []    

        # Figure out where we have punctuation in the input
        for i, token in enumerate(tokens):
            if re.fullmatch(r"\p{P}+", token):
                repaired.append('Punct')
            elif re.fullmatch(r"#([\w_]+)", token):
                repaired.append('#Hashtag')
            elif re.fullmatch(r"@([\w_]+)", token):
                repaired.append('@Mention')
            elif (token.startswith('http') and ':' in token) \
                    or token.startswith('pic.twitter'):
                repaired.append('URL')
            else:
                repaired.append(lang)

        # print('%s\n%s\n' % (predictions, repaired))
                
        return repaired

    else:
        # NOTE: the most rigorous thing to do would be a sequence alignment with
        # something like Smith-Waterman and then fill in the gaps, but this is
        # still likely overkill for the kinds of repair operations we expect

        # This is the output set of assignments, based on realignment
        repaired = []    
        n = len(predictions) - 1

        # Figure out where we have non-text stuff in the input as anchor points
        last_anchor = -1
        anchors = []


        rep_anchor_counts  = []
        pred_anchor_counts = []
        
        for pred in predictions:
            prev = 0
            if len(pred_anchor_counts) > 0:
                prev = pred_anchor_counts[-1]
            if len(pred) != 3:
                pred_anchor_counts.append(1 + prev)
            else:
                pred_anchor_counts.append(prev)

        for i, token in enumerate(tokens):
            if re.fullmatch(r"\p{P}+", token):
                repaired.append('Punct')
            elif re.fullmatch(r"#([\w_]+)", token):
                repaired.append('#Hashtag')
            elif re.fullmatch(r"@([\w_]+)", token):
                repaired.append('@Mention')
            elif (token.startswith('http') and ':' in token) \
                    or token.startswith('pic.twitter'):
                repaired.append('URL')
            else:
                repaired.append(None)

        for rep in repaired:
            prev = 0
            if len(rep_anchor_counts) > 0:
                prev = rep_anchor_counts[-1]
            if rep is not None:
                rep_anchor_counts.append(1 + prev)
            else:
                rep_anchor_counts.append(prev)
            

        for i in range(len(repaired)):
            if repaired[i] is not None:
                continue

            try:
                p = pred_anchor_counts[min(i, len(pred_anchor_counts)-1)]
                r = rep_anchor_counts[i]
            except IndexError as xcept:
                print(repr(xcept))
                print(i, len(pred_anchor_counts)-1, min(i, len(pred_anchor_counts)-1))
                continue
            
            nearest_lang = 'UNK'

            if p < r:
                # The prediction has fewer anchors than the repair at this
                # point, which means it added too many things, so skip ahead
                for j in range(i+1, len(predictions)):
                    if pred_anchor_counts[min(j, len(pred_anchor_counts)-1)] >= p:
                        if len(predictions[j]) == 3:
                            nearest_lang = predictions[j]
                            break

            elif p > r:
                # The prediction skipped some input tokens, so rewind until we
                # have the same number of anchors
                for j in range(min(n, i-1), -1, -1):
                    if pred_anchor_counts[min(j, n)] <= p:
                        if len(predictions[min(j, n)]) == 3:
                            nearest_lang = predictions[min(j, n)]
                            break
            else:
                # Just search backwards for a language
                for j in range(min(i, n), -1, -1):
                    if len(predictions[j]) == 3:
                        nearest_lang = predictions[j]
                        break
                    
            # For early tokens that didn't get an assignment from a backwards
            # search, search forward in a limited manner
            if nearest_lang is None:
                for j in range(i+1+anchors[i], min(n+1, i+5+anchors[i])):
                    if len(predictions[j]) == 3:
                        nearest_lang = predictions[j]

            repaired[i] = nearest_lang

        #print('%s\n%s\n' % (predictions, repaired))


        return repaired


cjk_ranges = [
  {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},
  {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},
  {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},
  {"from": ord(u"\U0002F800"), "to": ord(u"\U0002fa1f")},
  {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},        
  {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},        
  {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
  {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
  {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
  {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
  {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
  {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}
]

hangul_ranges = [
  {"from": ord(u"\uAC00"), "to": ord(u"\uD7AF")},
]

def is_cjk(char):
  return any([range["from"] <= ord(char) <= range["to"] for range in cjk_ranges])

def is_hangul(char):
  return any([range["from"] <= ord(char) <= range["to"] for range in hangul_ranges])


CJK_PROXY = str(ord(u"\u4E00"))
HANGUL_PROXY = str(ord(u"\uAC00"))

def to_token_ids(text, char_to_id):
    """
    Converts input text into its IDs based on a defined vocabularly.
    """
    ids = []
    for c in text:
        # The CJK and Hangul_Syllable unicode blocks are each collapsed into
        # single proxy characters since they are primarily used with a single
        # language and, because these blocks are huge, this saves significant
        # space in the model's lookup table.
        if is_cjk(c):
            c = CJK_PROXY
        elif is_hangul(c):
            c = HANGUL_PROXY
        else:
            c = str(ord(c))
        ids.append(char_to_id.get(c, UNK_ID))
    return ids

def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file."""

    # NOTE: the data-to-int conversion uses a +4 offset for indexing due to
    # the starting vocabulary.  We prepend the rev_vocab here to recognize
    # this
    rev_vocab = list(_START_VOCAB)

    with open(vocabulary_path, "rb") as f:
        for line in f:
            rev_vocab.append(line.split("\t")[0].strip())

    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab


def get_langs(text):
    token_langs = classify(text)
    langs = set([x for x in token_langs if len(x) == 3])
    return langs

# The lazily-loaded classifier, which is a tuple of the model
classifier = None

def classify(text):
    """
    
    """
    global classifier
    
    # Ensure the text is always treated as unicode
    text = unicode(text)

    if classifier is None:
        # Prediction uses a small batch size
        FLAGS.batch_size = 1
        load_model()

    # Unpack the classifier into the things we need
    sess, model, char_vocab, rev_char_vocab, lang_vocab, rev_lang_vocab = classifier

    # Convert the input into character IDs
    token_ids = to_token_ids(text, char_vocab)
    # print(token_ids)
                    
    # Which bucket does it belong to?
    possible_buckets = [b for b in xrange(len(_buckets))
                        if _buckets[b][0] > len(token_ids)]
    if len(possible_buckets) == 0:
        # Stick it in the last bucket anyway, even if it's too long.
        # Gotta predict something! #YOLO.  It might be worth logging
        # to the user here if we want to be super paranoid though
        possible_buckets.append(len(_buckets)-1)

    bucket_id = min(possible_buckets)
    # Get a 1-element batch to feed the sentence to the model.
    #
    # NB: Could we speed things up by pushing in multiple instances
    # to a single batch?
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)

    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)
    
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                
    # If there is an EOS symbol in outputs, cut them at that point.
    if EOS_ID in outputs:
        outputs = outputs[:outputs.index(EOS_ID)]
        
    predicted_labels = []
    try:
        predicted_labels = [tf.compat.as_str(rev_lang_vocab[output]) for output in outputs]
    except BaseException as e:
        print(repr(e))
        
    # Ensure we have predictions for each token
    predictions = repair(text.split(), predicted_labels)
    
    return predictions



def load_model():
    global classifier
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # Create model and load parameters.
    model = create_model(sess, True)

    print("Loading vocabs")
    # Load vocabularies.
    char_vocab_path = FLAGS.model_dir + '/vocab.src'
    lang_vocab_path = FLAGS.model_dir + '/vocab.tgt'
    char_vocab, rev_char_vocab = initialize_vocabulary(char_vocab_path)
    lang_vocab, rev_lang_vocab = initialize_vocabulary(lang_vocab_path)
    
    classifier = (sess, model, char_vocab, rev_char_vocab, lang_vocab, rev_lang_vocab)


def predict():
       
    # NB: is there a safer way to do this with a using statement if the file
    # is optionally written to but without having to buffer the output?
    output_file = FLAGS.predict_output_file
    if output_file is not None:
        outf = open(output_file, 'w')
    else:
        outf = None
        
    if FLAGS.predict_file:
        print('Reading data to predict from' + FLAGS.predict_file)
        predict_input = tf.gfile.GFile(FLAGS.predict_file, mode="r")
    else:
        print("No input file specified; reading from STDIN")
        predict_input = sys.stdin

    for i, source in enumerate(predict_input):
        # Strip off newline
        source = source[:-1]

        predictions = classify(source)
                
                
        if outf is not None:
            outf.write(('%d\t%s\t%s\t%s\n' % (i, label, source_text, predicted))\
                           .encode('utf-8'))
            # Since the model can take a while to predict, flush often
            # so the end-user can actually see progress when writing to a file
            if i % 10 == 0:
                outf.flush()
        else:
            print(('Instance %d\t%s\t%s' % \
                       (i, source.encode('utf-8'), ' '.join(predictions))).encode('utf-8'))
                

    if outf is not None:
        outf.close()

        
def set_param(name, val):
    """
    Programmatically sets a variable used in FLAGS.  This method is useful for
    configuring the model if Equilid is being retrained manually via function
    call.
    """
    setattr(FLAGS, name, val)
   
def main(_):
    with tf.device(FLAGS.gpu):
        if FLAGS.train:
            train()
        else:
            predict()

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               use_lstm=False,
               num_samples=512,
               forward_only=False,
               dtype=tf.float32):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w = tf.get_variable("proj_w", [size, self.target_vocab_size], dtype=dtype)
      w_t = tf.transpose(w)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(inputs, tf.float32)
        return tf.cast(
            tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                       num_samples, self.target_vocab_size),
            dtype)
      softmax_loss_function = sampled_loss

    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.nn.rnn_cell.GRUCell(size)
    if use_lstm:
      single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    cell = single_cell
    # Add drop out if we're not making the forward pass
    if not forward_only:
      cell = DropoutWrapper(cell, input_keep_prob=0.8, output_keep_prob=0.8)
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return tf.nn.seq2seq.embedding_attention_seq2seq(
          encoder_inputs,
          decoder_inputs,
          cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=size,
          output_projection=output_projection,
          feed_previous=do_decode,
          dtype=dtype)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[batch_size],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
          ]
    else:
      self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate) # self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))
        print("bucket:%s/%d" % (str(b), len(buckets)))

    self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    #print("step batch_size: %d" % (self.batch_size))

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    #print("batch_size: %d" % (self.batch_size))

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights


if __name__ == "__main__":
  tf.app.run()
