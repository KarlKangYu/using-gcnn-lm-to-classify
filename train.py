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

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import os
import random

import sys

from tensorflow.python.framework.graph_util import convert_variables_to_constants
from gcnn_model import WordModel
from config import Config
import config
import data_feeder as data_feeder
import os


#os.environ['CUDA_VISIBLE_DEVICES']='3'


FLAGS = config.FLAGS
data_type = config.data_type
index_data_type = config.index_data_type
np_index_data_type = config.np_index_data_type

def export_graph(session, iter, phase="lm"):
    if phase == "lm":
        # Export variables related to language model only
        variables_to_export = ["Online/WordModel/probabilities",
                               "Online/WordModel/state_out",
                               "Online/WordModel/top_k_prediction",
                               "Online/WordModel/cnn_out"
                               ]
    elif phase == "kc_full":
        variables_to_export = ["Online/WordModel/probabilities",
                               "Online/WordModel/state_out",
                               "Online/WordModel/top_k_prediction",
                               "Online/WordModel/cnn_out"
                               ]
    else:
        assert phase == "kc_slim"
        variables_to_export = ["Online/WordModel/state_out",
                               "Online/WordModel/cnn_out"
                               ]

    graph_def = convert_variables_to_constants(session, session.graph_def, variables_to_export)
    config_name = FLAGS.model_config
    model_export_path = os.path.join(FLAGS.graph_save_path)
    if not os.path.isdir(model_export_path):
        os.makedirs(model_export_path)
    model_export_name = os.path.join(model_export_path,
                                     config_name[config_name.rfind("/")+1:] + "-iter" + str(iter) + "-" + phase + '.pb')
    f = open(model_export_name, "wb")
    f.write(graph_def.SerializeToString())
    f.close()
    print("Graph is saved to: ", model_export_name)


def run_word_epoch(session, data, word_model, config, lm_phase_id, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    num_word = 0
    fetches = {}

    for step, (epoch_size, lm_data) in \
            enumerate(data_feeder.data_iterator(data, config)):
        if FLAGS.laptop_discount > 0 and step >= FLAGS.laptop_discount:
            break
        if step >= epoch_size:
            break

        fetches["cost"] = word_model.cost
        fetches["final_state"] = word_model.final_state

        if eval_op is not None:
            fetches["eval_op"] = eval_op[lm_phase_id]

        feed_dict = {word_model.input_data: lm_data[0],
                     word_model.target_data: lm_data[1],
                     word_model.output_masks: lm_data[2],
                     word_model.sequence_length: lm_data[3]
                     }
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]

        costs += cost[0]
        iters += np.sum(lm_data[2])

        num_word += np.sum(lm_data[3])
        if verbose and step % (epoch_size // 100) == 0:
            print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] "
                "%.3f word ppl: %.3f speed: %.0f wps"
                  % (step * 1.0 / epoch_size, np.exp(costs / iters),
                     num_word / (time.time() - start_time)))
            sys.stdout.flush()
    all_costs = np.exp(costs / iters)
    return all_costs


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    logfile = open(FLAGS.model_config + '.log', 'w')

    config = Config()
    config.get_config(FLAGS.vocab_path, FLAGS.model_config)

    test_config = Config()
    test_config.get_config(FLAGS.vocab_path, FLAGS.model_config)
    test_config.batch_size = 1
    test_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        # gpu_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_fraction
        train_data = data_feeder.read_file(FLAGS.data_path, config, is_train=True)
        valid_data = data_feeder.read_file(FLAGS.data_path, config, is_train=False)
        print("in words vocabulary size = %d\nout words vocabulary size = %d\nin letters vocabulary size = %d"
              "\nphrase vocabulary size = %d" % (
                  config.vocab_size_in, config.vocab_size_out, config.vocab_size_letter,
                  config.vocab_size_phrase))

        with tf.Session(config=gpu_config) as session:
            with tf.name_scope("Train"):

                with tf.variable_scope("WordModel", reuse=False, initializer=initializer):
                    mtrain = WordModel(is_training=True, config=config)
                    train_op = mtrain.train_op

            with tf.name_scope("Valid"):

                with tf.variable_scope("WordModel", reuse=True, initializer=initializer):
                    mvalid = WordModel(is_training=False, config=config)

            with tf.name_scope("Online"):

                with tf.variable_scope("WordModel", reuse=True, initializer=initializer):
                    monline = WordModel(is_training=False, config=test_config)

            restore_variables = dict()
            for v in tf.trainable_variables():
                print("store:", v.name)

                restore_variables[v.name] = v

            sv = tf.train.Saver(restore_variables)

            if not FLAGS.model_name.endswith(".ckpt"):
                FLAGS.model_name += ".ckpt"

            session.run(tf.global_variables_initializer())

            check_point_dir = os.path.join(FLAGS.save_path)
            ckpt = tf.train.get_checkpoint_state(check_point_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                sv.restore(session, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")

            print("training language model.")
            print("training language model", file=logfile)

            save_path = os.path.join(FLAGS.save_path)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            for lm_phase_id in range(1):
                # if lm_phase_id != 1:
                #     continue
                # ln_phase_id = [0,1,2], representing lm_train_phase, phrase_prob_train_phase and phrase_train_phase
                print("lm training phase: %d" % (lm_phase_id + 1), file=logfile)
                if lm_phase_id != 1:
                    max_train_epoch = config.max_max_epoch
                else:
                    max_train_epoch = 2
                # the max_train_epoch of phrase_prob_train_phase is 2, it is enough.
                for i in range(max_train_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)
                    print("lm training phase: %d" % (lm_phase_id + 1))
                    mtrain.assign_lr(session, config.learning_rate * lr_decay)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)), file=logfile)
                    train_perplexity = run_word_epoch(session, train_data, mtrain, config, lm_phase_id, train_op,
                                                      verbose=True)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity), file=logfile)
                    logfile.flush()

                    valid_perplexity = run_word_epoch(session, valid_data, mvalid, config, lm_phase_id)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity), file=logfile)
                    logfile.flush()

                    if FLAGS.save_path:
                        print("Saving model to %s." % FLAGS.save_path, file=logfile)
                        step = mtrain.get_global_step(session)
                        model_save_path = os.path.join(save_path, FLAGS.model_name)
                        sv.save(session, model_save_path, global_step=step)
                    print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin exporting lm graph!")
                    export_graph(session, i, phase="lm")
                    export_graph(session, i, phase="kc_slim")
                    print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish exporting lm graph!")

            logfile.close()


if __name__ == "__main__":
    tf.app.run()




