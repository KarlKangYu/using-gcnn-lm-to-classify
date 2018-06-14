from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import config
from pru import PRUCell

FLAGS = config.FLAGS
data_type = config.data_type
index_data_type = config.index_data_type
np_index_data_type = config.np_index_data_type


class WordModel(object):
    """Static PTB model. Modified from old saniti-checked version of dynamic model.
    """
    def __init__(self, is_training, config):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.embedding_size = config.word_embedding_size
        self.hidden_size = config.word_hidden_size
        self.vocab_size_in = config.vocab_size_in
        self.vocab_size_out = config.vocab_size_out
        self.filter_width = 6
        self.vocab_size_phrase = config.vocab_size_phrase
        self.input_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None], name="batched_input_word_ids")
        self.target_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None], name="batched_output_word_ids")
        self.output_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None], name="batched_output_word_masks")
        self.sequence_length = tf.placeholder_with_default(input=tf.fill(dims=[self.batch_size], value=self.num_steps),
                                                           shape=[self.batch_size], name="batched_input_sequence_length")
        self.top_k = tf.placeholder(dtype=index_data_type(), shape=[], name="top_k")

        self.target_phrase_p = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                          name="batched_output_phrase_p_ids")
        self.target_phrase_p_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                           name="batched_output_phrase_p_masks")
        self.target_phrase_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                                  name="batched_output_phrase_ids")
        self.target_phrase_data_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                           name="batched_output_phrase_masks")
        self.target_phrase_logits_masks = tf.placeholder_with_default(
            tf.ones([self.batch_size * self.num_steps, self.vocab_size_phrase], dtype=data_type()),
            [self.batch_size * self.num_steps, self.vocab_size_phrase], name="batched_output_phrase_logits_masks")

        with tf.variable_scope("Lm"):
            with tf.variable_scope("Embedding"):
                self._embedding = tf.get_variable("embedding", [self.vocab_size_in, self.embedding_size], dtype=data_type())
                inputs = tf.nn.embedding_lookup(self._embedding, self.input_data)
                embedding_to_rnn = tf.get_variable("embedding_to_rnn",
                                                  [self.embedding_size, self.hidden_size],
                                                  dtype=data_type())
                inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.embedding_size]),
                                          embedding_to_rnn),
                                shape=[self.batch_size, -1, self.hidden_size])

                if is_training and config.keep_prob < 1:
                    inputs = tf.nn.dropout(inputs, config.keep_prob)
            
            # The GCNN layer is defined as
            # def gated_cnn_layer(self, inputs, filter_width, output_dim, layer_num):
            gcnn1 = self.gated_cnn_layer(inputs, self.filter_width, self.hidden_size, 1)
            gcnn2 = self.gated_cnn_layer(gcnn1, self.filter_width, self.hidden_size, 2)
            gcnn3 = self.gated_cnn_layer(gcnn2, self.filter_width, self.hidden_size, 3) + gcnn1
            gcnn4 = self.gated_cnn_layer(gcnn3, self.filter_width, self.hidden_size, 4)
            gcnn5 = self.gated_cnn_layer(gcnn4, self.filter_width, self.hidden_size, 5)
            gcnn6 = self.gated_cnn_layer(gcnn5, self.filter_width, self.hidden_size, 6) + gcnn3
            #gcnn7 = self.gated_cnn_layer(gcnn6, self.filter_width, self.hidden_size, 7)
            #gcnn8 = self.gated_cnn_layer(gcnn7, self.filter_width, self.hidden_size, 8)e
            #gcnn9 = self.gated_cnn_layer(gcnn8, self.filter_width, self.hidden_size, 9) + gcnn6
            #gcnn10 = self.gated_cnn_layer(gcnn9, self.filter_width, self.hidden_size, 10)
            #gcnn11 = self.gated_cnn_layer(gcnn10, self.filter_width, self.hidden_size, 11)
            #gcnn12 = self.gated_cnn_layer(gcnn11, self.filter_width, self.hidden_size, 12) + gcnn9
            
            print ("the final output of GCNN network be shape:", tf.shape(gcnn6)) #gcnn12.get_shape().as_list()

            # output as the final layer (GCNN structure), noted as the final state also as input of the softmax  
            # check the dimension of the output
            # gcnn12 be shape [batch_size, num_steps, hidden_size]
            cnn_output = tf.reshape(gcnn6, [-1, self.hidden_size])


            state_output = tf.expand_dims(tf.expand_dims(gcnn6, axis=1), axis=1)
            #states = tf.transpose(state_output, perm=[3, 1, 2, 0, 4])
            unstack_states = tf.unstack(state_output, axis=0)
            cnn_state = tf.concat(unstack_states, axis=2)

            state = gcnn6[:,-1,:]
            state = tf.expand_dims(tf.expand_dims(state, axis=0), axis=0)

            print("output shape:", cnn_output.shape)
            print("state shape:", cnn_state.shape)

            with tf.variable_scope("Softmax"):
                cnn_output_to_final_output = tf.get_variable("cnn_output_to_final_output",
                                                                    [self.hidden_size, self.embedding_size],
                                                                    dtype=data_type())
                self._softmax_w = tf.get_variable("softmax_w", [self.embedding_size, self.vocab_size_out],
                                                  dtype=data_type())
                softmax_b = tf.get_variable("softmax_b", [self.vocab_size_out], dtype=data_type())
        
        with tf.variable_scope("PhraseProb"):
            self._softmax_phrase_p_w = tf.get_variable("softmax_phrase_p_w", [self.embedding_size, 2],
                                                           dtype=data_type())
            softmax_phrase_p_b = tf.get_variable("softmax_phrase_p_b", [2], dtype=data_type())
        
        with tf.variable_scope("Phrase"):
            self._softmax_phrase_w = tf.get_variable("softmax_phrase_w",
                                                     [self.embedding_size, self.vocab_size_phrase],
                                                     dtype=data_type())
            softmax_phrase_b = tf.get_variable("softmax_phrase_b", [self.vocab_size_phrase], dtype=data_type())

        self._logits_phrase_p = logits_phrase_p = tf.matmul(tf.matmul(cnn_output, cnn_output_to_final_output),
                                                                    self._softmax_phrase_p_w) + softmax_phrase_p_b
        
        self._phrase_p_probabilities = tf.nn.softmax(logits_phrase_p, name="phrase_p_probabilities")
        _, phrase_p_prediction = tf.nn.top_k(logits_phrase_p, 2, name="phrase_p_prediction")
        logits_phrase = (tf.matmul(tf.matmul(cnn_output, cnn_output_to_final_output),
                                   self._softmax_phrase_w) + softmax_phrase_b) * self.target_phrase_logits_masks
        self._logits_phrase = tf.identity(logits_phrase, name="logits_phrase")
        self._phrase_probabilities = tf.nn.softmax(logits_phrase, name="phrase_probabilities")
        _, self._phrase_prediction = tf.nn.top_k(logits_phrase, self.top_k, name="phrase_top_k_prediction")
        loss_phrase_p = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits_phrase_p],
                                                                               [tf.reshape(self.target_phrase_p,
                                                                                           [-1])],
                                                                               [tf.reshape(
                                                                                   self.target_phrase_p_masks,
                                                                                   [-1])],
                                                                               average_across_timesteps=False)
        loss_phrase = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits_phrase],
                                                                         [tf.reshape(self.target_phrase_data,
                                                                                     [-1])],
                                                                         [tf.reshape(self.target_phrase_data_masks,
                                                                                     [-1])],
                                                                         average_across_timesteps=False)
        self._phrase_p_cost = phrase_p_cost = tf.reduce_sum(loss_phrase_p)

        self._phrase_cost = phrase_cost = tf.reduce_sum(loss_phrase)

        logits = tf.matmul(tf.matmul(cnn_output, cnn_output_to_final_output),
                            self._softmax_w) + softmax_b

        probabilities = tf.nn.softmax(logits, name="probabilities")
        _, top_k_prediction = tf.nn.top_k(logits, self.top_k, name="top_k_prediction")

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(self.target_data, [-1])],
                                                                  [tf.reshape(self.output_masks, [-1])],
                                                                  average_across_timesteps=False)

        self._cost = cost = tf.reduce_sum(loss)

        self._final_state = tf.identity(state, "state_out")
        self._cnn_state = tf.identity(cnn_state, "cnn_state")
        self._logits = logits
        self._probabilities = probabilities
        self._top_k_prediction = top_k_prediction

        if not is_training:
            return

        self._lr = tf.get_variable(name="learning_rate", shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(config.learning_rate), trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/Lm")
        tvars_phrase_p = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/PhraseProb")
        tvars_phrase = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/Phrase")

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        grads_phrase_p, _ = tf.clip_by_global_norm(tf.gradients(phrase_p_cost, tvars_phrase_p), config.max_grad_norm)
        optimizer_phrase_p = tf.train.AdamOptimizer(0.001)
        self._train_op_phrase_p = optimizer_phrase_p.apply_gradients(zip(grads_phrase_p, tvars_phrase_p),
                                                                             global_step=self.global_step)

        grads_phrase, _ = tf.clip_by_global_norm(tf.gradients(phrase_cost, tvars_phrase), config.max_grad_norm)
        optimizer_phrase = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op_phrase = optimizer_phrase.apply_gradients(zip(grads_phrase, tvars_phrase),
                                                                 global_step=self.global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def gated_cnn_layer(self, inputs, filter_width, output_dim, layer_num):
        #inputs = tf.convert_to_tensor(inputs)
        shape_input = inputs.get_shape().as_list()

        paddings = [[0,0], [0, filter_width-1], [0,0]]
        inputs = tf.pad(inputs, paddings, "CONSTANT")
        inputs = tf.expand_dims(inputs, axis = 3)

        shape = [filter_width, shape_input[2], 1, output_dim]
        with tf.variable_scope("GCNN"):
            wc = tf.get_variable("%d_wc"%layer_num, shape, tf.float32, tf.random_normal_initializer(0.0, 0.5))
            bc = tf.get_variable("%d_bc"%layer_num, shape[-1], tf.float32, tf.random_normal_initializer(0.0, 0.001))
            gwc = tf.get_variable("%d_gwc"%layer_num, shape, tf.float32, tf.random_normal_initializer(0.0, 0.5))
            gbc = tf.get_variable("%d_gbc"%layer_num, shape[-1], tf.float32, tf.random_normal_initializer(0.0, 0.001))

        conv1 = tf.add(tf.nn.conv2d(inputs, wc, strides = [1,1,1,1], padding = 'VALID'), bc)
        conv2 = tf.add(tf.nn.conv2d(inputs, gwc, strides = [1,1,1,1], padding = 'VALID'), gbc)
        #conv = conv1 * tf.sigmoid(conv2)
        conv = tf.tanh(conv1) * tf.sigmoid(conv2)
        final = tf.squeeze(conv, axis = 2)
        return final

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def get_global_step(self, session):
        gs = session.run(self.global_step)
        return gs

    #@property
    #def initial_state(self):
    #    return self._initial_state

    @property
    def softmax_w(self):
        return self._softmax_w

    @property
    def cost(self):
        return [self._cost, self._phrase_p_cost, self._phrase_cost]

    @property
    def embedding(self):
        return self._embedding

    @property
    def final_state(self):
        return self._final_state

    @property
    def rnn_state(self):
        return self._cnn_state

    @property
    def lr(self):
        return self._lr

    @property
    def logits(self):
        return [self._logits, self._logits_phrase_p, self._logits_phrase]

    @property
    def probalities(self):
        return [self._probabilities, self._phrase_p_probabilities,self._phrase_probabilities]

    @property
    def top_k_prediction(self):
        return [self._top_k_prediction, self._phrase_prediction]

    @property
    def train_op(self):
        return [self._train_op, self._train_op_phrase_p, self._train_op_phrase]


class LetterModel(object):
    """Static PTB model. Modified from old saniti-checked version of dynamic model.
    """

    def __init__(self, is_training, config):
        self.num_steps = config.num_steps
        self.batch_size = config.batch_size * config.num_steps
        self.max_word_length = config.max_word_length
        self.embedding_size = config.letter_embedding_size
        self.hidden_size = config.letter_hidden_size
        self.vocab_size_in = config.vocab_size_letter
        self.vocab_size_out = config.vocab_size_out

        self.input_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                         name="batched_input_word_ids")
        self.target_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                          name="batched_output_word_ids")
        self.output_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                           name="batched_output_word_masks")
        self.sequence_length = tf.placeholder_with_default(input=tf.fill(dims=[self.batch_size], value=self.max_word_length),
                                                           shape=[self.batch_size],
                                                           name="batched_input_sequence_length")
        self.top_k = tf.placeholder(dtype=index_data_type(), shape=[], name="top_k")

        def pru_cell():
            return PRUCell(self.hidden_size)

        attn_cell = pru_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(pru_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)])

        lm_state_as_tensor_shape = [config.num_layers, 1, self.batch_size, config.word_hidden_size]
        letter_state_as_tensor_shape = [config.num_layers, 1, self.batch_size, config.letter_hidden_size]

        self.lm_state_in = tf.placeholder_with_default(tf.zeros(lm_state_as_tensor_shape, dtype=data_type()),
                                                          lm_state_as_tensor_shape, name="lm_state_in")
        with tf.variable_scope("StateMatrix"):
            lm_state_to_letter_state = tf.get_variable("lm_state_to_letter_state",
                                                       [config.word_hidden_size, config.letter_hidden_size],dtype=data_type())

        if config.word_hidden_size != config.letter_hidden_size:
            self._initial_state = tf.placeholder_with_default(
                                tf.reshape(tf.matmul(tf.reshape(self.lm_state_in, [-1, config.word_hidden_size]), lm_state_to_letter_state),
                                           letter_state_as_tensor_shape),
                                letter_state_as_tensor_shape, name="state")
        else:
            self._initial_state = tf.placeholder_with_default(
                self.lm_state_in, letter_state_as_tensor_shape, name="state")

        unstack_state = tf.unstack(self._initial_state, axis=0)
        
        #tuple_state = tuple(
        #    [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1])
        #     for idx in range(config.num_layers)]
        #)

        tuple_state = tuple(
            [unstack_state[idx][0] for idx in range(config.num_layers)]
        ) 
        # initial state is a tuple of (number of layers * tensor be shape [batch_size, hidden_size])

        with tf.variable_scope("Embedding"):

            self._embedding = tf.get_variable("embedding", [self.vocab_size_in, self.embedding_size],
                                                  dtype=data_type())

            inputs = tf.nn.embedding_lookup(self._embedding, self.input_data)
            embedding_to_rnn = tf.get_variable("embedding_to_rnn",
                                               [self.embedding_size, self.hidden_size],
                                               dtype=data_type())
            inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.embedding_size]),
                                          embedding_to_rnn),
                                shape=[self.batch_size, -1, self.hidden_size])

            if is_training and config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.keep_prob)

        with tf.variable_scope("RNN"):
            # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
            outputs, state_out = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.sequence_length,
                                                                 initial_state=tuple_state)

        output = tf.reshape(outputs, [-1, self.hidden_size])
        with tf.variable_scope("Softmax"):
            cnn_output_to_final_output = tf.get_variable("cnn_output_to_final_output",
                                                         [self.hidden_size, self.embedding_size],
                                                         dtype=data_type())
            self._softmax_w = tf.get_variable("softmax_w", [self.embedding_size, self.vocab_size_out],
                                              dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size_out], dtype=data_type())

        logits = tf.matmul(tf.matmul(output, cnn_output_to_final_output),
                           self._softmax_w) + softmax_b

        probabilities = tf.nn.softmax(logits, name="probabilities")
        _, top_k_prediction = tf.nn.top_k(logits, self.top_k, name="top_k_prediction")

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(self.target_data, [-1])],
                                                                  [tf.reshape(self.output_masks, [-1])],
                                                                  average_across_timesteps=False)

        self._cost = cost = tf.reduce_sum(loss)

        self._final_state = tf.identity(state_out, "state_out")
        self._logits = logits
        self._probabilities = probabilities
        self._top_k_prediction = top_k_prediction

        if not is_training:
            return

        self._lr = tf.get_variable(name="learning_rate", shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(config.learning_rate), trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LetterModel")

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def get_global_step(self, session):
        gs = session.run(self.global_step)
        return gs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def softmax_w(self):
        return self._softmax_w

    @property
    def cost(self):
        return self._cost

    @property
    def embedding(self):
        return self._embedding

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def logits(self):
        return self._logits

    @property
    def probalities(self):
        return self._probabilities

    @property
    def top_k_prediction(self):
        return self._top_k_prediction

    @property
    def train_op(self):
        return self._train_op