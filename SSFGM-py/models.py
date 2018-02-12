from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)




class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)



class GCN_APPRO(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_APPRO, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.supports = placeholders['support']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])

    def _build(self):
        # appr_support = self.placeholders['support'][0]
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            support=self.supports[0],
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support=self.supports[1],
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN_APPRO_Mix(Model): #mixture of dense and gcn
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_APPRO_Mix, self).__init__(**kwargs)
        self.inputs = placeholders['AXfeatures']# A*X for the bottom layer, not original feature X
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.support = placeholders['support']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])


    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support=self.support,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)



class GCN_APPRO_Onelayer(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_APPRO_Onelayer, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.supports = placeholders['support']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        appr_support = self.placeholders['support'][0]
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support=self.supports[0],
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)



class GCN_CRF(Model): 
    def __init__(self, placeholders, **kwargs):
        super(GCN_CRF, self).__init__(**kwargs)
        self.inputs = placeholders['AXfeatures']# A*X for the bottom layer, not original feature X
        self.batch_size = self.inputs.get_shape().as_list()[0]
        self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.support = placeholders['support']
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))
        # print('hidden', FLAGS.hidden1)
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support=self.support,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

        self.crf_layer = Dense(input_dim=self.output_dim,
                               output_dim=self.output_dim,
                               placeholders=self.placeholders,
                               act=tf.identity,
                               logging=self.logging)

    def build(self):
        """ Wrapper for _build() """
        reg = tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
        with tf.variable_scope(self.name, regularizer=reg):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.fx = self.activations[-1]

        #self.output1 = self.crf_layer(tf.concat([self.fx, self.placeholders['nb_y1']], 1))
        self.output1 = self.fx + self.crf_layer(self.placeholders['nb_y1'])
        self.output2 = self.fx + self.crf_layer(self.placeholders['nb_y2'])

        self.prob1 = tf.nn.softmax(self.output1)
        self.prob2 = tf.nn.softmax(self.output2)
        self.prob_local = tf.nn.softmax(self.fx)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # loss = tf.gather_nd(self.output2, self.placeholders['y2']) - tf.gather_nd(self.output1, self.placeholders['y1'])
        # self.loss = tf.reduce_mean(loss) #+ tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        self.accuracy = accuracy(self.output1, self.placeholders['labels'])

        self.cross_entropy = softmax_cross_entropy(self.output1, self.placeholders['labels'])
        loss2 = -tf.gather_nd(self.output1, self.placeholders['y1']) + tf.reduce_logsumexp(self.output2, axis=1)
        self.loss = tf.reduce_mean(loss2)

        self.opt_op = self.optimizer.minimize(self.loss)
        self.approx_opt_op = self.optimizer.minimize(self.cross_entropy)



class GCN_CRF_transductive(Model): 
    def __init__(self, placeholders, CRF=False, **kwargs):
        super(GCN_CRF_transductive, self).__init__(**kwargs)
        self.features = placeholders['features']# A*X for the bottom layer, not original feature X
        self.batch_size = self.features.get_shape().as_list()[0]
        self.input_dim = self.features.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        # self.index = placeholders['index']
        self.support = placeholders['support']
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.CRF = CRF
        self.build()

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=0.,
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support=self.support,
                                            act=lambda x: x,
                                            dropout=0.,
                                            concat=False,
                                            logging=self.logging))

        self.crf_layer = Dense(input_dim=self.output_dim,
                               output_dim=self.output_dim,
                               placeholders=self.placeholders,
                               act=tf.identity,
                               logging=self.logging)

        self.wide = Dense(input_dim=self.input_dim,
                          output_dim=self.output_dim,
                          placeholders=self.placeholders,
                          act=tf.identity)

        self.deep0 = Dense(input_dim=self.input_dim,
                           output_dim=FLAGS.hidden1,
                           bias=True,
                           placeholders=self.placeholders)
        self.deep1 = Dense(input_dim=FLAGS.hidden1,
                           output_dim=FLAGS.hidden2,
                           bias=True,
                           placeholders=self.placeholders)
        self.deep2 = Dense(input_dim=FLAGS.hidden2,
                           output_dim=self.output_dim,
                           placeholders=self.placeholders,
                           act=tf.identity)

    def build(self):
        """ Wrapper for _build() """
        regularizer = tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
        with tf.variable_scope(self.name, regularizer=regularizer):
            self._build()

        # Build sequential layer model
        if not self.CRF:
            self.activations.append(self.features)
            for layer in self.layers:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
            self.fx = self.activations[-1]
        else:
            self.fx = self.wide(self.features) 
            if FLAGS.deep:
                self.fx += self.deep2(self.deep1(self.deep0(self.features)))

        #self.output1 = self.crf_layer(tf.concat([self.fx, self.placeholders['nb_y1']], 1))
        self.output1 = self.fx + self.crf_layer(self.placeholders['nb_y1'])
        self.output2 = self.fx + self.crf_layer(self.placeholders['nb_y2'])

        self.prob1 = tf.nn.softmax(self.output1)
        self.prob2 = tf.nn.softmax(self.output2)
        self.prob_local = tf.nn.softmax(self.fx)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self.accuracy = accuracy(self.output1, self.placeholders['labels'])

        self.local_loss = softmax_cross_entropy(self.fx, self.placeholders['labels'])

        loss = - tf.where(self.placeholders['mask'], tf.gather_nd(self.output1, self.placeholders['y1']), tf.reduce_logsumexp(self.output1, axis=1)) + tf.reduce_logsumexp(self.output2, axis=1) 
        self.loss = tf.reduce_mean(loss)

        # self.loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # # Weight decay loss
        # if FLAGS.deep:
        #     for layer in [self.deep0, self.deep1]:
        #         for var in layer.vars.values():
        #             self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.opt_op = self.optimizer.minimize(self.loss)
        self.local_opt_op = self.optimizer.minimize(self.local_loss)



