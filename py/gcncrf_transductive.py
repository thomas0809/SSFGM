from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy.sparse as sp

from utils import *
from models import GCN_CRF_transductive
import json
from networkx.readwrite import json_graph
import os
import sys

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'reddit', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'crf', 'Model string.')  # 'gcn', 'crf', 'gcn_crf'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_bool('deep', False, 'Whether use deep factor')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('output', 'pred.txt', 'Output prediction file name.')


TRAIN_RATIO = 0

# Load data


def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples, batchsize):
        end_idx = min(numSamples, start_idx + batchsize)
        if shuffle:
            excerpt = indices[start_idx: end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield [input[excerpt] for input in inputs], excerpt


def loadRedditFromG(dataset_dir, inputfile):
    f= open(dataset_dir+inputfile)
    objects = []
    for _ in range(pkl.load(f)):
        objects.append(pkl.load(f))
    adj, train_labels, val_labels, test_labels, train_index, val_index, test_index = tuple(objects)
    feats = np.load(dataset_dir + "/reddit-feats.npy")
    return sp.csr_matrix(adj), sp.lil_matrix(feats), train_labels, val_labels, test_labels, train_index, val_index, test_index


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']



def transferRedditDataFormat(dataset_dir, output_file):
    G = json_graph.node_link_graph(json.load(open(dataset_dir + "/reddit-G.json")))
    labels = json.load(open(dataset_dir + "/reddit-class_map.json"))

    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n]['test']]
    val_ids = [n for n in G.nodes() if G.node[n]['val']]
    train_labels = [labels[i] for i in train_ids]
    test_labels = [labels[i] for i in test_ids]
    val_labels = [labels[i] for i in val_ids]
    feats = np.load(dataset_dir + "/reddit-feats.npy")
    ## Logistic gets thrown off by big counts, so log transform num comments and score
    feats[:, 0] = np.log(feats[:, 0] + 1.0)
    feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
    feat_id_map = json.load(open(dataset_dir + "reddit-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.iteritems()}

    # train_feats = feats[[feat_id_map[id] for id in train_ids]]
    # test_feats = feats[[feat_id_map[id] for id in test_ids]]

    # numNode = len(feat_id_map)
    # adj = sp.lil_matrix(np.zeros((numNode,numNode)))
    # for edge in G.edges():
    #     adj[feat_id_map[edge[0]], feat_id_map[edge[1]]] = 1

    train_index = [feat_id_map[id] for id in train_ids]
    val_index = [feat_id_map[id] for id in val_ids]
    test_index = [feat_id_map[id] for id in test_ids]
    np.savez(output_file, feats = feats, y_train=train_labels, y_val=val_labels, y_test = test_labels, train_index = train_index,
             val_index=val_index, test_index = test_index)


def transferG2ADJ():
    G = json_graph.node_link_graph(json.load(open("reddit/reddit-G.json")))
    feat_id_map = json.load(open("reddit/reddit-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.iteritems()}
    numNode = len(feat_id_map)
    adj = np.zeros((numNode, numNode))
    newEdges0 = [feat_id_map[edge[0]] for edge in G.edges()]
    newEdges1 = [feat_id_map[edge[1]] for edge in G.edges()]

    # for edge in G.edges():
    #     adj[feat_id_map[edge[0]], feat_id_map[edge[1]]] = 1
    adj = sp.csr_matrix((np.ones((len(newEdges0),)), (newEdges0, newEdges1)), shape=(numNode, numNode))
    sp.save_npz("reddit_adj.npz", adj)


def transferLabel2Onehot(labels, N):
    y = np.zeros((len(labels),N))
    for i in range(len(labels)):
        pos = labels[i]
        y[i,pos] =1
    return y

def construct_feeddict_forMixlayers(features, support, labels, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: features[0].shape})
    return feed_dict


def main(rank1):

    # transferRedditDataFormat("reddit/", 'data/reddit.npz')
    # transferG2ADJ()

    # config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
    #                 inter_op_parallelism_threads = 1,
    #                 intra_op_parallelism_threads = 4,
    #                 log_device_placement=False)


    if FLAGS.dataset == 'reddit':
        adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
        adj = adj+adj.T
        NUM_CLASS = np.max(y_train) + 1
    if FLAGS.dataset in ['twitter_usa', 'twitter_world', 'fb', 'weibo']:
        adj, features, y_train, y_val, y_test, train_index, val_index, test_index, label_names = load_location_data(FLAGS.dataset)
        NUM_CLASS = np.max(y_train) + 1
        if TRAIN_RATIO != 0:
            TRAIN_SIZE = int(train_index.size * TRAIN_RATIO)
            y_train = y_train[:TRAIN_SIZE]
            train_index = train_index[:TRAIN_SIZE]
    else:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, [groundtruth, graph] = load_data_original(FLAGS.dataset, True)
        NUM_CLASS = y_train.shape[1]
        train_index, val_index, test_index = np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0]
        y_train, y_val, y_test = np.where(y_train)[1], np.where(y_val)[1], np.where(y_test)[1]


    numNode = adj.shape[0]
    numNode_train = train_index.size
    numNode_val = val_index.size
    numNode_test = test_index.size
    print(NUM_CLASS)
    print(numNode, numNode_train, numNode_val, numNode_test)

    label = np.zeros(numNode, dtype=int)
    label[train_index] = y_train
    label[val_index] = y_val
    label[test_index] = y_test
    label_onehot = transferLabel2Onehot(label, NUM_CLASS)

    y1 = np.random.choice(NUM_CLASS, numNode)
    y1_mask = np.zeros(numNode, dtype=bool)
    for i in range(y_train.size):
        y1[train_index[i]] = y_train[i]
        y1_mask[train_index[i]] = True
    labeled_index = train_index
    unlabeled_index = np.where(np.logical_not(y1_mask))[0]
    print(unlabeled_index)
    
    y2 = np.array(y1)

    # features = sp.lil_matrix(features)
    normADJ = nontuple_preprocess_adj(adj)
    # eyeADJ = sp.eye(numNode).tocsr()

    # Some preprocessing
    if FLAGS.dataset not in ['twitter_usa', 'twitter_world', 'fb', 'weibo']:
        features = nontuple_preprocess_features(features).todense()
    else:
        features = nontuple_colnorm_features(features)

    CRF = (FLAGS.model == 'crf')

    if not CRF:
        features = normADJ.dot(features)


    # Define placeholders
    placeholders = {
        'support': tf.sparse_placeholder(tf.float32),
        'features': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, NUM_CLASS)),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'nb_y1': tf.placeholder(tf.float32, shape=(None, NUM_CLASS)),
        'nb_y2': tf.placeholder(tf.float32, shape=(None, NUM_CLASS)),
        'y1': tf.placeholder(tf.int32, shape=(None,2)),
        'y2': tf.placeholder(tf.int32, shape=(None,2)),
        'mask': tf.placeholder(tf.bool, shape=(None,))
    }

    # Create model
    model = GCN_CRF_transductive(placeholders, CRF=CRF, logging=True)

    # Initialize session
    sess = tf.Session()

    def calc_nb(y, ADJ, excerpt):
        nb = np.zeros((ADJ.shape[0], NUM_CLASS))
        A = ADJ.tocoo()
        for i,j,v in zip(A.row, A.col, A.data):
            if excerpt[i] == j:
                continue
            nb[i,y[j]] += v
        return nb

    # Define model evaluation function
    def evaluate(ADJ, features, label, y1, placeholders, local=False, final=False):
        t_test = time.time()
        if final:
            output_prob = np.zeros((numNode, NUM_CLASS))
            output_prob[labeled_index] = label[labeled_index]
        if local and not final:
            index = val_index
        else:
            index = unlabeled_index
        MAX_ITER = 2 if CRF else 1
        for i in range(MAX_ITER):
            for batch in iterate_minibatches_listinputs([index], batchsize=512, shuffle=True):
                [excerpt], __ = batch
                ADJ_batch = ADJ[excerpt]
                support = sparse_to_tuple(ADJ_batch)
                features_inputs = features[excerpt] if CRF else features
                if local:
                    feed_dict_val = construct_feeddict_forMixlayers(features_inputs, support, label[excerpt], placeholders)
                    [p1] = sess.run([model.prob_local], feed_dict=feed_dict_val)
                else:
                    nb_y1 = calc_nb(y1, ADJ_batch, excerpt)
                    feed_dict_val = construct_feeddict_forMixlayers(features_inputs, support, label[excerpt], placeholders)
                    feed_dict_val.update({placeholders['nb_y1']: nb_y1})
                    [p1] = sess.run([model.prob1], feed_dict=feed_dict_val)
                new_y = np.argmax(p1, axis=1)
                y1[excerpt] = np.where(y1_mask[excerpt], y1[excerpt], new_y)
                if final:
                    output_prob[excerpt] = p1
        if final:
            f = open(FLAGS.output, 'w')
            if 'label_names' in locals():
                f.write(' '.join(label_names) + '\n')
            for i in range(numNode):
                for j in range(NUM_CLASS):
                    f.write('%.8f '%output_prob[i,j])
                f.write('\n')
            f.close()
        train_acc = 1. # np.mean(np.equal(y1[train_index], np.argmax(label[train_index], axis=1)))
        val_acc = np.mean(np.equal(y1[val_index], np.argmax(label[val_index], axis=1)))
        test_acc = np.mean(np.equal(y1[test_index], np.argmax(label[test_index], axis=1)))
        return train_acc, val_acc, test_acc

    # Init variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    cost_val = []

    p0 = column_prop(normADJ)

    # valSupport = sparse_to_tuple(normADJ_val[numNode_train:, :])
    # testSupport = sparse_to_tuple(normADJ[test_index, :])

    t = time.time()
    maxACC = 0.0
    early_stop_cnt = 0

    # Train model
    for epoch in range(FLAGS.epochs):
        t1 = time.time()

        LOCAL_TRAINING = (FLAGS.model == 'gcn')

        n = 0
        if LOCAL_TRAINING:
            index = labeled_index
        elif numNode_train * 2 < numNode:
            index = np.concatenate([labeled_index, np.random.choice(unlabeled_index, numNode_train)])
        else:
            index = np.arange(numNode)
        for batch in iterate_minibatches_listinputs([index], batchsize=512, shuffle=True):
            [excerpt], __ = batch

            normADJ_batch = normADJ[excerpt]
            batch_size = excerpt.size

            nb_y1 = calc_nb(y1, normADJ_batch, excerpt)
            nb_y2 = calc_nb(y2, normADJ_batch, excerpt)

            # p1 = column_prop(normADJ_batch)
            if CRF or rank1 is None:
                support1 = sparse_to_tuple(normADJ_batch)
                features_inputs = features[excerpt] if CRF else features
            else:
                distr = np.nonzero(np.sum(normADJ_batch, axis=0))[1]
                if rank1 > len(distr):
                    q1 = distr
                else:
                    q1 = np.random.choice(distr, rank1, replace=False, p=p0[distr]/sum(p0[distr]))  # top layer
                # q1 = np.random.choice(np.arange(numNode_train), rank1, p=p0)  # top layer
                support1 = sparse_to_tuple(normADJ_batch[:, q1].dot(sp.diags(1.0 / (p0[q1] * rank1))))
                if len(support1[1])==0:
                    continue
                features_inputs = features[q1, :]  # selected nodes for approximation

            y1_batch = np.stack([range(batch_size), y1[excerpt]], axis=1)
            y2_batch = np.stack([range(batch_size), y2[excerpt]], axis=1)

            # Construct feed dictionary
            feed_dict = construct_feeddict_forMixlayers(features_inputs, support1, label_onehot[excerpt], placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            feed_dict.update({placeholders['nb_y1']: nb_y1})
            feed_dict.update({placeholders['nb_y2']: nb_y2})
            feed_dict.update({placeholders['y1']: y1_batch})
            feed_dict.update({placeholders['y2']: y2_batch})
            feed_dict.update({placeholders['mask']: y1_mask[excerpt]})

            # Training step
            if LOCAL_TRAINING:
                opt = model.local_opt_op
            else:
                opt = model.opt_op
            [_, p1, p2] = sess.run([opt, model.prob1, model.prob2], feed_dict=feed_dict)
            n = n+1

            for i in range(batch_size):
                u = excerpt[i]
                if y1_mask[u] == False:
                    y1[u] = np.random.choice(NUM_CLASS, p=p1[i])
                y2[u] = np.random.choice(NUM_CLASS, p=p2[i])

            if n % 100 != 0: 
                continue

        # Validation
        train_acc, val_acc, test_acc = evaluate(normADJ, features, label_onehot, y1, placeholders, local=LOCAL_TRAINING)

        if val_acc>maxACC:
            maxACC = val_acc
            early_stop_cnt = 0
            saver.save(sess, "tmp/tmp_gcncrf_%s.ckpt" % FLAGS.dataset)
        else:
            early_stop_cnt += 1 

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_acc=", "{:.5f}".format(train_acc), 
            "val_acc=", "{:.5f}".format(val_acc), "test_acc=", "{:.5f}".format(test_acc), 
            "time per batch=", "{:.5f}".format((time.time() - t1)/n))
        sys.stdout.flush()

        if early_stop_cnt > FLAGS.early_stopping:
            # print("Early stopping...")
            break

    train_duration = time.time() - t
    # Testing
    if os.path.exists("tmp/tmp_gcncrf_%s.ckpt.index" % FLAGS.dataset):
        saver.restore(sess, "tmp/tmp_gcncrf_%s.ckpt" % FLAGS.dataset)
    y1[val_index] = y_val
    y1_mask[val_index] = True
    train_acc, val_acc, test_acc = evaluate(normADJ, features, label_onehot, y1, placeholders, local=LOCAL_TRAINING, final=True)
    print("rank1 = {}".format(rank1),
          "accuracy=", "{:.5f}".format(test_acc), "training time=", "{:.5f}".format(train_duration),
          "epoch = {}".format(epoch+1))


    # for u in range(numNode):
    #     if y1[u] != groundtruth[u]:
    #         print(u, y1[u], groundtruth[u])
    #         print(y1[graph[u]], groundtruth[graph[u]])



if __name__=="__main__":
    main(None)
    # main(400)
    # for k in [25, 50, 100, 200, 400]:
    #     main(k)
