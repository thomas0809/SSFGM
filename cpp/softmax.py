from liblinearutil import *
import time
import numpy as np
import pickle
import os
import sys
import argparse
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--thread', type=int, default=1)
parser.add_argument('--directed', action='store_true', default=False)

args = parser.parse_args()

f = file(args.input)
feature2id = {}
features = []
labels = []
label2id = {}
cnt = 0
N = 0
for line in f:
    if line[0] == '#':
        break
    N += 1
    a = line.strip().split('#')[0].split('\t')
    label = a[0][1:]
    if label not in labels:
        label2id[label] = len(labels)
        labels.append(label)
    for x in a[1:]:
        if ':' not in x:
            continue
        token = x.split(':')[0]
        if token not in feature2id:
            feature2id[token] = cnt
            features.append(token)
            cnt += 1
f.close()


print 'num_label', len(labels)

train_x = []
train_y = []
test_x = []
test_y = []
valid_x = []
valid_y = []
all_x = []
all_y = []
all_t = []

graph_in = []
graph_out = []
for i in range(N):
    graph_in.append([])
    graph_out.append([])

f = file(args.input)
line2label = ['']
for line in f:
    if line[0] == '#':
        a = line.split(' ')
        u = int(a[1])
        v = int(a[2])
        graph_out[u].append(v)
        graph_in[v].append(u)
        continue
    a = line.strip().split('\t')
    type_ = a[0][0]
    label = a[0][1:]
    featv = np.zeros(len(feature2id))
    for x in a[1:]:
        if ':' not in x:
            continue
        b = x.split(':')
        fid = feature2id[b[0]]
        val = float(b[1])
        featv[fid] = val
    featv = list(featv)
    label = label2id[label]
    if type_ == '+':
        train_x.append(featv)
        train_y.append(label)
    if type_ == '*':
        valid_x.append(featv)
        valid_y.append(label)
    if type_ == '?':
        test_x.append(featv)
        test_y.append(label)
    all_x.append(featv)
    all_y.append(label)
    all_t.append(type_)


starttime = time.time()

print 'start training ...'
prob = problem(train_y, train_x)
param = parameter('-s 0 -c 1 -n %d -q' % args.thread)
clf = train(prob, param)
save_model('liblinear_LR.model', clf)

print 'start testing ...'
print 'valid',
y_v, p_acc, p_vals = predict(valid_y, valid_x, clf, '-b 1')
print 'test',
y_t, p_acc, p_vals = predict(test_y, test_x, clf, '-b 1')
sys.stdout.flush()

print 'valid', accuracy_score(valid_y, y_v)
print 'test', accuracy_score(test_y, y_t)

best_valid = 0
best_clf = clf
C = clf.get_nr_class()

for iter in range(10):

    pred_y = []
    cnt_t = 0
    cnt_v = 0
    for i in range(len(all_x)):
        if all_t[i] == '+':
            pred_y.append(all_y[i])
        if all_t[i] == '*':
            pred_y.append(int(y_v[cnt_v]))
            cnt_v += 1
        if all_t[i] == '?':
            pred_y.append(int(y_t[cnt_t]))
            cnt_t += 1

    train_x = []
    test_x = []
    valid_x = []
    new_all_x = []

    for i in range(len(all_x)):
        feat1 = np.zeros(C)
        feat2 = np.zeros(C)
        for v in graph_out[i]:
            yv = pred_y[v]
            feat1[yv] += 1
        for v in graph_in[i]:
            yv = pred_y[v]
            feat2[yv] += 1
        if args.directed:
            feat = all_x[i] + list(feat1) + list(feat2)
        else:
            feat = all_x[i] + list(feat1 + feat2)
        if all_t[i] == '+':
            train_x.append(feat)
        if all_t[i] == '*':
            valid_x.append(feat)
        if all_t[i] == '?':
            test_x.append(feat)
        new_all_x.append(feat)

    print 'start training ...'
    prob = problem(train_y, train_x)
    param = parameter('-s 0 -c 1 -n %d -q' % args.thread)
    clf = train(prob, param)

    print 'start testing ...'
    print 'valid',
    y_v, valid_acc, p_vals = predict(valid_y, valid_x, clf, '-b 1')
    print 'test',
    y_t, test_acc, p_vals = predict(test_y, test_x, clf, '-b 1')
    sys.stdout.flush()

    if valid_acc > best_valid:
        best_valid = valid_acc
        print 'all',
        y, p_acc, y_proba = predict(all_y, new_all_x, clf, '-b 1')
    else:
        break

endtime = time.time()
print 'Time:', endtime - starttime
sys.stdout.flush()

output = file('pred_SR.txt', 'w')
classes_ = clf.get_labels()
for c in classes_:
    output.write(labels[c] + ' ')
output.write('\n')
for i in range(len(y_proba)):
    for j in range(len(classes_)):
        output.write('%.3f ' % y_proba[i][j])
    output.write('\n')
output.close()

output = file('param.txt', 'w')

w = []
b = []
C = clf.get_nr_class()
F = clf.get_nr_feature()
for i in range(C):
    wi, bi = clf.get_decfun(i)
    if F != len(new_all_x[0]):
        wi += [0] * (len(new_all_x[0]) - F)
    w.append(np.asarray(wi))
    b.append(bi)

for i in range(C):
    for j in range(len(features)):
        output.write('#node %s %s %f\n' % (labels[classes_[i]], features[j], w[i][j]))
    for j in range(C):
        output.write('#edge mt %s %s %f\n' % (labels[classes_[i]], labels[classes_[j]], w[i][len(features) + j]))

output.close()


