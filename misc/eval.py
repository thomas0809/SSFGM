import sys
import operator
import json
from math import radians, cos, sin, asin, sqrt
#from make_map import make_map

def distance(coordu, coordv):
    lon1, lat1, lon2, lat2 = map(radians, [coordu[1], coordu[0], coordv[1], coordv[0]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

# Load coords
with open(sys.argv[3], 'r') as f:
    coords = json.load(f)

label = []
ltype = []

f = file(sys.argv[1])

# Load real states
if 'uids' in sys.argv[1]:
    for line in f:
        a = line.strip().split(' ')
        label.append(a[2])
        ltype.append(a[0])
else:
    for line in f:
        if line[0] == '#':
            break
        a = line.split('\t', 1)[0]
        label.append(a[1:])
        ltype.append(a[0])

f.close()

label_set = set(label)
M = len(label_set)

lbl_cnt = {lb:0 for lb in label_set}
for i in range(len(label)):
    if ltype[i] != '?':
        lbl_cnt[label[i]] += 1


hit = [0] * len(label_set)
test = 0

# Load predictions
#model_name = sys.argv[2].split('.')[0].split('_')[1]
f = file(sys.argv[2])

line = f.readline()
tags = line.strip().split(' ')

cnt = 0
tp = {x:0 for x in label_set}
fp = {x:0 for x in label_set}
fn = {x:0 for x in label_set}

none_cnt = 0
dist_errors = []

for line in f:
    lb = label[cnt]
    typ = ltype[cnt]
    cnt += 1
    if typ != '?':
        continue
    a = line.split(' ')
    b = {}
    for i in range(M):
        b[tags[i]] = float(a[i])
    if len(set(b.values())) == 1:
        b = lbl_cnt
    c = sorted(b.items(), key=operator.itemgetter(1), reverse=True)
    test += 1
    for i in range(M):
        if c[i][0] == lb: #and c[i][1] != 0:
            hit[i] += 1
    #if c[0][1] == c[1][1]:
    #   print c, lb
    pred_lb = c[0][0]
    if c[0][1] == 0:
        pred_lb = None
        none_cnt += 1
    if pred_lb == lb:
        tp[lb] += 1
        dist_errors.append(0)
    else:
        fn[lb] += 1
        if pred_lb != None:
            fp[pred_lb] += 1
            real = coords[lb]
            pred = coords[pred_lb]
            dist_errors.append(distance(real, pred))

f.close()

print 'None:', none_cnt

print hit
print test
print '%.4f %.4f' % (float(hit[0]) / test, float(hit[0] + hit[1] + hit[2]) / test)

prec = {}
rec = {}
f1 = {}

for lb in label_set:
    if tp[lb] == 0:
        prec[lb] = 0
        rec[lb] = 0
    else:
        prec[lb] = float(tp[lb]) / float(tp[lb] + fp[lb])
        rec[lb] = float(tp[lb]) / float(tp[lb] + fn[lb])
    if prec[lb] + rec[lb] == 0:
        f1[lb] = 0
    else:
        f1[lb] = 2 * prec[lb] * rec[lb] / (prec[lb] + rec[lb])
    #print lb, tp[lb], fp[lb], fn[lb]

#print prec
#print rec
#print f1

import numpy

print numpy.mean(prec.values())
print numpy.mean(rec.values())
P = numpy.mean(prec.values())
R = numpy.mean(rec.values())
print numpy.mean(f1.values()), 2*P*R/(P+R)

dist_errors.sort()

print 'Accuracy: %.4f' % (float(hit[0]) / test)
print 'Accuracy @3: %.4f' % (float(hit[0] + hit[1] + hit[2]) / test)
print 'Mean error distance: %.2f' % (numpy.mean(numpy.array(dist_errors)))
print 'Median error distance: %.2f' % (dist_errors[test/2])

#with open('map_{}.json'.format(model_name), 'w') as f:
#    json.dump(make_map(f1, 0, 1, "F1 score"), f, indent=2)
