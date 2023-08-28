import numpy as np
import pandas as pd
from collections import Counter
import math

# define node
class Node:
    def __init__(self, x=None, label=None, y=None, data=None):
        self.label = label
        self.x = x
        self.child = []
        self.y = y
        self.data = data

    def append(self, node):
        self.child.append(node)

    def predict(self, features):
        if self.y is not None:
            return self.y
        for c in self.child:
            if c.x == features[self.label]:
                return c.predict(features)

# print node of the tree
def printnode(node, depth=0):
    if node.label is None:
        print(depth, (node.label, node.x, node.y, len(node.data)))
    else:
        print(depth, (node.label, node.x))
        for c in node.child:
            printnode(c, depth+1)

# create decision tree
class DTree:
    def __init__(self, epsilon=0, alpha=0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.tree = Node()

    def prob(self, datasets):
        datalen = len(datasets)
        labelx = set(datasets)
        p = {l: 0 for l in labelx}
        for d in datasets:
            p[d] += 1
        for i in p.items():
            p[i[0]] /= datalen
        return p

    def calc_ent(self, datasets):
        p = self.prob(datasets)
        ent = sum([-v * math.log(v, 2) for v in p.values()])
        return ent

    def cond_ent(self, datasets, col):
        labelx = set(datasets.iloc[col])
        p = {x: [] for x in labelx}
        for i, d in enumerate(datasets.iloc[-1]):
            p[datasets.iloc[col][i]].append(d)
        return sum([self.prob(datasets.iloc[col])[k]
                    * self.calc_ent(p[k]) for k in p.keys()])

    def info_gain_train(self, datasets, datalabels):  # information gain
        datasets = datasets.T
        ent = self.calc_ent(datasets.iloc[-1])
        gainmax = {}
        for i in range(len(datasets) - 1):
            cond = self.cond_ent(datasets, i)
            #print(datalabels[i], ent - cond)
            gainmax[ent - cond] = i
        m = max(gainmax.keys())
        return gainmax[m], m

    def train(self, datasets, node):
        labely = datasets.columns[-1]
        if len(datasets[labely].value_counts()) == 1:
            node.data = datasets[labely]
            node.y = datasets[labely][0]
            return
        if len(datasets.columns[:-1]) == 0:
            node.data = datasets[labely]
            node.y = datasets[labely].value_counts().index[0]
            return
        gainmaxi, gainmax = self.info_gain_train(datasets, datasets.columns)

        if gainmax <= self.epsilon:
            node.data = datasets[labely]
            node.y = datasets[labely].value_counts().index[0]
            return

        vc = datasets[datasets.columns[gainmaxi]].value_counts()
        for Di in vc.index:
            node.label = gainmaxi
            child = Node(Di)
            node.append(child)
            new_datasets = pd.DataFrame([list(i) for i in datasets.values if i[gainmaxi]==Di], columns=datasets.columns)
            self.train(new_datasets, child)

    def fit(self, datasets):
        self.train(datasets, self.tree)

    def findleaf(self, node, leaf):
        for t in node.child:
            if t.y is not None:
                leaf.append(t.data)
            else:
                for c in node.child:
                    self.findleaf(c, leaf)

    def findfather(self, node, errormin):
        if node.label is not None:
            cy = [c.y for c in node.child]
            if None not in cy:
                childdata = []
                for c in node.child:
                    for d in list(c.data):
                        childdata.append(d)
                childcounter = Counter(childdata)

                old_child = node.child
                old_label = node.label
                old_y = node.y
                old_data = node.data

                node.label = None
                node.y = childcounter.most_common(1)[0][0]
                node.data = childdata

                error = self.c_error()
                if error <= errormin:
                    errormin = error
                    return 1
                else:
                    node.child = old_child
                    node.label = old_label
                    node.y = old_y
                    node.data = old_data
            else:
                re = 0
                i = 0
                while i < len(node.child):
                    if_re = self.findfather(node.child[i], errormin)  # 若剪过枝，则其父节点要重新检测
                    if if_re == 1:
                        re = 1
                    elif if_re == 2:
                        i -= 1
                    i += 1
                if re:
                    return 2
        return 0

    def c_error(self):
        leaf = []
        self.findleaf(self.tree, leaf)
        leafnum = [len(l) for l in leaf]
        ent = [self.calc_ent(l) for l in leaf]
        print("Ent:", ent)
        error = self.alpha*len(leafnum)
        for l, e in zip(leafnum, ent):
            error += l*e
        print("C(T):", error)
        return error

    def cut(self, alpha=0.0):
        if alpha:
            self.alpha = alpha
        errormin = self.c_error()
        self.findfather(self.tree, errormin)


datasets = np.array([['young', 'no', 'no', 'general', 'no'],
               ['young', 'no', 'no', 'good', 'no'],
               ['young', 'yes', 'no', 'good', 'yes'],
               ['young', 'yes', 'yes', 'general', 'yes'],
               ['young', 'no', 'no', 'general', 'no'],
               ['mid', 'no', 'no', 'general', 'no'],
               ['mid', 'no', 'no', 'good', 'no'],
               ['mid', 'yes', 'yes', 'good', 'yes'],
               ['mid', 'no', 'yes', 'Vgood', 'yes'],
               ['mid', 'no', 'yes', 'Vgood', 'yes'],
               ['old', 'no', 'yes', 'Vgood', 'yes'],
               ['old', 'no', 'yes', 'good', 'yes'],
               ['old', 'yes', 'no', 'good', 'yes'],
               ['old', 'yes', 'no', 'Vgood', 'yes'],
               ['old', 'no', 'no', 'general', 'no'],
               ['young', 'no', 'no', 'general', 'yes']])

datalabels = np.array(['age', 'job', 'house', 'loan', 'class'])
train_data = pd.DataFrame(datasets, columns=datalabels)
test_data = ['old', 'no', 'no', 'general']

dt = DTree(epsilon=0)
dt.fit(train_data)

print('DTree:')
printnode(dt.tree)
y = dt.tree.predict(test_data)
print('result:', y)

dt.cut(alpha=0.5)

print('DTree:')
printnode(dt.tree)
y = dt.tree.predict(test_data)
print('result:', y)