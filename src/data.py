from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.model_selection import train_test_split
from string import punctuation
import numpy as np
from collections import Counter
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

stop_words = list(stop_words.ENGLISH_STOP_WORDS) + list(punctuation)

class Newsgroups:
    def __init__(self, n_features=10000):
        self.n_features = n_features
        
        train = fetch_20newsgroups(subset='train')
        test = fetch_20newsgroups(subset='test')

        self.train_texts, self.train_flat_labels = train.data, train.target    
        self.test_texts, self.test_flat_labels = test.data, test.target

        tfidf = TfidfVectorizer(max_features=self.n_features, stop_words=stop_words)
        self.train_features = tfidf.fit_transform(self.train_texts).toarray().astype(np.float32)
        self.test_features = tfidf.transform(self.test_texts).toarray().astype(np.float32)

        self.class_counts = Counter(self.train_flat_labels)
        self.class_weights = np.array(
            [max(self.class_counts.values())/self.class_counts[i] for i in range(len(self.class_counts))]
        )

        self.flat_label_dict = dict([(k, v) for k, v in enumerate(train.target_names)])
        
        labels_split = [x.split('.') for x in list(self.flat_label_dict.values())]
        
        self.hier_label_dict = []
        self.inv_hier_label_dict = []
        for i in range(max([len(x) for x in labels_split])):
            hier_labels = set([x[i] for x in labels_split if len(x) >= i+1])
            label_dict = dict(enumerate(hier_labels))
            self.hier_label_dict.append(label_dict)
            self.inv_hier_label_dict.append(dict([(v, k) for k, v in label_dict.items()]))
            
        self.flat_to_hier = dict([
            (k, [self.inv_hier_label_dict[i][y] for i, y in enumerate(v.split('.'))]) 
            for k, v in self.flat_label_dict.items()
        ])
        
        self.train_hier_labels = [self.flat_to_hier[x] for x in self.train_flat_labels]
        self.test_hier_labels = [self.flat_to_hier[x] for x in self.test_flat_labels]
        
        self.tree, self.tree_label_dict = self._create_graph(tree=True)
        self.dag, self.dag_label_dict = self._create_graph(tree=False)
        
    def _remap_nodes(self, tree=False):
        if tree:
            labels = list(self.flat_label_dict.values())
            for i in range(len(labels)):
                if labels[i] == 'comp.os.ms-windows.misc':
                    labels[i] = 'comp.os.ms-windows.misc0'
                elif labels[i] == 'comp.sys.ibm.pc.hardware':
                    labels[i] = 'comp.sys.ibm.pc.hardware0'
                elif labels[i] == 'comp.sys.mac.hardware':
                    labels[i] = 'comp.sys.mac.hardware1'
                elif labels[i] == 'misc.forsale':
                    labels[i] = 'misc1.forsale'
                elif labels[i] == 'soc.religion.christian':
                    labels[i] = 'soc.religion0.christian'
                elif labels[i] == 'talk.politics.misc':
                    labels[i] = 'talk.politics.misc2'
                elif labels[i] == 'talk.religion.misc':
                    labels[i] = 'talk.religion1.misc3'
            nodes = [x.split('.') for x in labels]
            return nodes
        else:
            nodes = [x.split('.') for x in self.flat_label_dict.values()]
            nodes = [['misc_{}'.format(i) if x == 'misc' else x for i, x in enumerate(y)] for y in nodes]
            nodes = [['windows' if x == 'ms-windows' else x for x in y] for y in nodes]
            return nodes            
        
    def _create_graph(self, tree=False):
        split_labels = self._remap_nodes(tree=tree)
        graph_label_dict = dict([(i, x) for i, x in enumerate(split_labels)])

        edges = []
        for label in split_labels:
            nodes = ['ROOT'] + label
            edges += [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]

        G = nx.DiGraph()
        G.add_node('ROOT')
        G.add_edges_from(edges)
        return G, graph_label_dict