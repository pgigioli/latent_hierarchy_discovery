import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

def straight_through_estimator(logits):
    argmax = torch.eq(logits, logits.max(-1, keepdim=True).values).to(logits.dtype)
    return (argmax - logits).detach() + logits

def gumbel_softmax(logits, temperature=1.0, eps=1e-20):
    u = torch.rand(logits.size(), dtype=logits.dtype, device=logits.device)
    g = -torch.log(-torch.log(u + eps) + eps)
    return F.softmax((logits + g) / temperature, dim=-1)

class CategoricalLayer(nn.Module):
    def __init__(self, input_dim, categorical_dim, output_dim=None):
        super().__init__()
        
        if output_dim == None:
            output_dim = input_dim
            
        self.dense_in = nn.Linear(input_dim, categorical_dim, bias=True)
        self.dense_out = nn.Linear(input_dim+categorical_dim, output_dim, bias=True)
        
    def forward(self, inputs, straight_through=True, sample=False, temperature=1.0, return_logits=False):
        logits = self.dense_in(inputs)
        
        if sample:
            dist = gumbel_softmax(logits, temperature=temperature)
        else:
            dist = F.softmax(logits, dim=-1)
            
        if straight_through:
            dist = straight_through_estimator(dist)
            
        h = torch.tanh(self.dense_out(torch.cat([inputs, dist], dim=-1)))
        
        if return_logits:
            return h, dist, logits
        else:
            return h, dist
    
class HLGC(nn.Module):
    def __init__(self, n_classes, input_dim, categorical_dims, hidden_dim=128, dropout_rate=0.5, batch_size=16,
                 lr=0.001, val_split=0.1, n_epochs=25, class_weights=None, recon_loss_weight=1.0, latent_loss_weight=1.0, 
                 l1_lambda=0.001, display_interval=100, early_stopping_epochs=5, label_dict=None):
        super().__init__()
        
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.categorical_dims = categorical_dims
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.val_split = val_split
        self.class_weights = class_weights
        self.recon_loss_weight = recon_loss_weight
        self.latent_loss_weight = latent_loss_weight
        self.l1_lambda = l1_lambda
        self.display_interval = display_interval
        self.early_stopping_epochs = early_stopping_epochs
        self.label_dict = label_dict
        
        if type(self.class_weights) is not torch.tensor:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32)
        
        self._build_model()
        
    def _build_model(self):
        # classifier
        self.input_dense = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        
        self.categorical_layers = nn.ModuleList([
            CategoricalLayer(self.hidden_dim, dim) for dim in self.categorical_dims
        ])
        
        self.global_dense = nn.Linear(sum(self.categorical_dims), self.hidden_dim, bias=True)
        self.out_dense = nn.Linear(self.hidden_dim, self.n_classes, bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)
                
        # generator
        self.encoder = nn.ModuleList([
            nn.Linear(self.hidden_dim+dim, self.hidden_dim, bias=True) for dim in self.categorical_dims
        ])
        self.encoder_out = nn.Linear(self.hidden_dim, self.n_classes, bias=True) 
        
        self.decoder_in = nn.Linear(self.n_classes, self.hidden_dim, bias=True)
        self.decoder = nn.ModuleList([
            CategoricalLayer(self.hidden_dim, dim) for dim in self.categorical_dims
        ])
        
    def _calc_loss(self, targets, clf_logits, support_logits, clf_states, gen_logits, gen_states, z):
        # flat classification loss
        clf_loss = nn.CrossEntropyLoss(weight=self.class_weights)(clf_logits, targets)
        
        # support classification loss
        support_loss = nn.CrossEntropyLoss(weight=self.class_weights)(support_logits, targets)

        # generator DAG reconstruction loss
        recon_loss = 0.0
        for clf_state, gen_logit in zip(clf_states, gen_logits):
            recon_loss += nn.CrossEntropyLoss()(gen_logit, clf_state.argmax(-1))
        recon_loss /= len(clf_states)

        # generator latent loss
        latent_loss = nn.CrossEntropyLoss(weight=self.class_weights)(z, clf_logits.argmax(-1))
        
        # L1 regularization
        l1_loss = 0.0
        for param in self.global_dense.parameters():
            l1_loss += torch.norm(param, 1)

        loss = (
            clf_loss + 
            support_loss +
            self.recon_loss_weight*recon_loss +
            self.latent_loss_weight*latent_loss + 
            self.l1_lambda*l1_loss
        )
        return loss, clf_loss, support_loss, recon_loss, latent_loss
    
    def _calc_metrics(self, targets, clf_logits, clf_states, gen_states):
        clf_acc = (targets == clf_logits.argmax(-1)).to(float).mean()

        recon_acc = 0.0
        for clf_state, gen_state in zip(clf_states, gen_states):
            recon_acc += (clf_state.argmax(-1) == gen_state.argmax(-1)).to(float).mean()
        recon_acc /= len(clf_states)

        return clf_acc, recon_acc
        
    def encode(self, dists):
        h = torch.zeros(dists[0].size(0), self.hidden_dim, device=dists[0].device) 
        
        for dist, layer in zip(dists, self.encoder):
            h = torch.tanh(layer(torch.cat([h, dist], dim=-1))) 
        z = self.encoder_out(h)
        return z
    
    def generate(self, z_sample, straight_through=True, temperature=0.1): 
        h = torch.tanh(self.decoder_in(z_sample))
        
        gen_states, gen_logits = [], []
        for layer in self.decoder:
            h, dist, logits = layer(
                h, straight_through=straight_through, temperature=temperature, sample=True, 
                return_logits=True
            )
            gen_states.append(dist)
            gen_logits.append(logits)
        return gen_states, gen_logits
    
    def generate_dag(self, display=True):
        # generate edges
        edges, labeled_paths = [], []
        for i, z in enumerate(torch.eye(self.n_classes)):
            states, _ = self.generate(z)
            path = [x.argmax(-1).item() for x in states]
            nodes = ['ROOT'] + ['{}_{}'.format(i, x) for i, x in enumerate(path)]
            edges += [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]

            label = self.label_dict[i] if self.label_dict else i
            labeled_paths.append((label, path))
        labeled_paths = sorted(labeled_paths, key=lambda x : x[1][0])

        # print labeled paths
        curr = labeled_paths[0][1][0]
        for l, p in labeled_paths: 
            if curr != p[0]:
                print('-----')
            print('{} : {}'.format(l, p))
            curr = p[0]

        # construct DAG
        G = nx.DiGraph()
        G.add_node('ROOT')
        G.add_edges_from(edges)

        # display DAG
        plt.title('DAG')
        pos=graphviz_layout(G, prog='dot')
        nx.draw(G, pos)
        plt.show()
        return G, labeled_paths
    
    def classify(self, inputs, return_states=False):
        h_0 = torch.tanh(self.input_dense(inputs))
        self.dropout(h_0)
        
        h = h_0
        states = []
        for layer in self.categorical_layers:
            h, dist = layer(h, straight_through=True, sample=False)
            states.append(dist)
            
        h = torch.tanh(self.global_dense(torch.cat(states, dim=-1)))
        support_logits = self.out_dense(h)
        clf_logits = self.out_dense(h_0)
        
        if return_states:
            return clf_logits, support_logits, states
        else:
            return clf_logits
        
    def forward(self, inputs, temperature=1.0):
        # classifier
        clf_logits, support_logits, clf_states = self.classify(inputs, return_states=True)
        
        # generator
        z = self.encode([x.detach() for x in clf_states])
        z_sample = straight_through_estimator(gumbel_softmax(z, temperature=temperature))
        gen_states, gen_logits = self.generate(z_sample, straight_through=True, temperature=temperature)
        return clf_logits, support_logits, clf_states, gen_logits, gen_states, z
    
    def score(self, features, targets):
        data_loader = DataLoader(list(zip(features, targets)), batch_size=self.batch_size)

        self.eval()
        eval_loss, eval_clf_loss, eval_support_loss, eval_recon_loss, eval_latent_loss, eval_clf_acc, eval_recon_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in data_loader:
                batch_features, batch_targets = batch
                clf_logits, support_logits, clf_states, gen_logits, gen_states, z = self.forward(batch_features)

                loss, clf_loss, support_loss, recon_loss, latent_loss = self._calc_loss(
                    batch_targets, clf_logits, support_logits, clf_states, gen_logits, gen_states, z
                )

                clf_acc, recon_acc = self._calc_metrics(batch_targets, clf_logits, clf_states, gen_states)

                scale = len(batch_features)/len(features)
                eval_loss += loss.item()*scale
                eval_clf_loss += clf_loss.item()*scale
                eval_support_loss += support_loss.item()*scale
                eval_recon_loss += recon_loss.item()*scale
                eval_latent_loss += latent_loss.item()*scale
                eval_clf_acc += clf_acc.item()*scale
                eval_recon_acc += recon_acc.item()*scale
            print(
                '[ Eval ] loss - (total : {:3f}, clf : {:3f}, support : {:3f}, recon : {:3f}, latent : {:3f}), \
acc - (clf : {:3f}, recon : {:3f})'.format(eval_loss, eval_clf_loss, eval_support_loss, eval_recon_loss, eval_latent_loss, 
                                                   eval_clf_acc, eval_recon_acc)
            )
        return eval_clf_acc
    
    def fit(self, train_features, train_targets, val_features=None, val_targets=None, save_dir=None):
        if (val_features is None) or (val_targets is None):
            print('No validation data provided, using {}% of train data'.format(100*self.val_split))
            
            train_data, val_data = train_test_split(
                list(zip(train_features, train_targets)), test_size=self.val_split, random_state=0
            )
            train_features, train_targets = zip(*train_data)
            val_features, val_targets = zip(*val_data)
        
        train_loader = DataLoader(
            list(zip(train_features, train_targets)), 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        optimizer = optim.Adam(
            filter(lambda x: x.requires_grad, self.parameters()), betas=(0.9, 0.98),
            eps=1e-09,
            lr=self.lr
        )
        
        itr, since_best = 0, 0
        best_acc, best_loss = 0.0, 10e9
        for epoch in range(self.n_epochs):
            self.train()
            for batch in train_loader:
                itr += 1

                batch_features, batch_targets = batch
                clf_logits, support_logits, clf_states, gen_logits, gen_states, z = self.forward(batch_features)

                loss, clf_loss, support_loss, recon_loss, latent_loss = self._calc_loss(
                    batch_targets, clf_logits, support_logits, clf_states, gen_logits, gen_states, z
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()

                if itr == 1 or itr % self.display_interval == 0:
                    clf_acc, recon_acc = self._calc_metrics(batch_targets, clf_logits, clf_states, gen_states)

                    log_string = '[{}, {:5d}] loss - (total : {:3f}, clf : {:3f}, support : {:3f}, recon : {:3f}, latent : {:3f}), \
acc - (clf : {:3f}, recon : {:3f})'.format(epoch, itr, loss.item(), clf_loss.item(), support_loss.item(), recon_loss.item(),
                                                   latent_loss.item(), clf_acc.item(), recon_acc.item())
                    print(log_string)

            val_acc = self.score(val_features, val_targets)
            if val_acc > best_acc:
                best_acc = val_acc
                since_best = 0
                if save_dir:
                    path = os.path.join(save_dir, 'model.weights'.format(itr))
                    torch.save(self.state_dict(), path)
            else:
                since_best += 1
                
            _, _ = self.generate_dag()
                
            if since_best == self.early_stopping_epochs:
                break
        print('Training complete!')
        
    def load(self, path):
        self.load_state_dict(path)