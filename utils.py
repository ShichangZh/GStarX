import os
import pytz
import logging
import numpy as np
import random
import torch
import networkx as nx
import copy
from rdkit import Chem
from datetime import datetime
from torch_geometric.utils import subgraph, to_dense_adj
from torch_geometric.data import Data, Batch, Dataset, DataLoader

# For associated game
from itertools import combinations
from scipy.sparse.csgraph import connected_components as cc

# For superadditive extension
from more_itertools import set_partitions

# For visualization
from typing import Union, List
from textwrap import wrap
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)

def timetz(*args):
    tz = pytz.timezone('US/Pacific')
    return datetime.now(tz).timetuple()        

def get_logger(log_path, log_file, console_log=False, log_level=logging.INFO):
    check_dir(log_path)
    
    tz = pytz.timezone('US/Pacific')
    logger = logging.getLogger(__name__)
    logger.propagate = False # avoid duplicate logging
    logger.setLevel(log_level)
    
    # Clean logger first to avoid duplicated handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    
    file_handler = logging.FileHandler(os.path.join(log_path, log_file))
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%b%d %H-%M-%S')
    formatter.converter = timetz
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    elif build_method.lower() == 'remove':
        return graph_build_remove
    else:
        raise NotImplementedError

'''
Graph building/Perturbation
`graph_build_zero_filling` and `graph_build_split` are adapted from the DIG library
'''
def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index

def graph_build_split(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through spliting the selected nodes from the original graph """
    ret_X = X
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return ret_X, ret_edge_index

def graph_build_remove(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through removing the unselected nodes from the original graph """
    ret_X = X[node_mask == 1]
    ret_edge_index, _ = subgraph(node_mask.bool(), edge_index, relabel_nodes=True)
    return ret_X, ret_edge_index

'''
Associated game of the HN value
Implementated using sparse tensor
'''
def get_ordered_coalitions(n):
    coalitions = sum([[set(c) for c in combinations(range(n), k)] for k in range(1, n+1)], [])
    return coalitions

def get_associated_game_matrix_M(coalitions, n, tau):
    indices = []
    values = []
    for i, s in enumerate(coalitions):
        for j, t in enumerate(coalitions):
            if i == j:
                indices += [[i, j]]
                values += [1 - (n - len(s)) * tau]
            elif len(s) + 1 == len(t) and s.issubset(t):
                indices += [[i, j]]
                values += [tau]
            elif len(t) == 1 and not t.issubset(s):
                indices += [[i, j]]
                values += [- tau]

    indices = torch.Tensor(indices).t()
    size = (2 ** n - 1, 2 ** n - 1)
    M = torch.sparse_coo_tensor(indices, values, size)
    return M

def get_associated_game_matrix_P(coalitions, n, adj):
    indices = []
    for i, s in enumerate(coalitions):
        idx_s = torch.LongTensor(list(s))
        num_cc, labels = cc(adj[idx_s, :][:, idx_s])
        cc_s = []
        for k in range(num_cc):
            cc_idx_s = (labels == k).nonzero()[0]
            cc_s += [set((idx_s[cc_idx_s]).tolist())]
        for j, t in enumerate(coalitions):
            if t in cc_s:
                indices += [[i,j]]

    indices = torch.Tensor(indices).t()
    values = [1.0] * indices.shape[-1]
    size = (2 ** n - 1, 2 ** n - 1)

    P = torch.sparse_coo_tensor(indices, values, size)
    return P

def get_limit_game_matrix(H, exp_power=7, tol=1e-3, is_sparse=True):
    '''
    Speed up the power computation by 
    1. Use sparse matrices
    2. Put all tensors on cuda
    3. Compute powers exponentially rather than linearly
        i.e. H -> H^2 -> H^4 -> H^8 -> H^16 -> ...
    '''
    i = 0
    diff_norm = tol + 1
    while i < exp_power and diff_norm > tol:
        if is_sparse:
            H_tilde = torch.sparse.mm(H, H)
        else:
            H_tilde = torch.mm(H, H)
        diff_norm = (H_tilde - H).norm()
        H = H_tilde
        i += 1
    return H_tilde


'''
khop or random sampling to generate subgraphs
'''
def sample_subgraph(data, max_sample_size, sample_method, target_node=None, k=0, adj=None):
    if sample_method == 'khop': 
        # pick nodes within k-hops of target node. Hop by hop until reach max_sample_size
        if adj is None:
            adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].detach().cpu()  
        
        adj_self_loop = adj + torch.eye(data.num_nodes)
        k_hop_adj = adj_self_loop
        sampled_nodes = set()
        m = max_sample_size
        l = 0
        while k > 0 and l < m:
            k_hop_nodes = k_hop_adj[target_node].nonzero().view(-1).tolist()
            next_hop_nodes = list(set(k_hop_nodes) - sampled_nodes)
            sampled_nodes.update(next_hop_nodes[:m-l])
            l = len(sampled_nodes)
            k -= 1
            k_hop_adj = torch.mm(k_hop_adj, adj_self_loop)
        sampled_nodes = torch.tensor(list(sampled_nodes))
    
    elif sample_method == 'random': # randomly pick #max_sample_size nodes
        sampled_nodes = torch.randperm(data.num_nodes)[:max_sample_size]
    else:
        ValueError("Unknown sample method")
        
    sampled_x = data.x[sampled_nodes]
    sampled_edge_index, _ = subgraph(sampled_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    sampled_data = Data(x=sampled_x, edge_index=sampled_edge_index)
    sampled_adj = adj[sampled_nodes, :][:, sampled_nodes]

    return sampled_nodes, sampled_data, sampled_adj


'''
Payoff computation
'''
def get_char_func(model, target_class, payoff_type='norm_prob', payoff_avg=None):
    def char_func(data):
        with torch.no_grad():
            logits = model(data=data)
            if payoff_type == 'raw':
                payoff = logits[:, target_class]
            elif payoff_type == 'prob':
                payoff = logits.softmax(dim=-1)[:, target_class]
            elif payoff_type == 'norm_prob':
                prob = logits.softmax(dim=-1)[:, target_class]
                payoff = prob - payoff_avg[target_class]
            elif payoff_type == 'log_prob':
                payoff = logits.log_softmax(dim=-1)[:, target_class]
            else:
                raise ValueError('unknown payoff type')                
        return payoff
    return char_func

class MaskedDataset(Dataset):
    def __init__(self, data, mask, subgraph_building_func):
        super().__init__()
        
        self.num_nodes = data.num_nodes
        self.x = data.x
        self.edge_index = data.edge_index
        self.device = data.x.device
        self.y = data.y
        
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask)
        
        self.mask = mask.type(torch.float32).to(self.device)
        self.subgraph_building_func = subgraph_building_func
        
    def __len__(self):
        return self.mask.shape[0]

    def __getitem__(self, idx):
        masked_x, masked_edge_index = self.subgraph_building_func(self.x, self.edge_index, self.mask[idx])
        masked_data = Data(x=masked_x, edge_index=masked_edge_index)
        return masked_data

def get_coalition_payoffs(data, coalitions, char_func, subgraph_building_func):
    n = data.num_nodes
    masks = []
    for coalition in coalitions:
        mask = torch.zeros(n)
        mask[list(coalition)] = 1.0
        masks += [mask]
        
    coalition_mask = torch.stack(masks, axis=0)
    masked_dataset = MaskedDataset(data, coalition_mask, subgraph_building_func)
    masked_dataloader = DataLoader(masked_dataset, batch_size=256, shuffle=False, num_workers=0)

    masked_payoff_list = []
    for masked_data in masked_dataloader:
        masked_payoff_list.append(char_func(masked_data))

    masked_payoffs = torch.cat(masked_payoff_list, dim=0)
    return masked_payoffs        

'''
Superadditive extension
'''
class TrieNode:
    def __init__(self, player, payoff=0, children=[]):
        self.player = player
        self.payoff = payoff
        self.children = children

class CoalitionTrie:
    def __init__(self, coalitions, n, v):
        self.n = n
        self.root = self.get_node(None, 0)
        for i, c in enumerate(coalitions):
            self.insert(c, v[i])
    
    def get_node(self, player, payoff):
        return TrieNode(player, payoff, [None] * self.n)
    
    def insert(self, coalition, payoff):
        curr = self.root
        for player in coalition:
            if curr.children[player] is None:
                curr.children[player] = self.get_node(player, 0)
            curr = curr.children[player]
        curr.payoff = payoff
    
    def search(self, coalition):
        curr = self.root
        for player in coalition:
            if curr.children[player] is None:
                return None
            curr = curr.children[player]
        return curr.payoff

    def visualize(self):
        self._visualize(self.root, 0)
        
    def _visualize(self, node, level):
        if node:
            print(f"{'-'*level}{node.player}:{node.payoff}")
            for child in node.children:
                self._visualize(child, level+1)

def superadditive_extension(n, v):
    '''
    n (int): number of players
    v (list of floats): dim = 2 ** n - 1, each entry is a payoff
    '''
    coalition_sets = get_ordered_coalitions(n)
    coalition_lists = [sorted(list(c)) for c in coalition_sets]
    coalition_trie = CoalitionTrie(coalition_lists, n, v)
    v_ext = v[:]
    for i, coalition in enumerate(coalition_lists):
        partition_payoff = []
        for part in set_partitions(coalition, 2):
            subpart_payoff = []
            for subpart in part:
                subpart_payoff += [coalition_trie.search(subpart)]
            partition_payoff += [sum(subpart_payoff)]
        v_ext[i] = max(partition_payoff + [v[i]])
        coalition_trie.insert(coalition, v_ext[i])
    return v_ext

'''
Evaluation functions
'''
def scores2coalition(scores, sparsity):
    scores_tensor = torch.tensor(scores)  
    top_idx = scores_tensor.argsort(descending=True).tolist()
    cutoff = int(len(scores) * (1 - sparsity))
    cutoff = min(cutoff, (scores_tensor > 0).sum().item())
    coalition = top_idx[:cutoff]
    return coalition

def evaluate_coalition(explainer, data, coalition):
    device = explainer.device
    data = data.to(device)
    pred_prob = explainer.model(data).softmax(dim=-1)
    target_class = pred_prob.argmax(-1).item()
    original_prob = pred_prob[:, target_class].item()

    num_nodes = data.num_nodes
    if len(coalition) == num_nodes:
        # Edge case: pick the graph itself as the explanation, for synthetic data
        masked_prob = original_prob
        maskout_prob = 0
    elif len(coalition) == 0:
        # Edge case: pick the empty set as the explanation, for synthetic data
        masked_prob = 0         
        maskout_prob = original_prob
    else:
        mask = torch.zeros(num_nodes).type(torch.float32).to(device)
        mask[coalition] = 1.0
        masked_x, masked_edge_index = explainer.subgraph_building_func(data.x, data.edge_index, mask)
        masked_data = Data(x=masked_x, edge_index=masked_edge_index).to(device)
        masked_prob = explainer.model(masked_data).softmax(dim=-1)[:, target_class].item()

        maskout_x, maskout_edge_index = explainer.subgraph_building_func(data.x, data.edge_index, 1 - mask)
        maskout_data = Data(x=maskout_x, edge_index=maskout_edge_index).to(device)
        maskout_prob = explainer.model(maskout_data).softmax(dim=-1)[:, target_class].item()

    fidelity = original_prob - maskout_prob
    inv_fidelity = original_prob - masked_prob
    sparsity = 1 - len(coalition) / num_nodes
    return fidelity, inv_fidelity, sparsity

def fidelity_normalize_and_harmonic_mean(fidelity, inv_fidelity, sparsity):
    '''
    The idea is similar to the F1 score, two measures are summarized to one through harmonic mean.

    Step1: normalize both scores with sparsity
        norm_fidelity = fidelity * sparsity
        norm_inv_fidelity = inv_fidelity * (1 - sparsity)
    Step2: rescale both normalized scores from [-1, 1] to [0, 1]
        rescaled_fidelity = (1 + norm_fidelity) / 2
        rescaled_inv_fidelity = (1 - norm_inv_fidelity) / 2
    Step3: take the harmonic mean of two rescaled scores
        2 / (1/rescaled_fidelity + 1/rescaled_inv_fidelity)

    Simplifying these three steps gives the formula
    '''
    norm_fidelity = fidelity * sparsity
    norm_inv_fidelity = inv_fidelity * (1 - sparsity)
    harmonic_fidelity = (1 + norm_fidelity) * (1 - norm_inv_fidelity) / (2 + norm_fidelity - norm_inv_fidelity)
    return norm_fidelity, norm_inv_fidelity, harmonic_fidelity

def evaluate_scores_list(explainer, data_list, scores_list, sparsity, logger=None):
    '''
    Evaluate the node importance scoring methods, where each node has an associated score, 
    i.e. GStarX and GraphSVX.

    Args:
    data_list (list of PyG data)
    scores_list (list of lists): each entry is a list with scores of nodes in a graph
    
    '''
    
    assert len(data_list) == len(scores_list)
    
    f_list = []
    inv_f_list = []
    n_f_list = []
    n_inv_f_list = []
    sp_list = []
    h_f_list = []
    for i, data in enumerate(data_list):
        node_scores = scores_list[i]
        coalition = scores2coalition(node_scores, sparsity)
        f, inv_f, sp = evaluate_coalition(explainer, data, coalition)
        n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)
        
        f_list += [f]
        inv_f_list += [inv_f]
        n_f_list += [n_f]
        n_inv_f_list += [n_inv_f]
        sp_list += [sp]
        h_f_list += [h_f]
            
    f_mean = np.mean(f_list).item()
    inv_f_mean = np.mean(inv_f_list).item()
    n_f_mean = np.mean(n_f_list).item()
    n_inv_f_mean = np.mean(n_inv_f_list).item()
    sp_mean = np.mean(sp_list).item()
    h_f_mean = np.mean(h_f_list).item()
    
    if logger is not None:
        logger.info(f'Fidelity Mean: {f_mean:.4f}\n'
                  f'Inv-Fidelity Mean: {inv_f_mean:.4f}\n'
                  f'Norm-Fidelity Mean: {n_f_mean:.4f}\n'
                  f'Norm-Inv-Fidelity Mean: {n_inv_f_mean:.4f}\n'
                  f'Sparsity Mean: {sp_mean:.4f}\n'
                  f'Harmonic-Fidelity Mean: {h_f_mean:.4f}\n') 
    
    return sp_mean, f_mean, inv_f_mean, n_f_mean, n_inv_f_mean, h_f_mean


'''
Visualization
'''
def coalition2subgraph(coalition, data, relabel_nodes=True):
    sub_data = copy.deepcopy(data)
    node_mask = torch.zeros(data.num_nodes)
    node_mask[coalition] = 1
    
    sub_data.x = data.x[node_mask == 1]
    sub_data.edge_index, _ = subgraph(node_mask.bool(), data.edge_index, relabel_nodes=relabel_nodes)
    return sub_data

def to_networkx(data, node_index=None, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    r"""
    Extend the PyG to_networkx with extra node_index argument, so subgraphs can be plotted with correct ids
    
    Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
            
            
        node_index (iterable): Pass in it when there are some nodes missing.
                 max(node_index) == max(data.edge_index)
                 len(node_index) == data.num_nodes
    """
    import networkx as nx
    
    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    if node_index is not None:
        '''
        There are some nodes missing. The max(data.edge_index) > data.x.shape[0]
        '''
        G.add_nodes_from(node_index)
    else:
        G.add_nodes_from(range(data.num_nodes))
    
    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []

    values = {}
    for key, item in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

        for key in edge_attrs:
            G[u][v][key] = values[key][i]

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G

'''
Adapted from SubgraphX DIG implementation
https://github.com/divelab/DIG/blob/dig/dig/xgraph/method/subgraphx.py

Slightly modified the molecule drawing args
'''
class PlotUtils(object):
    def __init__(self, dataset_name, is_show=True):
        self.dataset_name = dataset_name
        self.is_show = is_show

    def plot(self, graph, nodelist, figname, title_sentence=None, **kwargs):
        """ plot function for different dataset """
        if self.dataset_name.lower() in ['ba_2motifs']:
            self.plot_ba2motifs(graph, nodelist, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['mutag', 'bbbp', 'bace']:
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['graph_sst2', 'twitter']:
            words = kwargs.get('words')
            self.plot_sentence(graph, nodelist, words=words, title_sentence=title_sentence, figname=figname)
        else:
            raise NotImplementedError

    def plot_subgraph(self,
                      graph,
                      nodelist,
                      colors: Union[None, str, List[str]] = '#FFA500',
                      labels=None,
                      edge_color='gray',
                      edgelist=None,
                      subgraph_edge_color='black',
                      title_sentence=None,
                      figname=None):

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]

        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos_nodelist,
                               nodelist=nodelist,
                               node_color='black',
                               node_shape='o',
                               node_size=400)
        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=200)
        nx.draw_networkx_edges(graph, pos, width=2, edge_color=edge_color, arrows=False)
        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color='black',
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)), fontdict={'fontsize':15})
        if figname is not None:
            plt.savefig(figname, format=figname[-3:])

        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_sentence(self, graph, nodelist, words, edgelist=None, title_sentence=None, figname=None):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist,
                                   node_color='yellow',
                                   node_shape='o',
                                   node_size=500)
            if edgelist is None:
                edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                            if n_frm in nodelist and n_to in nodelist]
                nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=5, edge_color='yellow')

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='grey')
        nx.draw_networkx_labels(graph, pos, words_dict)

        plt.axis('off')
        plt.title('\n'.join(wrap(' '.join(words), width=50)))
        if title_sentence is not None:
            string = '\n'.join(wrap(' '.join(words), width=50)) + '\n'
            string += '\n'.join(wrap(title_sentence, width=60))
            plt.title(string)
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_ba2motifs(self,
                       graph,
                       nodelist,
                       edgelist=None,
                       title_sentence=None,
                       figname=None):
        return self.plot_subgraph(graph, nodelist,
                                  edgelist=edgelist,
                                  title_sentence=title_sentence,
                                  figname=figname)
    def plot_molecule(self,
                      graph,
                      nodelist,
                      x,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        # collect the text information and node color
        if self.dataset_name == 'mutag':
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        elif self.dataset_name in ['bbbp', 'bace']:
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                           for k, v in element_idxs.items()}
            node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
            colors = [node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, nodelist,
                           colors=colors,
                           labels=node_labels,
                           edgelist=edgelist,
                           edge_color='gray',
                           subgraph_edge_color='black',
                           title_sentence=title_sentence,
                           figname=figname)


if __name__ == '__main__':
    # Test HN value and Shapley value via unanimity games
    def test_unanimity_game_value(n, S, tau=0.05, edges=None, value_type='hn', device='cpu'):
        '''
        S (a set of ints in [0, n)): selected players in the unanimity_game.
            A coalition evaluates to 1 if S is its subset. Otherwise 0.
        '''
        coalitions = get_ordered_coalitions(n)
        M = get_associated_game_matrix_M(coalitions, n, tau)
        M = M.to(device)

        if value_type == 'hn':
            adj = torch.zeros(n, n)
            for src, tgt in edges:
                adj[src, tgt] = 1
                adj[tgt, src] = 1

            P = get_associated_game_matrix_P(coalitions, n, adj)
            P = P.to(device)
            H = torch.sparse.mm(P, torch.sparse.mm(M, P))
        elif value_type == 'shapley':
            H = M

        H_tilde = get_limit_game_matrix(H)
        v = [1.0 if S.issubset(coalition) else 0.0 for coalition in coalitions]
        v = torch.tensor(v)
        v_tilde = torch.sparse.mm(H_tilde, v.view(-1,1)).view(-1)
        return v_tilde[:n]

    device = 'cpu'
    examples = [{'n': 4, 
      'edges':[(0,1), (0,2), (0,3), (1,2)],
     'values':[4/9, 7/36, 7/36, 1/6]},
               {'n': 4, 
      'edges':[(0,1), (0,2), (0,3), (1,2), (1,3)],
     'values':[5/18, 5/18, 2/9, 2/9]},
               {'n': 5, 
      'edges':[(0,1), (0,2), (0,3), (1,2), (1,3), (2,4)],
     'values':[7/36, 7/36, 10/27, 7/54, 1/9]},
               {'n': 5, 
      'edges':[(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (3,4), (2,4)],
     'values':[3/14, 11/56, 11/56, 11/56, 11/56]},
               {'n': 5, 
      'edges':[(0,1), (0,2), (0,3), (1,4)],
     'values':[5/12, 11/36, 7/72, 7/72, 1/12]}]

    for ex in examples:
        HN_values = test_unanimity_game_value(ex['n'], set(range(ex['n'])), edges=ex['edges'], value_type='hn')
        assert torch.allclose(HN_values, torch.Tensor(ex['values']), rtol=1e-3)
        shapley_values = test_unanimity_game_value(ex['n'], set(range(ex['n'])), value_type='shapley')
        print(HN_values, shapley_values)

    n = 3
    edges=[(0,1), (1,2)]
    S = {0,1,2}
    tau = 0.05

    coalitions = get_ordered_coalitions(n)
    M = get_associated_game_matrix_M(coalitions, n, tau)
    M = M.to(device)

    M_tilde = get_limit_game_matrix(M)
    weight_shapley = M_tilde.to_dense().numpy().round(4)[0]

    adj = torch.zeros(n, n)
    for src, tgt in edges:
        adj[src, tgt] = 1
        adj[tgt, src] = 1

    P = get_associated_game_matrix_P(coalitions, n, adj)
    P = P.to(device)
    H = torch.sparse.mm(P, torch.sparse.mm(M, P))

    H_tilde = get_limit_game_matrix(H)
    weight_HN = H_tilde.to_dense().numpy().round(4)[0]

    [print(f'{str(c) : >15}: {weight_HN[i]:.4f}   {weight_shapley[i]:.4f}') for i, c in enumerate(coalitions)]

    # Visualize CoalitionTrie
    set_seed(2)
    n = 5
    v = torch.randperm(2**(n+1))[:2**n-1].tolist()

    coalition_sets = get_ordered_coalitions(n)
    coalition_lists = [sorted(list(c)) for c in coalition_sets]
    coalition_trie = CoalitionTrie(coalition_lists, n, v)
    coalition_trie.visualize()

    for i, c in enumerate(coalition_lists):
        print(c, v[i])

    # Test CoalitionTrie
    def test_coalition_trie(n, seed):
        set_seed(seed)
        v = torch.randperm(2**(n+1))[:2**n-1].tolist()
        coalition_sets = get_ordered_coalitions(n)
        coalition_lists = [sorted(list(c)) for c in coalition_sets]
        coalition_trie = CoalitionTrie(coalition_lists, n, v)
        for i, c in enumerate(coalition_lists):
            assert v[i] == coalition_trie.search(c)

    for seed in range(5):
        test_coalition_trie(4, seed)
        test_coalition_trie(5, seed)
        test_coalition_trie(6, seed)
    
    # Test superadditive_extension
    set_seed(2)
    n = 5
    v = torch.randperm(2**(n+1))[:2**n-1].tolist()

    coalition_sets = get_ordered_coalitions(n)
    coalition_lists = [sorted(list(c)) for c in coalition_sets]
    coalition_trie = CoalitionTrie(coalition_lists, n, v)
    v_ext = superadditive_extension(n, v)

    for i, c in enumerate(coalition_lists):
        print(c, v[i], v_ext[i])
