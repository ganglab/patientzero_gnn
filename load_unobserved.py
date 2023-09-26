import torch
from torch.utils.data import Dataset
import numpy as np
import random
import networkx as nx
import itertools as its
import collections as cls
import torch.nn.functional as F
import os
import data_processing as dp
from sklearn.model_selection import train_test_split

HOME = os.environ['HOME']
Neg = -1e12


class Custom_dataset(Dataset):

    def __init__(self, catagory = 'sex', p=0.3, q=0.01, type= 'train', unobserved=0. ):

        directed = False if catagory == 'sex' else True

        data = dp.pre_processing(catagory=catagory)

        if catagory == 'sex':  # timestep start from 1
            start_interval = [1, 10]
            self.max_day = 100  # max_week  # infect population larger, spreding time long
            data.rename(columns={'timestep': 'timestep', 'female': 'source', 'male': 'target'}, inplace=True)
        elif catagory == 'Bitcoin':
            start_interval = [1, 10]
            self.max_day = 80
            data.rename(columns={'timestep': 'timestep', 'source': 'source', 'target': 'target'}, inplace=True)
        elif catagory == 'Eu':
            start_interval = [1, 3]
            self.max_day = 10
            data.rename(columns={'timestep': 'timestep', 'source': 'source', 'target': 'target'}, inplace=True)
        elif catagory == 'math':
            start_interval = [1, 3]
            self.max_day = 254
            data.rename(columns={'timestep': 'timestep', 'source': 'source', 'target': 'target'}, inplace=True)
        elif catagory == 'msg':
            start_interval = [1, 3]
            self.max_day = 10
            data.rename(columns={'timestep': 'timestep', 'source': 'source', 'target': 'target'}, inplace=True)
        elif catagory == 'hos':
            start_interval = [1, 5]
            self.max_day = 70
            data.rename(columns={'timestep': 'timestep', 'source': 'source', 'target': 'target'}, inplace=True)

        data2 = data[ ( data['timestep']<= self.max_day ) ]#& ( data['timestep']>=min_day )]
        nodes = np.unique( data2[['source','target']].values ) #nodes已经按小到大排好序
        self.nodes_num = len( nodes )
        nodes_rename = dict( zip(nodes, range(len(nodes))) )
        data2['source'] = [ nodes_rename[x] for x in data2['source']]
        data2['target'] = [ nodes_rename[x] for x in data2['target']]

        self.g = nx.Graph()
        self.g.add_edges_from( data2[['source','target']].values ) #for neigh in hop1

        edge = {}
        for sou,tar,time in data2.values:
            if ( sou, tar ) in  edge.keys():
                edge[(sou,tar)][time] = 1
            else:
                edge[(sou, tar)] = np.zeros( self.max_day +1 )
                edge[(sou, tar)][time] = 1

            if directed == False:
                if ( tar, sou ) in  edge.keys():
                    edge[(tar, sou)][time] = 1
                else:
                    edge[(tar, sou)] = np.zeros( self.max_day +1 )
                    edge[(tar, sou)][time] = 1


        self.edge_index = np.array(list(edge.keys())) #[num_edges,2]
        self.edge_feature = np.array(list(edge.values())).astype( np.float32 ) #[num_edges, man_day+1]


        f1 = open(HOME + '/source/data/{0}_reocrd(p={1},q={2}).txt'.
                      format(catagory, p, q), 'r')
        record = eval(f1.read( ))
        f1.close()

        train, test = train_test_split( record, test_size=0.02, random_state=42 )
        valid, test = train_test_split( test, test_size=0.5, random_state=42 )
        if type == 'train':
            self.record = train
        elif type == 'valid':
            self.record = valid
        else:
            self.record = test

        self.sample_num = len( self.record )
        self.start_scope = np.arange(start_interval[0], start_interval[-1] + 1)
        self.type = type
        self.unobserved = unobserved

        print( '{0}_set, sample_num:{1}, edge_feature_shape:{2} node_num:{3} edge_index_shape:{4} unobserved:{5} catagory:{6}'.
               format( type, self.sample_num, self.edge_feature.shape,
                       self.nodes_num, self.edge_index.shape, unobserved, catagory ) )



    def __getitem__(self, index ):

        record = self.record[index]

        _, source, infect, recovery = record #infect--0,1,0 recovery--0,0,1

        # if self.type != 'train':
        #     np.random.seed( index )
        node_feature = np.zeros( [self.nodes_num, 4] ).astype( np.float32 ) #[nodes, 3]
        node_feature[:,0] = 1

        eliminate = np.ones( self.nodes_num ).astype( np.float32 ) * Neg #impossible to be source

        for n in infect:
            node_feature[n,:] = [ 0, 1, 0, 0 ]
            eliminate[ n ] = 0
        for n in recovery:
            node_feature[n,:] = [ 0, 0, 1, 0 ]
            eliminate[ n ] = 0

        # if self.type == 'test':
        #     np.random.seed( index )
        unob_index = np.random.random_sample( self.nodes_num ) < self.unobserved
        node_feature[ unob_index, : ] = [ 0, 0, 0, 1 ]


        edge_feature = self.edge_feature.copy()
        # start = np.random.choice( self.start_scope  )
        # edge_feature[:, :start] = 0

        # source_ = np.zeros( self.nodes_num ).astype( np.int )
        # source_[ source ] = 1

        temp = list(nx.neighbors(self.g, source)) #for hop1
        neigh = np.zeros( self.nodes_num ).astype( np.int )
        neigh[temp] = 1
        neigh[source] =1

        return node_feature, edge_feature, source, eliminate, neigh




    def __len__(self):

        return self.sample_num