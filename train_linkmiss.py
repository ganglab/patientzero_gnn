import pandas as pd
import numpy as np
import networkx as nx
import gnn_model as gnn
from load_linkmiss import  Custom_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import torch


HOME = os.environ['HOME']


batch_size = 128
lr = 0.002
epoch = 200
k = 5 #acc in topk


def Backtracking( catagory = 'sex', p=0.3, q=0.01, test=False, link_miss=0. ):


    train_dataset = Custom_dataset( catagory=catagory, p=p, q=q, type='train', link_miss=link_miss )
    # train_loader = DataLoader( dataset=train_dataset, batch_size= batch_size, shuffle=True, num_workers=1,
    #                            persistent_workers=True )

    BN = gnn.BN( num_node_features=3, num_edge_features=train_dataset.max_day+1 ).cuda() # 3 for SIR, 2 for SI

    op = torch.optim.Adam(BN.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)  # , weight_decay=5e-4 )
    path = HOME + '/source/model_saver/{0}/p={1},q={2},link_miss={3}'.format( catagory, p, q, link_miss )
    if not os.path.exists(path):
        os.makedirs(path)
    cd = path + '/{0}.pth'.format('Backtracking')

    nodes_num = train_dataset.nodes_num


    if test == True:
        checkpoint = torch.load(cd)
        BN.load_state_dict(checkpoint['model'])
        op.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['epoch']

        test_dataset = Custom_dataset( catagory=catagory, p=p, q=q, type='test', link_miss=link_miss )
        test_loader = DataLoader( dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                 persistent_workers=True )
        LOGIT, LABEL, NEIGH = [], [], []
        for node_feature, edge_feature, source, eliminate, neigh in test_loader:
            node_feature = node_feature.to('cuda')
            edge_feature = edge_feature.to('cuda')
            source = source.to('cuda')
            eliminate = eliminate.to('cuda')
            neigh = neigh.to('cuda')

            with torch.no_grad():
                logit = BN(node_feature, edge_feature, test_dataset.edge_index)
            logit += eliminate
            LOGIT.append(logit)
            LABEL.append(source)
            NEIGH.append(neigh)

        top1_test, topk_test, hop1_test = measure(LOGIT, LABEL, NEIGH, nodes_num)
        print('test_epoch:{0},top1:{1}, topk:{2}, hop1:{3}'. format( step, top1_test, topk_test, hop1_test) )
        return top1_test, topk_test, hop1_test
        # top1_best_vaild, topk_best_vaild, hop1_best_vaild = top1_test, topk_test, hop1_test
        # step = 1
    else:
        if link_miss > 0.2:
            path_p = HOME + '/source/model_saver/{0}/p={1},q={2},link_miss={3}'.format(catagory, p, q, round(link_miss-0.2,1) )
            cd_p = path_p + '/{0}.pth'.format('Backtracking')

        else:
            path_p = HOME + '/source/model_saver/{0}/p={1},q={2}_Exp'.format( catagory, p, q )
            cd_p = path_p + '/{0}.pth'.format('Backtracking')

        checkpoint = torch.load(cd_p)
        BN.load_state_dict(checkpoint['model'])
        op.load_state_dict(checkpoint['optimizer'])
        step = 1
        top1_best_vaild = 0

    valid_dataset = Custom_dataset(catagory=catagory, p=p, q=q, type='valid', link_miss=link_miss )
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                              persistent_workers=True)  #

    for i in range( step, epoch+1 ):
        LOSS, LOGIT, LABEL, NEIGH = [], [], [], []
        train_dataset.misslink() #
        print( 'train_epoch:{0}, edge_index_shape:{1}'.format( i, train_dataset.edge_index.shape ) )
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                   persistent_workers=True )
        for node_feature, edge_feature, source, eliminate, neigh in train_loader:  #
            node_feature = node_feature.to('cuda')
            edge_feature = edge_feature.to('cuda')
            source = source.to('cuda')
            eliminate = eliminate.to('cuda')
            neigh = neigh.to('cuda')


            op.zero_grad()
            logit = BN( node_feature, edge_feature, train_dataset.edge_index ) #[bs,num_nodes]
            logit += eliminate #expertise
            loss = F.cross_entropy( logit, source )
            loss.backward()
            op.step()

            LOSS.append( loss.detach().cpu().numpy()  )
            LOGIT.append( logit )
            LABEL.append( source )
            NEIGH.append(neigh)

        top1, topk, hop1 = measure(LOGIT, LABEL, NEIGH, nodes_num)
        print( 'train_epoch:{0}, loss:{1}, top1:{2}, topk:{3}, hop1:{4}'.format(i,np.mean( LOSS ), top1, topk, hop1) )

        # if i == 1:
        if i % 20 == 0:
            #valid
            LOGIT, LABEL, NEIGH = [], [], []
            for node_feature, edge_feature, source, eliminate, neigh in valid_loader:
                node_feature = node_feature.to('cuda')
                edge_feature = edge_feature.to('cuda')
                source = source.to('cuda')
                eliminate = eliminate.to('cuda')
                neigh = neigh.to('cuda')

                with torch.no_grad():
                    logit = BN(node_feature, edge_feature, valid_dataset.edge_index)
                logit += eliminate
                LOGIT.append(logit)
                LABEL.append(source)
                NEIGH.append(neigh)

            top1_vaild, topk_vaild, hop1_vaild = measure(LOGIT, LABEL, NEIGH, nodes_num)


            if top1_vaild >= top1_best_vaild:
                top1_best_vaild, topk_best_vaild, hop1_best_vaild = top1_vaild, topk_vaild, hop1_vaild

                state = {'model': BN.state_dict(), 'optimizer': op.state_dict(), 'epoch': i}
                torch.save(state, cd)


            print('valid_epoch:{0},top1:{1}, topk:{2}, hop1:{3}, top1_best:{4}, topk_best:{5}, hop1_best:{6}'.
                  format(i, top1_vaild, topk_vaild, hop1_vaild, top1_best_vaild, topk_best_vaild, hop1_best_vaild))





def measure( LOGIT, LABEL, NEIGH, nodes_num ):
    LOGIT, LABEL, NEIGH = torch.cat(LOGIT, dim=0), torch.cat(LABEL, dim=0), torch.cat(NEIGH, dim=0)
    N = LOGIT.shape[0]  # samples_num
    #top
    _, maxk = torch.topk(LOGIT, k, dim=-1)  # [sample, nodes] - [samples,k]
    LABEL = LABEL.view(-1, 1)  # [samples] - [samples,1]
    top1 = (LABEL == maxk[:, 0:1]).sum().item() / N
    topk = (LABEL == maxk).sum().item() / N
    #hop
    source_p = F.one_hot( maxk[:,0], num_classes= nodes_num ) #[samples] - [samples, nodes_num]
    source_p2 = NEIGH * source_p #element: cover 1*1, no cover 0*1
    hop1 = ( N- (source_p.sum() - source_p2.sum()).item() )/ N

    return top1, topk, hop1




if __name__ == '__main__':
    gpu_id = 1
    torch.cuda.set_device( gpu_id )
    results = {}
    # for cata in ['Bitcoin', 'Eu', 'msg', 'hos']:
    for cata in ['sex']:
        temp_ = []
        for link_miss in [ 0.2, 0.4, 0.6, 0.8 ]:
            # temp = Backtracking( catagory=cata, p=0.3, q=0.01, test=False, link_miss=link_miss )
            temp = Backtracking(catagory=cata, p=0.8, q=0.05, test=True, link_miss=link_miss)
            # temp = Backtracking( catagory = 'sex', p=0.8, q=0.05, test=False, link_miss=link_miss  )
            # temp = Backtracking( catagory = 'sex', p=0.3, q=0.01, test=True, link_miss=link_miss  )
            temp_.append(temp)
        results[cata] = temp_
    print(results)