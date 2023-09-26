from typing import Callable, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric.nn as nn


class BN( torch.nn.Module ):
    def __init__(self, num_node_features=3, num_edge_features=301, layers=5 ): #3 for SIR
        super().__init__()

        # print( num_node_features, num_edge_features )
        self.n = nn.Sequential( nn.Linear( in_features=num_node_features, out_features=16  ),
                                nn.ReLU(),
                                )
        self.e = nn.Sequential(nn.Linear(in_features=num_edge_features, out_features=128 ),
                               nn.ReLU(),
                               nn.Linear(in_features=128, out_features=64  ),
                               nn.ReLU(),
                               nn.Linear(in_features=64, out_features=32 ),
                               nn.ReLU(),
                               )


        self.bn1 = NNConv( in_channels=16, out_channels=32 )
        self.bn2 = NNConv( in_channels=32, out_channels=32 )
        self.bn3 = NNConv( in_channels=32, out_channels=32 )
        self.bn4 = NNConv( in_channels=32, out_channels=32 )
        self.bn5 = NNConv( in_channels=32, out_channels=32 )


        self.bn6 = NNConv( in_channels=32, out_channels=32 )
        self.bn7 = NNConv( in_channels=32, out_channels=32 )



        self.bn_last = nn.Linear( 32,1 ) #[bs,nodes,1]
        self.layers = layers


    def forward( self, node_feature, edge_feature, edge_index ):
        # node_feature-[bs,num_nodes, num_node_features], edge_index-[2, num_edges]
        # edege feature-[bs,num_edges,num_edge_feature]
        # print( node_feature.shape, edge_feature.shape )
        node_feature = self.n( node_feature )
        edge_feature = self.e( edge_feature )


        x = self.bn1( node_feature, edge_index, edge_feature )
        x = F.relu(x)

        if self.layers >= 2:
            x = self.bn2(x, edge_index, edge_feature)
            x = F.relu(x)

            if self.layers >= 3:
                x = self.bn3(x, edge_index, edge_feature)
                x = F.relu(x)

                if self.layers >= 4:
                    x = self.bn4(x, edge_index, edge_feature)
                    x = F.relu(x)

                    if self.layers >= 5:
                        x = self.bn5(x, edge_index, edge_feature)
                        x = F.relu(x)

                        if self.layers >= 6:
                            x = self.bn6(x, edge_index, edge_feature)
                            x = F.relu(x)

                            if self.layers >= 7:
                                x = self.bn7(x, edge_index, edge_feature)
                                x = F.relu(x)

        x = self.bn_last( x ) #[bs,nodes,1]
        x = torch.squeeze( x, dim=-1 ) #[bs,nodes]
        # print(x.shape)

        return x



class NNConv( MessagePassing):

    def __init__(self, in_channels: int, out_channels: int,  aggr: str = 'add', **kwargs): #
        super().__init__(aggr=aggr, **kwargs)
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Linear( in_features=in_channels+32, out_features=32  )


        self.lin = nn.Linear(in_channels, out_channels, bias=True)



    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                      edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate( edge_index, x=x, edge_attr=edge_attr, size=size )
        out += self.lin(x)

        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        x_j = torch.cat( [x_j, edge_attr], dim=-1 )
        x_j = self.net(x_j)
        return x_j






# class bn_layer( MessagePassing ):
#
#     def __init__( self, node_channels=3,  edge_channels=1 ,out_features=32 ):
#         super().__init__()
#         # self.node_channels = node_channels
#         # self.edge_channels = edge_channels
#         # self.out_features = out_features
#         # self.conv_e = nn.Sequential(
#         #     nn.Conv1d( in_features=1, out_features=32, kernel_size=7, ),
#         #     nn.ReLU(),
#         #     nn.Conv1d( in_features=32, out_features=32, kernel_size=7, ),
#         #     nn.ReLU(),
#         #     nn.Conv1d( in_features=32, out_features=32, kernel_size=7, ),
#         #     nn.ReLU(),
#         #     nn.Conv1d(in_features=32, out_features=out_features, kernel_size=7, ),
#         # )
#
#         self.lin_e = nn.Sequential(
#             nn.Linear(in_features=edge_channels, out_features=128 ),
#             nn.ReLU(),
#             nn.Linear(in_features=128, out_features=out_features ), )
#
#         self.nn = nn.Sequential(
#             nn.Linear( in_features=node_channels*2 + out_features, out_features=128 ),
#             nn.ReLU(),
#             # nn.Linear( in_features=128, out_features=128 ),
#             # nn.ReLU(),
#             nn.Linear( in_features=128, out_features=out_features),
#         )
#
#         # for neighbor
#         self.lin = nn.Linear( in_features=node_channels+out_features,
#                               out_features=out_features, bias=True )
#
#
#     def forward( self,x, edge_index, edge_attr ):
#         print(1)
#         edge_attr = self.edge_updater( edge_index, x, edge_attr )
#         edge_updater
#         print(2)
#         out = self.propagate( edge_index, x=x, edge_attr=edge_attr, size=None )
#
#         x = torch.cat( [x,out], dim=-1 ) #[nodes,n_feature+out_features],
#         x = self.lin( x )
#         return x, edge_attr
#
#
#     def message( self, edge_attr ): #[edges,feature], [edges,feature], [edges,temporal,feautre]
#         return edge_attr #[ bs, edges, feature ]
#
#
#     def edge_update( self, x_i, x_j, edge_attr ):
#         # if len(edge_attr.shape)==4:
#         #     edge_attr = self.conv_e( edge_attr )  # [ bs, edges, temporal2, feautre ]
#         #     edge_attr = edge_attr.sum( dim=-2 )  # [ bs, edges, feautre ]
#         # else:
#         edge_attr = self.lin_e( edge_attr )  # [ bs, edges,  feautre ]
#
#         edge_attr = torch.cat( [x_i, x_j, edge_attr], dim=-1 )  # [ bs, edges, feature ]
#         edge_attr = self.nn( edge_attr )
#
#         return edge_attr



