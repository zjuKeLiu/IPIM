"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
import copy,math
# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

from alignn.models.utils import RBFExpansion
from alignn.utils import BaseSettings


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    num_classes: int = 2

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self, in_features: int, out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


def clone(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class CrystalEncoder(nn.Module):
    def __init__(self, layer, N, d_model, dropout=0.1):
        super(CrystalEncoder, self).__init__()
        self.layers = clone(layer,N)
        self.norm = nn.LayerNorm(layer.layerNormSize)
        self.unit_position_encoding = UnitPositionalEncoding(dropout, 0, d_model)
    
    def forward(self, x, lattice):
        num_mol = x.size(0)
        padded_coords_matrix = self.unit_position_encoding.get_dist_matrix(num_mol, lattice)

        for layer in self.layers:
            x = layer(x, padded_coords_matrix, self.unit_position_encoding)
        return self.norm(x)


class UnitPositionalEncoding(nn.Module):
    def __init__(self, dropout, pad_num, d_model):
        super(UnitPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(3, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.pad_num = pad_num
    
    def get_dist_matrix(self, num_mol, lattice):
        #num_mol = x.size(0) #batch_size
        #max_num_atom = x.size(1)  #max_seq_len
        temp_padded_coords = torch.zeros((num_mol, 27, 3)) #3维坐标点, 27 个邻接unit
        #print("****************************************************************")
        #print("lattice_train.shape", lattice)
        #print("****************************************************************")
        lattice_reshape = lattice.reshape(-1,3,3)

        for i in range(num_mol):
            
            lattice_a = lattice_reshape[i,0,:]
            lattice_b = lattice_reshape[i,1,:]
            lattice_c = lattice_reshape[i,2,:]
            count = 0
            #padded_mol = torch.from_numpy(padded_mol)
            for j in [0, -1, 1]:
                for k in [0, -1, 1]:
                    for l in [0, -1, 1]:
                        tran_vec = j*lattice_a+k*lattice_b+l*lattice_c
                        #print(padded_mol.shape,tran_vec.shape)
                        temp_padded_coords[i][count] = tran_vec
                        count += 1
        return temp_padded_coords

    def forward(self, x, padded_coords, central_atom_index):
        temp_padded_coords = torch.tensor(padded_coords - padded_coords[:, central_atom_index, :].reshape(padded_coords.shape[0],1,padded_coords.shape[2]),dtype=torch.float32)
        ###########################cuda()#############################
        return self.norm(x + self.dropout(gelu(self.linear(temp_padded_coords.cuda()))))
        #return self.norm(x + self.dropout(gelu(self.linear(temp_padded_coords))))


class SublayerConnection(nn.Module):
    def __init__(self, layerNormSize, p):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(layerNormSize)
        self.dropout = nn.Dropout(p)

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class CrystalEncoderLayer(nn.Module):
    def __init__(self,layerNormSize,self_atten,feed_forward,dropout):
        # self_atten = MultiHeadAttention, feed_forward = PositionwiseFeedForward
        super(CrystalEncoderLayer, self).__init__()
        self.self_atten = self_atten
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(layerNormSize,dropout),2)
        self.layerNormSize = layerNormSize

    def forward(self,x, padded_coords_matrix, position_encodeing):
        # num_mol * 27 * max_num_atom * emdding_dim
        # mask : num_mol * max_num_atom

        #############################cuda()############################################
        mask_pre_use = torch.zeros(27).cuda()
        #mask_pre_use = torch.zeros(27)
        mask_pre_use[0] = 1
        atom_reps = position_encodeing(x.unsqueeze(1), padded_coords_matrix, 0)
        atom_reps_reshape = atom_reps.reshape(atom_reps.shape[0],-1,atom_reps.shape[-1])
        atom_reps_res_temp = self.sublayer[0](atom_reps_reshape,lambda atom_reps_reshape:self.self_atten(atom_reps_reshape,atom_reps_reshape,atom_reps_reshape))
        atom_reps_res = mask_pre_use.unsqueeze(-1)*(atom_reps_res_temp.reshape(atom_reps.shape[0], 27, atom_reps.shape[2]))
        return self.sublayer[1](atom_reps_res.sum(1),self.feed_forward)


def attention(query,key,value,mask=None,dropout=None):
    """
    :param query: (batch_size,h,seq_len,embedding)
    :param key:
    :param value:
    :param mask: (batch_size,1,1,seq_len)
    :param dropout:
    :return: (batch_size,h,seq_len,embedding)
    """
    d_k = query.size(-1)
    score = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)
    p_atten = F.softmax(score,dim=-1)
    if dropout is not None:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten,value),p_atten


class MultiHeadAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.d_model = d_model
        self.atten = None
        self.dropout = nn.Dropout(dropout)
        self.linears = clone(nn.Linear(d_model,d_model),4)

    def forward(self,query,key,value,mask=None):
        batch_size = query.size(0)
        #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        #print(query.shape, batch_size, self.h, self.d_k, self.d_model)
        #for l,x in zip(self.linears,(query,key,value)):
        #    print(l(x).view(batch_size,-1,self.h,self.d_k).transpose(1,2))
        query,key,value = [l(x).view(batch_size,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key,value))]
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size,1,dk) => (batch_size,1,1,seq_len)
        x,self.atten = attention(query,key,value,mask,self.dropout)
        return self.linears[-1](x.transpose(1,2).contiguous().view(batch_size,-1,self.d_model))


class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification
        self.ff_unit = PositionwiseFeedForward(config.hidden_features, config.hidden_features, 0.2)
        self.atten_unit = MultiHeadAttention(config.hidden_features, config.hidden_features)
        self.crystal_encoder = CrystalEncoder(CrystalEncoderLayer(config.hidden_features, copy.deepcopy(self.atten_unit), copy.deepcopy(self.ff_unit), 0.2),
                               3, config.hidden_features, dropout=0.1)
        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features,),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1, vmax=1.0, bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(config.hidden_features, config.hidden_features,)
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.
        
        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        
        if len(self.alignn_layers) > 0:
            g, lg, lattice = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        if len(lattice) > 1:
            lattice_stack = torch.stack(lattice,0)
        else:
            lattice_stack = lattice[0]
        out = self.crystal_encoder(h, lattice_stack)
        out = self.fc(out)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)
