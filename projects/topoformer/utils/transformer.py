import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time
from scipy.spatial.distance import cosine
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # import pdb;pdb.set_trace()
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, use_counterfactual_att=False, use_counterfactual_dist_weight_matrix = False):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        if not use_counterfactual_att:
            self.fc_q = nn.Linear(d_model, h * d_k)
            self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.use_counterfactual_att = use_counterfactual_att
        self.use_counterfactual_dist_weight_matrix = use_counterfactual_dist_weight_matrix
        self.init_weights()
        
    def init_weights(self):
        if not self.use_counterfactual_att:
            nn.init.xavier_uniform_(self.fc_q.weight)
            nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        if not self.use_counterfactual_att:
            nn.init.constant_(self.fc_q.bias, 0)
            nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    # def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, way='mul', use_counterfactual_att=False, use_counterfactual_SPM=False):
    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, way='mul'):
        # import pdb;pdb.set_trace()
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        #print(queries.shape, keys.shape, values.shape)
        if self.use_counterfactual_att:
            q = None
            k = None
        else:
            q = self.fc_q(queries)
            k = self.fc_k(keys)
            q = q.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
            k = k.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        # q = self.fc_q(queries)
        #print(q.shape, b_s, nq, self.h, self.d_k)
        

        # k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        #通过v的形状推算出att的形状
        att_shape = (b_s, self.h, nq, nk)
        # att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if self.use_counterfactual_att:
            # att = torch.randn_like(att)
            att = torch.randn(att_shape, device=v.device, dtype=v.dtype)
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        
        if attention_weights is not None and self.use_counterfactual_dist_weight_matrix:
            attention_weights = torch.randn_like(attention_weights)
            # print('use_counterfactual_dist_weight_matrix is True')
        #构造att_weight
        if attention_weights is not None:
            #处理正常的Attention
            if way == 'mul':
                att = att * attention_weights
            elif way == 'add':
                # print(att.shape, attention_weights.shape, '<< att shape; add')
                # import pdb;pdb.set_trace()
                att = att + attention_weights
            elif way == 'had':
                # import pdb; pdb.set_trace()
                att = att @ attention_weights
            else:
                raise NotImplementedError(way)
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
            if self.use_counterfactual:
                counterfacual_att = counterfacual_att.masked_fill(attention_mask, -np.inf)
        # import pdb;pdb.set_trace()
        att = torch.softmax(att, -1)
        # import pdb;pdb.set_trace()
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        # if use_counterfactual_att:
        #     counterfacual_att = torch.softmax(counterfacual_att, -1)
        #     counterfacual_out = torch.matmul(counterfacual_att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        #     counterfacual_out = self.fc_o(counterfacual_out)  # (b_s, nq, d_model)
        #     return out, counterfacual_out
        return out

    def forward_faster(self, queries, keys, values, attention_pos, attention_weights, way='mul'):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_pos: Mask over attention values (b_s, nq, pk). True indicates masking indices. pk << nk_real
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, pk).
        :return:
        '''
        b_s, nq, d_model = queries.shape
        nk = keys.shape[1]
        pk = attention_pos.shape[2]
        # print("attention_pos0", attention_pos.shape) #4 256 20
        #print(queries.shape, keys.shape, values.shape)
        #print(q.shape, b_s, nq, self.h, self.d_k)
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)[: ,:, :, None, :]  # (b_s, h, nq, 1, d_k)

        attention_pos = attention_pos.view(b_s, nq*pk)

        k = self.fc_k(keys)
        v = self.fc_v(values)
        i_ind = torch.arange(b_s)[:, None].type_as(attention_pos) * nq
        attention_pos = attention_pos + i_ind.long()  # faster...
        # return queries
        # print("attention_pos", attention_pos.shape) #4, 5120
        # print("k", k.view(b_s*nq, -1).shape) #1024, 128
        # print("b_s, nq, pk, self.h, self.d_k", b_s, nq, pk, self.h, self.d_k) #4 256 20 4 32
        # print("k[attention_pos]", k.view(b_s*nq, -1)[attention_pos].shape)
        k = k.view(b_s*nq, -1)[attention_pos].view(b_s, nq, pk, self.h, self.d_k)  # (b_s, nq, pk, h, dk)
        v = v.view(b_s*nq, -1)[attention_pos].view(b_s, nq, pk, self.h, self.d_v)  # (b_s, nq, pk, h, dv)
        # return queries  # 26ms
        # i_ind = torch.arange(b_s)[:, None].type_as(attention_pos).repeat(1, nq*pk)
        # k = k[i_ind, attention_pos].view(b_s, nq, pk, self.h, self.d_k)  # (b_s, nq, pk, h, dk)
        # v = v[i_ind, attention_pos].view(b_s, nq, pk, self.h, self.d_v)  # (b_s, nq, pk, h, dv)

        k = k.permute(0, 3, 1, 4, 2)  # (b_s, h, nq, d_k, pk)
        v = v.permute(0, 3, 1, 2, 4)  # (b_s, h, nq, pk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, 1, p_k)
        if attention_weights is not None:
            attention_weights = attention_weights[:, :, :, None, :]
            if way == 'mul':
                att = att * attention_weights
            elif way == 'add':
                # print(att.shape, attention_weights.shape, '<< att shape; add')
                att = att + attention_weights
            else:
                raise NotImplementedError(way)
        att = torch.softmax(att, -1)  # softmax;
        out = torch.matmul(att, v)  # (b_s, h, nq, 1, d_v)
        out = out.permute(0, 2, 1, 4, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out
class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, m = 20)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, way='mul'):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights, way)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights, way)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

    def forward_faster(self, queries, keys, values, attention_pos, attention_weights, way):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention.forward_faster(q_norm, k_norm, v_norm, attention_pos, attention_weights, way)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention.forward_faster(queries, keys, values, attention_pos, attention_weights, way)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

class CounterfactualAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''
    # import pdb;pdb.set_trace()
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None,
                use_counterfactual_dist_weight_matrix = False,
                use_counterfactual_att = True,):
        super(CounterfactualAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, m = 20)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, use_counterfactual_att=use_counterfactual_att, use_counterfactual_dist_weight_matrix=use_counterfactual_dist_weight_matrix)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, way='mul'):
        # import pdb;pdb.set_trace()
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights, way)
            out = queries + self.dropout(torch.relu(out))
        else:
            # import pdb;pdb.set_trace()
            out = self.attention(queries, keys, values, attention_mask, attention_weights, way)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

    def forward_faster(self, queries, keys, values, attention_pos, attention_weights, way):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention.forward_faster(q_norm, k_norm, v_norm, attention_pos, attention_weights, way)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention.forward_faster(queries, keys, values, attention_pos, attention_weights, way)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out
class lclcRelationshipHead(nn.Module):
    def __init__(self,
                in_channels_o1,
                in_channels_o2=None,
                shared_param=True,
                use_one_MLP_as_Edges_Head = False):
        super().__init__()
        self.use_one_MLP_as_Edges_Head = use_one_MLP_as_Edges_Head
        if not use_one_MLP_as_Edges_Head:
            self.MLP_o1 = MLP(in_channels_o1, in_channels_o1, 128, 3)
            self.shared_param = shared_param
            if shared_param:
                self.MLP_o2 = self.MLP_o1
            else:
                self.MLP_o2 = MLP(in_channels_o2, in_channels_o2, 128, 3)
            self.classifier = MLP(256, 256, 1, 3)
        else:
             self.classifier = MLP(256*2, 256*2, 1, 3)

    def forward(self, o1_feats, o2_feats, q_1_end_point=None, q_2_start_point=None):
        # feats: B, num_query, num_embedding
        # import pdb;pdb.set_trace()
        if not self.use_one_MLP_as_Edges_Head:
            o1_embeds = self.MLP_o1(o1_feats)
            o2_embeds = self.MLP_o2(o2_feats)
        else:
            o1_embeds = o1_feats
            o2_embeds = o2_feats
            # print('use one MLP as Edges Head')
        num_query_o1 = o1_embeds.size(1)
        num_query_o2 = o2_embeds.size(1)
        o1_tensor = o1_embeds.unsqueeze(2).repeat(1, 1, num_query_o2, 1)
        o2_tensor = o2_embeds.unsqueeze(1).repeat(1, num_query_o1, 1, 1)

        relationship_tensor = torch.cat([o1_tensor, o2_tensor], dim=-1)
        relationship_pred = self.classifier(relationship_tensor)
        return relationship_pred
class topologicHead(nn.Module):
    def __init__(self,
                 in_channels_o1,
                 in_channels_o2,
                 distance_mapping_function=None,
                 alpha=0.2,
                 lambd=2.0,
                 lamda_sim=1.0,
                 lamda_geo=1.0,):
        super().__init__()
        #Similarity Topologic
        self.MLP_o1 = MLP(in_channels_o1, in_channels_o1, 128, 3)
        self.MLP_o2 = MLP(in_channels_o2, in_channels_o2, 128, 3)
        #Geometric Distance Topology
        if distance_mapping_function == 'topologic':
            self.distance_mapping_function = distance_mapping_function
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
            self.lambd = nn.Parameter(torch.tensor(lambd, dtype=torch.float32))
        elif distance_mapping_function == 'sigmod':
            self.distance_mapping_function = distance_mapping_function
        elif distance_mapping_function == 'gau':
            self.distance_mapping_function = distance_mapping_function
        elif distance_mapping_function == 'tanh':
            self.distance_mapping_function = distance_mapping_function
        elif distance_mapping_function == 'nothing':
            self.distance_mapping_function = distance_mapping_function
        else:
            self.distance_mapping_function = None
            raise ValueError('No distance mapping function!')
        if self.distance_mapping_function != 'nothing':
            self.lamda_sim = nn.Parameter(torch.tensor(lamda_sim, dtype=torch.float32))
            self.lamda_geo = nn.Parameter(torch.tensor(lamda_geo, dtype=torch.float32))
            # print(self.lamda_sim, self.lamda_geo)
        # import pdb;pdb.set_trace()          
    def forward(self, o1_feats, o2_feats, q_1_end_point, q_2_start_point):
        #Similarity Topologic
        # import pdb;pdb.set_trace()
        o1_feats = self.MLP_o1(o1_feats)#[1, 200, 256]
        o2_feats = self.MLP_o2(o2_feats)#[1, 200, 256]
        #检查
        if torch.isnan(o1_feats).any() or torch.isinf(o1_feats).any():
            print("o1_feats contains NaN or Inf")
        if torch.isnan(o2_feats).any() or torch.isinf(o2_feats).any():
            print("o2_feats contains NaN or Inf")
        #print("o1_feats stats - min: {}, max: {}, mean: {}".format(o1_feats.min(), o1_feats.max(), o1_feats.mean()))
        #print("o2_feats stats - min: {}, max: {}, mean: {}".format(o2_feats.min(), o2_feats.max(), o2_feats.mean()))
        lane_similarity = torch.matmul(o1_feats, o2_feats.transpose(2, 1))#[1, 200, 200]
        #检查这个值的范围
        
        #if torch.isnan(lane_similarity).any() or torch.isinf(lane_similarity).any():
            #print("lane_similarity contains NaN or Inf after matmul")
        #print("lane_similarity stats before sigmoid - min: {}, max: {}, mean: {}".format(lane_similarity.min(), lane_similarity.max(), lane_similarity.mean()))
        # lane_similarity = lane_similarity.sigmoid()#[1, 200, 200]
        # if torch.isnan(lane_similarity).any() or torch.isinf(lane_similarity).any():
        #     print("lane_similarity contains NaN or Inf after sigmoid")
        if self.distance_mapping_function != 'nothing':
            #Geometric Distance Topology
            q_1_end_point = q_1_end_point.unsqueeze(1).repeat(1, q_2_start_point.shape[1], 1, 1)#[1, 200, 200, 3]
            q_2_start_point = q_2_start_point.unsqueeze(2).repeat(1, 1, q_1_end_point.shape[2], 1)#[1, 200, 200, 3]
            # import pdb;pdb.set_trace()
            distance_matrix = torch.abs(q_1_end_point - q_2_start_point).sum(dim=-1)#[1, 200, 200]
            # distance_matrix = distance_matrix.unsqueeze(-1)#[1, 200, 200, 1]
            epsilon = 1e-10

            # 检查是否存在全零张量，并添加 epsilon
            distance_matrix = distance_matrix + (distance_matrix == 0.0).float() * epsilon
            if self.distance_mapping_function == 'topologic':
                # 定义一个非常小的值 epsilon
                
                sigma = torch.std(distance_matrix)
                if torch.isnan(sigma):
                    import pdb;pdb.set_trace()
                # 计算映射函数 f(x) = e^(-x^α / (λ * σ))
                exponent = -torch.pow(distance_matrix, self.alpha) / (self.lambd * sigma)
                distance_matrix = torch.exp(exponent)#[1, 200, 200]
            elif self.distance_mapping_function == 'sigmod':
                distance_matrix = torch.sigmoid(distance_matrix)#[1, 200, 200]
            elif self.distance_mapping_function == 'gau':
                distance_matrix = torch.exp(-torch.pow(distance_matrix, 2)/2)#[1, 200, 200]
            elif self.distance_mapping_function == 'tanh':
                distance_matrix = torch.tanh(distance_matrix)#[1, 200, 200]
            else:
                print('No distance mapping function!')
            # distance_matrix = distance_matrix.squeeze(-1)#[1, 200, 200]
            
            #Add the two similarity matrix      
            relationship_pred = lane_similarity * self.lamda_sim + distance_matrix * self.lamda_geo#[1, 200, 200]
            relationship_pred = relationship_pred.unsqueeze(-1)#[1, 200, 200, 1]
        else:
            relationship_pred = lane_similarity.unsqueeze(-1)#[1, 200, 200, 1]
        return relationship_pred

