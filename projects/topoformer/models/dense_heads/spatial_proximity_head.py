import torch.nn as nn
from mmdet.models import HEADS, build_loss
import torch
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
import copy
import sys
from projects.topoformer.utils.transformer import MultiHeadAttention, CounterfactualAttention, lclcRelationshipHead, topologicHead

@HEADS.register_module()
class SpatialProximityHead(nn.Module):
    """
    利用节点终点和起点之间的空间接近性矫正lclc head
    这是其中的一层,每一层都有对应的k值
    """

    def __init__(self, 
                input_dim, 
                hidden_dim, 
                output_dim, 
                transformer_num_layers=6,
                num_layers=2,
                loss_rel=None,
                num_query=200,
                num_heads=4,
                dropout=0.1,
                use_dist_weight_matrix=True,
                use_counterfactual=False,
                use_counterfactual_dist_weight_matrix=False,
                use_one_MLP_as_Edges_Head = False,
                counterfactual_type='random',
                counterfactual_prob=0.5,
                use_topologic=False,
                distance_mapping_function=None,
                alpha=0.2,
                lambd=2.0,
                lamda_sim=1,
                lamda_geo=1,
                pts_dim =3,
                att_spm_type='add'
                 ):
        super().__init__()
        self.pts_dim = pts_dim
        self.att_spm_type = att_spm_type
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(
        #     nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        # )
        self.embed_dims=hidden_dim
        self.loss_cls = build_loss(loss_rel)
        self.num_pred = transformer_num_layers
        # self.use_sigmod = use_sigmod
        self.num_query = num_query
        # import pdb;pdb.set_trace()
        self.use_counterfactual = use_counterfactual
        self.use_counterfactual_dist_weight_matrix = use_counterfactual_dist_weight_matrix
        self.use_dist_weight_matrix = use_dist_weight_matrix
        self.use_one_MLP_as_Edges_Head = use_one_MLP_as_Edges_Head
        self._init_layers()
        if self.use_counterfactual:
            self.counterfactual_self_attn = nn.ModuleList(
                CounterfactualAttention(d_model=hidden_dim, d_k=hidden_dim // num_heads, d_v=hidden_dim // num_heads, h=num_heads, use_counterfactual_dist_weight_matrix=self.use_counterfactual_dist_weight_matrix) for i in range(1))
            self.self_attn = nn.ModuleList(
                MultiHeadAttention(d_model=hidden_dim, d_k=hidden_dim // num_heads, d_v=hidden_dim // num_heads, h=num_heads) for i in range(self.num_layers))
        else:
            self.self_attn = nn.ModuleList(
                MultiHeadAttention(d_model=hidden_dim, d_k=hidden_dim // num_heads, d_v=hidden_dim // num_heads, h=num_heads) for i in range(self.num_layers))
        self.features_concat = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim, 1),
        )
        self.use_topologic = use_topologic
        if self.use_topologic:
            #使用topologic的方法，https://arxiv.org/pdf/2405.14747
            self.rel_predictor = topologicHead(
                in_channels_o1=hidden_dim,
                in_channels_o2=hidden_dim,
                distance_mapping_function=distance_mapping_function,
                alpha=alpha,
                lambd=lambd,
                lamda_sim=lamda_sim,
                lamda_geo=lamda_geo,
            )
        else:
            self.rel_predictor = lclcRelationshipHead(
                in_channels_o1=hidden_dim,
                in_channels_o2=hidden_dim,
                shared_param=False,
                use_one_MLP_as_Edges_Head= self.use_one_MLP_as_Edges_Head)
        self.transformer_num_layers = transformer_num_layers
        self.loss_rel = build_loss(loss_rel)
        
        self.counterfactual_prob = counterfactual_prob
        self.counterfactual_type = counterfactual_type
    def _init_layers(self):
        pass
    def forward(self, outs):
        """
        Args:
           outs['history_states']: [6, 1, 200, 256]
           outs['outputs_coords']: [6, 1, 200, 33]
        Returns:
            outs['all_lclc_preds']:[6, 1, 200, 200]
        """
        
        query_feats = outs['history_states'][-1]#[1, 200, 256]
        query_coords = outs['all_lanes_preds'][-1]#[1, 200, 33]

        key_feats = outs['history_states'][-1]#[1, 200, 256]
        key_coords = outs['all_lanes_preds'][-1]#[1, 200, 33]
        # import pdb;pdb.set_trace()
        #构建终点和起点之间的空间接近性
        if self.pts_dim == 3:
            query_coords =  query_coords.view(query_coords.size(0), query_coords.size(1), query_coords.size(2)//3, 3)
            key_coords =  key_coords.view(key_coords.size(0), key_coords.size(1), key_coords.size(2)//3, 3)
        else:
            query_coords =  query_coords.view(query_coords.size(0), query_coords.size(1), query_coords.size(2)//2, 2)
            key_coords =  key_coords.view(key_coords.size(0), key_coords.size(1), key_coords.size(2)//2, 2)
        query_end_point = query_coords[:, :, -1, :]#[1, 200, 3]
        key_start_point = key_coords[:, :, 0, :]#[1, 200, 3]

        if self.use_dist_weight_matrix:
            # Attention Weight
            
            N_K = query_end_point.shape[1]
            center_A = key_start_point[:, None, :, :].repeat(1, N_K, 1, 1)
            center_B = query_end_point[:, :, None, :].repeat(1, 1, N_K, 1)
            dist = (center_A - center_B).pow(2)
            dist = torch.sqrt(torch.sum(dist, dim=-1))[:, None, :, :]
            dist_weights = 1 / (dist+1e-2)
            norm = torch.sum(dist_weights, dim=2, keepdim=True)
            dist_weights = dist_weights / norm
            zeros = torch.zeros_like(dist_weights)
            # slightly different with our ICCV paper, which leads to higher results (3DVG-Transformer+)
            dist_weights = torch.cat([dist_weights, -dist, zeros, zeros], dim=1).detach()
            if self.att_spm_type == 'add':
                attention_matrix_way = 'add'
            elif self.att_spm_type == 'had':
                attention_matrix_way = 'had'
            elif self.att_spm_type == 'mul':
                attention_matrix_way = 'mul'
        else:
            dist_weights = None
            attention_matrix_way = 'mul'
        init_query_feats = self.features_concat(query_feats)
        # import pdb;pdb.set_trace()
        
        #2025.1.24这里的反事实应该添加到第二层中，而不是第一层
        if self.use_counterfactual:
            for i in range(self.num_layers):
                if i == 0:
                    query_feats = self.self_attn[i](init_query_feats, init_query_feats, init_query_feats, attention_weights=dist_weights, way=attention_matrix_way)
                elif i == self.num_layers - 1:
                    # query_feats, counterfactual_query_feats = self.counterfactual_self_attn(query_feats, query_feats, query_feats, attention_weights=dist_weights, way=attention_matrix_way)
                    
                    query_feats = self.self_attn[i](query_feats, query_feats, query_feats, attention_weights=dist_weights, way=attention_matrix_way)
                    # import pdb;pdb.set_trace()
                    counterfactual_query_feats = self.counterfactual_self_attn[0](query_feats, query_feats, query_feats, None, dist_weights, attention_matrix_way)
                else:
                    query_feats = self.self_attn[i](query_feats, query_feats, query_feats, attention_weights=dist_weights, way=attention_matrix_way)
                    # counterfactual_query_feats = self.self_attn(counterfactual_query_feats, counterfactual_query_feats, counterfactual_query_feats, attention_weights=dist_weights, way=attention_matrix_way)
        else:
            for i in range(self.num_layers):
                if i == 0:
                    query_feats = self.self_attn[i](init_query_feats, init_query_feats, init_query_feats, attention_weights=dist_weights, way=attention_matrix_way)
                else:
                    query_feats = self.self_attn[i](query_feats, query_feats, query_feats, attention_weights=dist_weights, way=attention_matrix_way)

        
        # query_feats = query_feats.permute(0, 2, 1)
        #之后再计算连接性
        lclc_pred = self.rel_predictor(query_feats, query_feats, query_end_point, key_start_point)
        # import pdb;pdb.set_trace()
        lclc_pred = lclc_pred.squeeze(-1)
        if self.use_counterfactual:
            counterfactual_lclc_pred = self.rel_predictor(counterfactual_query_feats, counterfactual_query_feats, query_end_point, key_start_point)
            counterfactual_lclc_pred = counterfactual_lclc_pred.squeeze(-1)
            lclc_pred = lclc_pred - counterfactual_lclc_pred
        lclc_pred = lclc_pred.unsqueeze(0).repeat(self.transformer_num_layers, 1, 1, 1)
        outs['all_lclc_preds'] = lclc_pred
        return outs
        #基于L1距离计算空间接近性矩阵
        # 定义距离函数，这里使用 L1 距离
        # def l1_distance(x, y):
        #     return torch.sum(torch.abs(x - y), dim=-1)

        # # 计算距离矩阵
        # batch_size, num_queries, num_dims = query_end_point.shape
        # _, num_keys, _ = key_start_point.shape

        # # 扩展维度以便进行广播
        # query_end_point_expanded = query_end_point.unsqueeze(2).expand(batch_size, num_queries, num_keys, num_dims)
        # key_start_point_expanded = key_start_point.unsqueeze(1).expand(batch_size, num_queries, num_keys, num_dims)

        # # 计算距离
        # distances = l1_distance(query_end_point_expanded, key_start_point_expanded)  # [batch_size, num_queries, num_keys]

        # # 添加小常数以避免无穷大
        # epsilon = 1e-8
        # inverse_distances = 1.0 / (distances + epsilon)

        # # 归一化操作
        # mean_inverse_distance = torch.mean(inverse_distances, dim=(1, 2), keepdim=True)
        # normalized_distances = inverse_distances / mean_inverse_distance

        # # k-近邻搜索
        # k = 5  # 假设 k = 5
        # _, indices = torch.topk(normalized_distances, k, dim=-1, largest=True, sorted=False)

        # # 创建稀疏矩阵
        # sparse_matrix = torch.full_like(normalized_distances, float('-inf'))
        # batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(batch_size, num_queries, k)
        # query_indices = torch.arange(num_queries).unsqueeze(0).unsqueeze(2).expand(batch_size, num_queries, k)

        # sparse_matrix[batch_indices, query_indices, indices] = normalized_distances[batch_indices, query_indices, indices]
        # #[1, 200, 200]
        # #import pdb;pdb.set_trace()
        # # print(sparse_matrix)
    def loss(self, lclc_preds,lclc_targets):
        lclc_preds = lclc_preds[-1]
        lclc_targets = lclc_targets[-1]
        # print('DEBUG shapes: pred', lclc_preds.shape, 'target', lclc_targets.shape)
        # import pdb;pdb.set_trace()
        loss_lclc = self.loss_rel(lclc_preds, lclc_targets)
        return  loss_lclc
    def get_lclc(self, preds_dicts):

        all_lclc_preds = preds_dicts['all_lclc_preds'][-1].squeeze(-1).sigmoid().detach().cpu().numpy()
        all_lclc_preds = [_ for _ in all_lclc_preds]

        return all_lclc_preds
