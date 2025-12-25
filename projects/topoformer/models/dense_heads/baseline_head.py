import torch.nn as nn
from mmdet.models import HEADS, build_loss
import torch
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
import copy
import sys
from projects.topoformer.utils.transformer import MultiHeadAttention, CounterfactualAttention, lclcRelationshipHead, topologicHead

@HEADS.register_module()
class BaselineHead(nn.Module):
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
                counterfactual_type='random',
                counterfactual_prob=0.5,
                use_topologic=False,
                distance_mapping_function=None,
                alpha=0.2,
                lambd=2.0,
                lamda_sim=1,
                lamda_geo=1,
                 ):
        super().__init__()
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
        self.use_counterfactual = use_counterfactual
        self._init_layers()
        self.rel_predictor = lclcRelationshipHead(
            in_channels_o1=hidden_dim,
            in_channels_o2=hidden_dim,
            shared_param=False)
        self.transformer_num_layers = transformer_num_layers
        self.loss_rel = build_loss(loss_rel)
        
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
        # query_feats = query_feats.permute(0, 2, 1)
        #之后再计算连接性
        # import pdb;pdb.set_trace()
        lclc_pred = self.rel_predictor(query_feats, query_feats)
        # import pdb;pdb.set_trace()
        lclc_pred = lclc_pred.squeeze(-1)
        lclc_pred = lclc_pred.unsqueeze(0).repeat(self.transformer_num_layers, 1, 1, 1)
        outs['all_lclc_preds'] = lclc_pred
        return outs
    def loss(self, lclc_preds,lclc_targets):
        lclc_preds = lclc_preds[-1]
        lclc_targets = lclc_targets[-1]
        # import pdb;pdb.set_trace()
        loss_lclc = self.loss_rel(lclc_preds, lclc_targets)
        return  loss_lclc
    def get_lclc(self, preds_dicts):

        all_lclc_preds = preds_dicts['all_lclc_preds'][-1].squeeze(-1).sigmoid().detach().cpu().numpy()
        all_lclc_preds = [_ for _ in all_lclc_preds]

        return all_lclc_preds
