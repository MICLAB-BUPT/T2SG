import torch.nn as nn
from mmdet.models import HEADS, build_loss
import torch
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
import copy
@HEADS.register_module()
class CustomDeformableDetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.
    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    """

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 transformer_num_layers=6,
                 num_layers=None,
                 loss_rel=None,
                 num_query=200
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
        self._init_layers()
    def _init_layers(self):
        proj_q =[]
        proj_q.append(nn.Linear(self.embed_dims, self.embed_dims))
        proj_q = nn.Sequential(*proj_q)
        proj_k =[]
        proj_k.append(nn.Linear(self.embed_dims, self.embed_dims))
        proj_k = nn.Sequential(*proj_k)
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        self.proj_q = _get_clones(proj_q, self.num_pred)
        self.proj_k = _get_clones(proj_k, self.num_pred)
        self.final_sub_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.final_obj_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.rel_predictor_gate = nn.Linear(2 * self.embed_dims, 1)
        # self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        self.rel_predictor = MLPHEAD(
            input_dim=2 * self.embed_dims,
            hidden_dim=self.embed_dims,
            output_dim=1,
            num_layers=self.num_layers,
        )
    def forward(self, outs):
        """
        输入的是outs,
        outs = {
            'all_cls_scores': outputs_classes,
            'all_lanes_preds': outputs_coords,
            # 'all_lclc_preds': pred_rel,
            'all_lcte_preds': lcte_rel_out,
            'history_states': hs
        } 
        """
        hs = outs['history_states']
        output_lane_query = []
        output_lane_key = []
        for lvl in range(hs.shape[0]):
            lane_query = self.proj_q[lvl](hs[lvl])#(bs, num_query, embed_dims)
            output_lane_query.append(lane_query)
            lane_key = self.proj_k[lvl](hs[lvl])#(bs, num_query, embed_dims)
            output_lane_key.append(lane_key)
        #Stacking attention keys and queries
        
        decoder_attention_queries = torch.stack(output_lane_query, dim=-2)#(bs, num_query, num_dec_layers, embed_dims)
        decoder_attention_keys = torch.stack(output_lane_key, dim=-2)#(bs, num_query, num_dec_layers, embed_dims)
        del output_lane_query, output_lane_key
        num_object_queries = decoder_attention_queries.size(1)
        #Pairwise concatenation
        decoder_attention_queries = decoder_attention_queries.unsqueeze(2).repeat(
            1, 1, num_object_queries, 1, 1
        )
        #[bsz, num_object_queries, num_object_queries, num_layers, d_model]
        decoder_attention_keys = decoder_attention_keys.unsqueeze(1).repeat(
            1, num_object_queries, 1, 1, 1
        )
        #[bsz, num_object_queries, num_object_queries, num_layers, d_model]
        relation_source = torch.cat(
            [decoder_attention_queries, decoder_attention_keys], dim=-1
        )  # [bsz, num_object_queries, num_object_queries, num_layers, 2*d_model]
        del decoder_attention_queries, decoder_attention_keys
        sequence_output = hs[-1]
        # [bsz, num_object_queries, d_model]
        subject_output = (
            self.final_sub_proj(sequence_output)
            .unsqueeze(2)
            .repeat(1, 1, num_object_queries, 1)
        )
        object_output = (
            self.final_obj_proj(sequence_output)
            .unsqueeze(1)
            .repeat(1, num_object_queries, 1, 1)
        )
        del sequence_output
        relation_source = torch.cat(
            [
                relation_source,
                torch.cat([subject_output, object_output], dim=-1).unsqueeze(-2),
            ],
            dim=-2,
        )
        #(batch_size, sequence_length, num_object_queries ,num_layer + 1, hidden_size)
        del subject_output, object_output
        rel_gate = torch.sigmoid(self.rel_predictor_gate(relation_source))
        #(batch_size, sequence_length, num_object_queries ,num_layer + 1, 1)
        gated_relation_source = torch.mul(rel_gate, relation_source).sum(dim=-2)
        #(batch_size, sequence_length, num_object_queries ,hidden_size)
        # if self.use_sigmod:
        #     pred_rel = torch.sigmoid(self.rel_predictor(gated_relation_source))
        pred_rel = self.rel_predictor(gated_relation_source)
        #(batch_size, sequence_length, num_object_queries ,1)
        #torch.Size([1, 200, 200, 1])
        pred_rel = pred_rel.unsqueeze(0).repeat(hs.shape[0], 1, 1, 1, 1)
        # pred_rel = pred_rel.sigmoid()
        pred_rel = pred_rel.squeeze(-1)
        outs['all_lclc_preds'] = pred_rel
        return outs
    def loss(self, pred, target):
        # import pdb;pdb.set_trace()
        pred = pred[-1]
        target = target[-1]
        return self.loss_cls(pred, target)
    
    
class MLPHEAD(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers=None
                 ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
# class DeformableDetrMLPPredictionHead(nn.Module):
#     """
#     Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
#     height and width of a bounding box w.r.t. an image.
#     Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py
#     """

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x