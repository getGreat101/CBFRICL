import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import DataSet
from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss


class GraphEncoder(nn.Module):
    def __init__(self, layers, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):

        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
            # x = self.dropout(x)
        return x

class info_nce_loss(nn.Module):
    def __init__(self):
        super(info_nce_loss, self).__init__()
    def forward(self, user_embedding, pos_item_embedding, neg_item_embedding, temperature=0.2):
        pos_similarity = F.cosine_similarity(user_embedding, pos_item_embedding)
        neg_similarity = F.cosine_similarity(user_embedding, neg_item_embedding)
        logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        logits = logits.to('cuda')
        labels = labels.to('cuda')
        loss = F.cross_entropy(logits / temperature, labels)

        return loss

class CBFRICL(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(CBFRICL, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.transformation_matrices = nn.ModuleDict({
            behavior: nn.Parameter(torch.randn(embedding_dim, embedding_dim)) for behavior in enumerate(self.behaviors)
        })
        self.Graph_encoder = nn.ModuleDict({
            behavior: GraphEncoder(self.layers[index], self.embedding_size, self.node_dropout) for index, behavior in enumerate(self.behaviors)
        })
        
        self.info_nce_loss = info_nce_loss()


        self.reg_weight = args.reg_weight
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_all_embeddings = None
   
        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.xavier_uniform_(self.transformation_matrices[behavior])


    def _load_model(self):
        if self.if_load_model:
            print(self.model_path)
            print(self.check_point)
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)


    def gcn_propagate(self):
        """
        gcn propagate in each behavior
        """
        all_embeddings = {}
        total_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        for behavior in self.behaviors:
            layer_embeddings = total_embeddings
            # layer_embeddings = torch.matmul(layer_embeddings, self.transformation_matrices[behavior])
            layer_embeddings = F.dropout(layer_embeddings, p=0.1, training=self.training)
            indices = self.edge_index[behavior].to(self.device)
            layer_embeddings = self.Graph_encoder[behavior](layer_embeddings, indices)
            layer_embeddings = F.normalize(layer_embeddings, dim=-1)
            total_embeddings = layer_embeddings + total_embeddings
            all_embeddings[behavior] = total_embeddings
        return all_embeddings

    def forward(self, batch_data):
        self.storage_all_embeddings = None

        all_embeddings = self.gcn_propagate()
        total_loss = 0
        
        for index, behavior in enumerate(self.behaviors):
            data = batch_data[:, index]
            users = data[:, 0].long()
            items = data[:, 1:].long()
            user_all_embedding, item_all_embedding = torch.split(all_embeddings[behavior], [self.n_users + 1, self.n_items + 1])

            user_feature = user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)
            item_feature = item_all_embedding[items]
            pos_item_embedding, neg_item_embedding = torch.unbind(item_feature, dim=1)
            cl_user_embedding = user_all_embedding[users.view(-1, 1)].squeeze(1)
            cl_loss = self.info_nce_loss(cl_user_embedding, pos_item_embedding, neg_item_embedding, temperature=0.1)
            scores = torch.sum(user_feature * item_feature, dim=2)
            total_loss += self.bpr_loss(scores[:, 0], scores[:, 1]) * weights[behavior]
            total_loss += cl_loss
        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        return total_loss

    def full_predict(self, users):
        if self.storage_all_embeddings is None:
            self.storage_all_embeddings = self.gcn_propagate()

        user_embedding, item_embedding = torch.split(self.storage_all_embeddings[self.behaviors[-1]], [self.n_users + 1, self.n_items + 1])
        user_emb = user_embedding[users.long()]
        scores = torch.matmul(user_emb, item_embedding.transpose(0, 1))
        return scores

