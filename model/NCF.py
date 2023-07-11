from typing import List

import torch
import torch.nn.functional as F
from torch import nn
# import torchvision.mode

class NCF_Hyper(nn.Module):
    """LeNet Hypernetwork
    """

    def __init__(self,
        preference_dim=2,
        preference_embedding_dim=32,
        hidden_dim=100,
        num_chunks=300,
        chunk_embedding_dim=64,
        num_ws=11,
        w_dim=10000, drop_out=0.15,
        num_users = 84580,
        num_items = 25896,
        mf_dim = 32,
        mlp_dim = 32,

        ):
        super().__init__()
        self.drop_out = drop_out

        self.preference_embedding_dim = preference_embedding_dim
        self.num_chunks = num_chunks

        self.chunk_embedding_matrix = nn.Embedding(
            num_embeddings=num_chunks, embedding_dim=chunk_embedding_dim
        )
        self.preference_embedding_matrix = nn.Embedding(
            num_embeddings=preference_dim, embedding_dim=preference_embedding_dim
        )

        self.fc = nn.Sequential(
            nn.Linear(preference_embedding_dim + chunk_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(self.drop_out)
        )

        list_ws = [self._init_w((w_dim, hidden_dim)) for _ in range(num_ws)]
        self.ws = nn.ParameterList(list_ws)

        # initialization
        torch.nn.init.normal_(
            self.preference_embedding_matrix.weight, mean=0.0, std=0.1
        )
        torch.nn.init.normal_(self.chunk_embedding_matrix.weight, mean=0.0, std=0.1)
        for w in self.ws:
            torch.nn.init.normal_(w, mean=0.0, std=0.1)

        self.layer_to_shape = {
            "emb_MF_users": torch.Size(
                [num_users, mf_dim]
            ),
            "emb_MF_items" : torch.Size(
                [num_items, mf_dim]
            ),
            "emb_MLP_users": torch.Size(
                [num_users, mlp_dim]
            ),
            "emb_MLP_items": torch.Size(
                [num_items, mlp_dim]
            ),
            "mlp1.weights": torch.Size(
                [32, 64]
            ),
            "mlp1.bias": torch.Size(
                [32]
            ),
            "mlp2.weigts": torch.Size(
                [16,32]
            ),
            "mlp2.bias": torch.Size(
                [16]
            ),
            "mlp3.weights":torch.Size(
                [8,16]
            ),
            "mlp3.bias":torch.Size(
                [8]  
            ),
            "predict.weigts": torch.Size(
                [1,40]
            ),
            "predict.bias": torch.Size(
                [1]
            )
        }

    def _init_w(self, shapes):
        return nn.Parameter(torch.randn(shapes), requires_grad=True)

    def forward(self, preference):
        pref_embedding = torch.zeros(
            (self.preference_embedding_dim,), device=preference.device
        )

        for i, pref in enumerate(preference):
            pref_embedding += (
                self.preference_embedding_matrix(
                    torch.tensor([i], device=preference.device)
                ).squeeze(0)
                * pref
            )

        weights = []

        for chunk_id in range(self.num_chunks):
            chunk_embedding = self.chunk_embedding_matrix(
                torch.tensor([chunk_id], device=preference.device)
            ).squeeze(0)
            # input to fc
            input_embedding = torch.cat((pref_embedding, chunk_embedding)).unsqueeze(0)
            # hidden representation
            rep = self.fc(input_embedding)

            weights.append(torch.cat([F.linear(rep, weight=w) for w in self.ws], dim=1))


        weight_vector = torch.cat(weights, dim=1).squeeze(0)
        # print(len(weight_vector))
        # print("="*25)
        out_dict = dict()
        position = 0
        for name, shapes in self.layer_to_shape.items():
            out_dict[name] = weight_vector[
                position : position + shapes.numel()
            ].reshape(shapes)
            position += shapes.numel()
            # print(name, shapes, out_dict[name].shape, position)
        
        return out_dict

class NCF_Target(nn.Module):
    def __init__(self):
        super(NCF_Target, self).__init__()
        self.relu = nn.ReLU()
    
    def forward(self, user, item , weights = None):
        MF_Embedding_User = nn.Embedding.from_pretrained(weights["emb_MF_users"])
        MF_Embedding_Item = nn.Embedding.from_pretrained(weights["emb_MF_items"])
        MLP_Embedding_User = nn.Embedding.from_pretrained(weights["emb_MLP_users"])
        MLP_Embedding_Item = nn.Embedding.from_pretrained(weights["emb_MLP_items"])

        #MF part
        mf_user_latent = MF_Embedding_User(user)
        mf_item_latent = MF_Embedding_Item(item)
        mf_vector = torch.mul(mf_user_latent,mf_item_latent)  # dim=32

        # MLP part
        mlp_user_latent= MLP_Embedding_User(user)
        mlp_item_latent= MLP_Embedding_Item(item)

        mlp_vector = torch.cat((mlp_user_latent,mlp_item_latent), dim=1)

        mlp_vector = F.Linear(mlp_vector, weight= weights["mlp1.weights"].reshape(32,64), bias = weights["mlp1.bias"])
        mlp_vector = relu(mlp_vector)
        mlp_vector = F.Linear(mlp_vector, weight= weights["mlp2.weights"].reshape(16,32), bias = weights["mlp2.bias"])
        mlp_vector = relu(mlp_vector)
        mlp_vector = F.Linear(mlp_vector, weight= weights["mlp3.weights"].reshape(8,16), bias = weights["mlp3.bias"])
        mlp_vector = relu(mlp_vector)
        #dim=8

        predict_vector = torch.cat((mf_vector,mlp_vector), dim=1)
        predict_vector = F.Linear(predict_vector, weight = weights["predict.weigts"].reshape(1,40), bias=weights["predict.bias"])
        predict_vector = torch.sigmoid(predict_vector)

        return predict_vector
