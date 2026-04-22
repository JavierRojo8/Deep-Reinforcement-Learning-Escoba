

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

NUM_CARDS = 40  # cartas 1-40; 0 = padding

# Máximos conocidos para normalizar globales:
_GLOBALS_MAX = np.array(
    [4, 4, 40, 40, 10, 10, 1, 1, 20, 20, 30, 10, 100],
    dtype=np.float32,
)


_CARD_FEATURES = np.zeros((NUM_CARDS + 1, 3), dtype=np.float32)

for i in range(NUM_CARDS):
    palo = i // 10
    idx_relativo = i % 10
    es_oro = 1.0 if palo == 0 else 0.0
    numero_real = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12][idx_relativo]
    valor_juego = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][idx_relativo]
    es_siete = 1.0 if numero_real == 7 else 0.0
    

    _CARD_FEATURES[i + 1] = [valor_juego / 10.0, es_oro, es_siete]


class EscobaFeaturesExtractor(BaseFeaturesExtractor):
    

    def __init__(
        self,
        observation_space: spaces.Dict,
        embed_dim: int = 16,
        hidden_dim: int = 32,
        features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        
        self.card_embedding = nn.Embedding(NUM_CARDS + 1, embed_dim, padding_idx=0)

        
        self.register_buffer(
            "static_card_features",
            torch.tensor(_CARD_FEATURES, dtype=torch.float32),
        )

        
        self.card_mlp = nn.Sequential(
            nn.Linear(embed_dim + 3, hidden_dim),
            nn.ReLU(),
        )

        
        n_globals = observation_space["globales"].shape[0]
        globals_hidden = 16
        self.global_mlp = nn.Sequential(
            nn.Linear(n_globals, globals_hidden),
            nn.ReLU(),
        )

        self.register_buffer(
            "globals_max",
            torch.tensor(_GLOBALS_MAX[:n_globals], dtype=torch.float32),
        )

        
        self.played_mlp = nn.Sequential(
            nn.Linear(NUM_CARDS, 16),
            nn.ReLU(),
        )

        
        
        combined_dim = hidden_dim + hidden_dim + globals_hidden + 16
        self.output_mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    

    def freeze_encoder(self):
        for p in self.card_embedding.parameters():
            p.requires_grad = False
        for p in self.card_mlp.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.card_embedding.parameters():
            p.requires_grad = True
        for p in self.card_mlp.parameters():
            p.requires_grad = True

    

    def forward(self, observations: dict) -> torch.Tensor:
        hand_ids  = observations["hand"].long()    
        table_ids = observations["table"].long()   
        globals_f = observations["globales"].float()  

        
        hand_emb  = self.card_embedding(hand_ids)    # (B, 3,  embed_dim)
        table_emb = self.card_embedding(table_ids)   # (B, 10, embed_dim)

        
        hand_static  = self.static_card_features[hand_ids]   # (B, 3, 3)
        table_static = self.static_card_features[table_ids]  # (B, 10, 3)

        
        hand_combined  = torch.cat([hand_emb, hand_static], dim=-1)   # (B, 3, embed_dim + 3)
        table_combined = torch.cat([table_emb, table_static], dim=-1) # (B, 10, embed_dim + 3)

        
        hand_feat  = self.card_mlp(hand_combined)    # (B, 3,  hidden_dim)
        table_feat = self.card_mlp(table_combined)   # (B, 10, hidden_dim)

        
        hand_mask  = (hand_ids  > 0).float().unsqueeze(-1)   
        table_mask = (table_ids > 0).float().unsqueeze(-1)   

        hand_latent  = (hand_feat  * hand_mask ).sum(1) / hand_mask .sum(1).clamp(min=1.0)
        table_latent = (table_feat * table_mask).sum(1) / table_mask.sum(1).clamp(min=1.0)

        # Normalizar globales y pasar por MLP
        globals_norm   = globals_f / self.globals_max.clamp(min=1.0)
        globals_latent = self.global_mlp(globals_norm)

        
        played_f = observations["played_cards"].float()
        played_latent = self.played_mlp(played_f)

        
        combined = torch.cat([hand_latent, table_latent, globals_latent, played_latent], dim=-1)
        return self.output_mlp(combined)