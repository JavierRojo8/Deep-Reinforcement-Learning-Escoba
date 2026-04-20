"""
EscobaFeaturesExtractor — custom SB3 features extractor con representación latente aprendida + conocimiento experto inyectado.

Arquitectura:
  hand_ids  (B, 3)  ──┐
                       ├── Embedding(41, embed_dim) + [valor, es_oro, es_siete] ──► card_mlp ──► masked mean ──► hand_latent  (B, hidden_dim)
  table_ids (B, 10) ──┘                                                               (compartido)  masked mean ──► table_latent (B, hidden_dim)

  globales  (B, 13) ──► normalización ──► global_mlp ──► globals_latent (B, 16)

  cat([hand_latent, table_latent, globals_latent]) ──► output_mlp ──► features (B, features_dim)
"""

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

# ── NUEVO: Matriz estática de características de las cartas ─────────────────
# Índice 0 es padding [0, 0, 0]. Índices 1 a 40 son las cartas reales.
# Propiedades: [valor_juego (normalizado / 10), es_oros (0/1), es_siete (0/1)]
_CARD_FEATURES = np.zeros((NUM_CARDS + 1, 3), dtype=np.float32)

for i in range(NUM_CARDS):
    palo = i // 10
    idx_relativo = i % 10
    es_oro = 1.0 if palo == 0 else 0.0
    numero_real = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12][idx_relativo]
    valor_juego = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][idx_relativo]
    es_siete = 1.0 if numero_real == 7 else 0.0
    
    # Guardamos en i+1 porque el ID 0 es el padding de ranuras vacías
    _CARD_FEATURES[i + 1] = [valor_juego / 10.0, es_oro, es_siete]


class EscobaFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor para EscobaEnv.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        embed_dim: int = 16,
        hidden_dim: int = 32,
        features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        # ── Encoder de cartas (compartido entre mano y mesa) ────────────────
        self.card_embedding = nn.Embedding(NUM_CARDS + 1, embed_dim, padding_idx=0)

        # NUEVO: Registramos la info estática como un Buffer para que PyTorch 
        # la mueva automáticamente a la GPU (cuda/mps) junto con el modelo.
        self.register_buffer(
            "static_card_features",
            torch.tensor(_CARD_FEATURES, dtype=torch.float32),
        )

        # NUEVO: El MLP ahora recibe el embedding (embed_dim) + 3 características extra
        self.card_mlp = nn.Sequential(
            nn.Linear(embed_dim + 3, hidden_dim),
            nn.ReLU(),
        )

        # ── MLP para variables globales ─────────────────────────────────────
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

        # ── NUEVO: MLP para procesar la memoria de cartas jugadas ───────────
        self.played_mlp = nn.Sequential(
            nn.Linear(NUM_CARDS, 16),
            nn.ReLU(),
        )

        # ── MLP de salida (Actualizar el combined_dim) ──────────────────────
        # Sumamos los 16 del nuevo played_mlp
        combined_dim = hidden_dim + hidden_dim + globals_hidden + 16
        self.output_mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    # ── API de pre-entrenamiento ────────────────────────────────────────────

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

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, observations: dict) -> torch.Tensor:
        hand_ids  = observations["hand"].long()    
        table_ids = observations["table"].long()   
        globals_f = observations["globales"].float()  

        # 1. Obtenemos el embedding que el modelo APRENDE (Táctica)
        hand_emb  = self.card_embedding(hand_ids)    # (B, 3,  embed_dim)
        table_emb = self.card_embedding(table_ids)   # (B, 10, embed_dim)

        # 2. Obtenemos las reglas estáticas que le INYECTAMOS (Conocimiento puro)
        hand_static  = self.static_card_features[hand_ids]   # (B, 3, 3)
        table_static = self.static_card_features[table_ids]  # (B, 10, 3)

        # 3. Concatenamos ambas fuentes de información
        hand_combined  = torch.cat([hand_emb, hand_static], dim=-1)   # (B, 3, embed_dim + 3)
        table_combined = torch.cat([table_emb, table_static], dim=-1) # (B, 10, embed_dim + 3)

        # MLP por carta (ahora procesa la combinación)
        hand_feat  = self.card_mlp(hand_combined)    # (B, 3,  hidden_dim)
        table_feat = self.card_mlp(table_combined)   # (B, 10, hidden_dim)

        # Media enmascarada
        hand_mask  = (hand_ids  > 0).float().unsqueeze(-1)   
        table_mask = (table_ids > 0).float().unsqueeze(-1)   

        hand_latent  = (hand_feat  * hand_mask ).sum(1) / hand_mask .sum(1).clamp(min=1.0)
        table_latent = (table_feat * table_mask).sum(1) / table_mask.sum(1).clamp(min=1.0)

        # Normalizar globales y pasar por MLP
        globals_norm   = globals_f / self.globals_max.clamp(min=1.0)
        globals_latent = self.global_mlp(globals_norm)

        # --- NUEVO: Procesar la memoria de cartas ---
        played_f = observations["played_cards"].float()
        played_latent = self.played_mlp(played_f)

        # Concatenar todo (Mano + Mesa + Globales + Memoria)
        combined = torch.cat([hand_latent, table_latent, globals_latent, played_latent], dim=-1)
        return self.output_mlp(combined)