"""
EscobaFeaturesExtractor — custom SB3 features extractor con representación latente aprendida.

Arquitectura:
  hand_ids  (B, 3)  ──┐
                       ├── Embedding(41, embed_dim) ──► card_mlp ──► masked mean ──► hand_latent  (B, hidden_dim)
  table_ids (B, 10) ──┘                              (compartido)  masked mean ──► table_latent (B, hidden_dim)

  globales  (B, 13) ──► normalización ──► global_mlp ──► globals_latent (B, 16)

  cat([hand_latent, table_latent, globals_latent]) ──► output_mlp ──► features (B, features_dim)

Por qué es representation learning defendible:
  - El embedding de cartas se aprende end-to-end a partir de señales de juego.
  - El modelo debe descubrir qué cartas son similares (valor, palo, importancia táctica)
    sin que eso esté codificado a mano; la única señal es la recompensa.
  - La agregación por media enmascarada es invariante a permutaciones (tratar la mano
    como un conjunto, no una secuencia ordenada).
  - Las variables globales se normalizan antes de pasar al MLP para estabilidad numérica.

Para pre-entrenamiento futuro del encoder:
  - Acceder con model.policy.features_extractor
  - Llamar .freeze_encoder() / .unfreeze_encoder() para congelar/descongelar
    el embedding y card_mlp independientemente del resto de la política.
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

NUM_CARDS = 40  # cartas 1-40; 0 = padding


# Máximos conocidos para normalizar globales (orden de _get_obs en escoba_gym.py):
# [sietes_op, sietes_tu, cartas_op, cartas_tu, oros_op, oros_tu,
#  tiene_7oro_op, tiene_7oro_tu, escobas_op, escobas_tu, mazo, n_mesa, suma_mesa]
_GLOBALS_MAX = np.array(
    [4, 4, 40, 40, 10, 10, 1, 1, 20, 20, 30, 10, 100],
    dtype=np.float32,
)


class EscobaFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor para EscobaEnv.

    Parámetros:
        observation_space : spaces.Dict del entorno.
        embed_dim         : dimensión del embedding de carta (defecto: 16).
        hidden_dim        : dimensión de la MLP por carta (defecto: 32).
        features_dim      : dimensión del vector de salida (defecto: 64).
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
        # 41 tokens: 0 = padding, 1..40 = carta real
        self.card_embedding = nn.Embedding(NUM_CARDS + 1, embed_dim, padding_idx=0)

        # MLP por carta (aplicado elemento a elemento; pesos compartidos)
        self.card_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
        )

        # ── MLP para variables globales ─────────────────────────────────────
        n_globals = observation_space["globales"].shape[0]
        globals_hidden = 16
        self.global_mlp = nn.Sequential(
            nn.Linear(n_globals, globals_hidden),
            nn.ReLU(),
        )

        # Tensor de normalización (registrado como buffer → se mueve con .to(device))
        self.register_buffer(
            "globals_max",
            torch.tensor(_GLOBALS_MAX[:n_globals], dtype=torch.float32),
        )

        # ── MLP de salida ───────────────────────────────────────────────────
        combined_dim = hidden_dim + hidden_dim + globals_hidden
        self.output_mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    # ── API de pre-entrenamiento ────────────────────────────────────────────

    def freeze_encoder(self):
        """Congela el embedding y el card_mlp para fine-tuning."""
        for p in self.card_embedding.parameters():
            p.requires_grad = False
        for p in self.card_mlp.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        """Descongela el encoder (vuelta a end-to-end)."""
        for p in self.card_embedding.parameters():
            p.requires_grad = True
        for p in self.card_mlp.parameters():
            p.requires_grad = True

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, observations: dict) -> torch.Tensor:
        # SB3 convierte los Box a float32 internamente; recuperamos enteros para Embedding
        hand_ids  = observations["hand"].long()    # (B, HAND_SIZE)
        table_ids = observations["table"].long()   # (B, MAX_TABLE_CARDS)
        globals_f = observations["globales"].float()  # (B, n_globals)

        # Embedding de cartas
        hand_emb  = self.card_embedding(hand_ids)    # (B, 3,  embed_dim)
        table_emb = self.card_embedding(table_ids)   # (B, 10, embed_dim)

        # MLP por carta
        hand_feat  = self.card_mlp(hand_emb)    # (B, 3,  hidden_dim)
        table_feat = self.card_mlp(table_emb)   # (B, 10, hidden_dim)

        # Media enmascarada (las posiciones rellenas con ID=0 no contribuyen)
        hand_mask  = (hand_ids  > 0).float().unsqueeze(-1)   # (B, 3,  1)
        table_mask = (table_ids > 0).float().unsqueeze(-1)   # (B, 10, 1)

        hand_latent  = (hand_feat  * hand_mask ).sum(1) / hand_mask .sum(1).clamp(min=1.0)
        table_latent = (table_feat * table_mask).sum(1) / table_mask.sum(1).clamp(min=1.0)

        # Normalizar globales y pasar por MLP
        globals_norm   = globals_f / self.globals_max.clamp(min=1.0)
        globals_latent = self.global_mlp(globals_norm)

        # Concatenar y proyectar
        combined = torch.cat([hand_latent, table_latent, globals_latent], dim=-1)
        return self.output_mlp(combined)
