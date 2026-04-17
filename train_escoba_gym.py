import os
import glob
import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend no-interactivo → guarda a fichero
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from escoba_gym import EscobaEnv
from card_encoder import EscobaFeaturesExtractor

# ── Configuración centralizada ──────────────────────────────────────────────
CONFIG = {
    "total_timesteps":  200_000_000,
    "opponent_type":    "greedy",          # "random" | "greedy" | "model" (se sobreescribe con self_play)
    "policy":           "MultiInputPolicy",
    "learning_rate":    3e-4,
    "n_steps":          256,
    "batch_size":       256,
    "n_epochs":         10,
    "ent_coef":         0.01,
    "gamma":            0.99,
    "gae_lambda":       0.95,
    "clip_range":       0.2,
    "eval_freq":        5_000,
    "n_eval_episodes":  20,
    # ── Encoder de cartas ────────────────────────────────────────────────────
    "embed_dim":        16,
    "hidden_dim":       32,
    "features_dim":     64,
    # ── Self-play (desactivado por defecto) ──────────────────────────────────
    #
    # Modos de uso:
    #   A) Un único boss fijo:
    #        self_play=True, boss_model_path="models/PPO/PPO_1/best_model"
    #   B) Carpeta con hasta 3 bosses rotando:
    #        self_play=True, boss_models_dir="bosses/", boss_rotation_mode="cyclic"
    #   C) Self-checkpoints congelados del propio agente:
    #        self_play=True, use_frozen_self_checkpoints=True
    #   D) Combinar B + C:
    #        self_play=True, boss_models_dir="bosses/",
    #        use_frozen_self_checkpoints=True
    #
    # Notas:
    #   - boss_model_path y boss_models_dir son mutuamente excluyentes.
    #   - change_boss_steps debe ser > 0.
    #   - Si boss_models_dir tiene más de 3 modelos, se usan los 3 más recientes.
    "self_play":                    False,
    "boss_model_path":              None,   # str o None
    "boss_models_dir":              None,   # str o None
    "change_boss_steps":            50_000,
    "boss_rotation_mode":           "cyclic",   # "cyclic" | "random" | "weighted"
    "boss_deterministic":           True,
    "use_frozen_self_checkpoints":  False,
    "self_checkpoint_dir":          "checkpoints",
    "self_checkpoint_freq":         100_000,
}

LOG_DIR    = "logs"
MODELS_DIR = "models/PPO"
_MAX_EXTERNAL_BOSSES = 3
_MAX_SELF_CHECKPOINTS = 5   # cuántos snapshots propios se mantienen en el pool a la vez


# ══════════════════════════════════════════════════════════════════════════════
#  BOSS MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class BossManager:
    """
    Gestiona el pool de modelos boss para self-play.

    Fuentes de bosses:
      • Externos  — rutas a modelos .zip cargados una sola vez (lazy).
      • Self-checkpoints — snapshots congelados del propio agente en entrenamiento.

    Modos de rotación:
      cyclic   — round-robin determinista.
      random   — uniforme al azar (sin repetir boss si hay más de uno).
      weighted — prioriza bosses contra los que el agente gana menos:
                 weight(boss) = max(1 − win_rate_reciente, 0) + 0.1
    """

    def __init__(
        self,
        rotation_mode: str = "cyclic",
        deterministic: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        assert rotation_mode in ("cyclic", "random", "weighted"), (
            f"boss_rotation_mode debe ser 'cyclic', 'random' o 'weighted', "
            f"no '{rotation_mode}'"
        )
        self.rotation_mode = rotation_mode
        self.deterministic = deterministic
        self._log = logger or logging.getLogger("BossManager")

        # Cada entrada: {"id": str, "path": str, "model": PPO|None, "is_self": bool}
        self._entries: list = []
        self._idx: int = 0
        self._rng = np.random.default_rng()
        # boss_id → lista de 1(win)/0(no-win) para weighted
        self._results: dict = {}

    # ── Añadir bosses ─────────────────────────────────────────────────────────

    def add_external(self, path: str, boss_id: Optional[str] = None) -> bool:
        real = path if os.path.exists(path) else (path + ".zip")
        if not os.path.exists(real):
            self._log.warning(f"[BossManager] Boss ignorado (no encontrado): {path}")
            return False
        bid = boss_id or Path(path).stem
        self._entries.append({"id": bid, "path": path, "model": None, "is_self": False})
        self._results[bid] = []
        self._log.info(f"[BossManager] Boss externo añadido: {bid}")
        return True

    def add_self_checkpoint(self, path: str) -> None:
        """
        Añade un snapshot congelado del propio agente al pool.
        Si ya hay _MAX_SELF_CHECKPOINTS snapshots propios, elimina el más antiguo
        (libera RAM y mantiene el pool acotado).
        Si el boss eliminado era el activo, avanza al siguiente antes de borrarlo.
        """
        # Expulsar el self-checkpoint más antiguo si se supera el límite
        self_entries = [e for e in self._entries if e["is_self"]]
        if len(self_entries) >= _MAX_SELF_CHECKPOINTS:
            oldest = self_entries[0]
            oldest_idx = next(i for i, e in enumerate(self._entries) if e["id"] == oldest["id"])
            if self._idx == oldest_idx:
                # El activo es el que vamos a borrar: avanzar primero
                self._idx = (self._idx + 1) % len(self._entries)
            self._entries.pop(oldest_idx)
            self._results.pop(oldest["id"], None)
            # Ajustar índice si el borrado era antes del activo
            if oldest_idx < self._idx:
                self._idx -= 1
            self._log.info(f"[BossManager] Self-checkpoint antiguo expulsado: {oldest['id']}")

        bid = Path(path).stem
        model = PPO.load(path)
        for p in model.policy.parameters():
            p.requires_grad = False
        self._entries.append({"id": bid, "path": path, "model": model, "is_self": True})
        self._results[bid] = []
        self._log.info(f"[BossManager] Self-checkpoint añadido: {bid}  (pool={len(self._entries)})")

    def has_bosses(self) -> bool:
        return len(self._entries) > 0

    # ── Acceso al boss actual ─────────────────────────────────────────────────

    @property
    def boss_id(self) -> Optional[str]:
        return self._entries[self._idx]["id"] if self._entries else None

    @property
    def current_model(self) -> Optional[object]:
        if not self._entries:
            return None
        entry = self._entries[self._idx]
        if entry["model"] is None:              # carga lazy para externos
            entry["model"] = PPO.load(entry["path"])
            self._log.info(f"[BossManager] Modelo cargado en memoria: {entry['id']}")
        return entry["model"]

    # ── Rotación ──────────────────────────────────────────────────────────────

    def rotate(self) -> None:
        n = len(self._entries)
        if n <= 1:
            return
        old = self.boss_id
        if self.rotation_mode == "cyclic":
            self._idx = (self._idx + 1) % n
        elif self.rotation_mode == "random":
            choices = [i for i in range(n) if i != self._idx]
            self._idx = int(self._rng.choice(choices))
        else:  # weighted
            self._idx = self._weighted_next()
        self._log.info(f"[BossManager] Rotación: {old} → {self.boss_id}")

    def _weighted_next(self) -> int:
        weights = []
        for entry in self._entries:
            r = self._results.get(entry["id"], [])
            wr = float(np.mean(r[-20:])) if len(r) >= 5 else 0.5
            weights.append(max(1.0 - wr, 0.0) + 0.1)
        total = sum(weights)
        probs = [w / total for w in weights]
        return int(self._rng.choice(len(self._entries), p=probs))

    # ── Estadísticas ──────────────────────────────────────────────────────────

    def record_result(self, boss_id: str, result: str) -> None:
        if boss_id in self._results:
            self._results[boss_id].append(1 if result == "win" else 0)

    def recent_win_rate(self, boss_id: str, n: int = 50) -> Optional[float]:
        r = self._results.get(boss_id, [])
        return float(np.mean(r[-n:])) if r else None

    def summary(self) -> dict:
        out = {}
        for e in self._entries:
            bid = e["id"]
            r = self._results.get(bid, [])
            out[bid] = {
                "episodes": len(r),
                "win_rate": float(np.mean(r)) if r else None,
                "is_self":  e["is_self"],
            }
        return out


# ── Helper para construir un BossManager desde CONFIG ────────────────────────

def _build_boss_manager(cfg: dict, logger: logging.Logger) -> Optional[BossManager]:
    """
    Valida la configuración de self-play y construye el BossManager.
    Devuelve None si self_play=False.
    """
    if not cfg.get("self_play", False):
        return None

    if cfg["boss_model_path"] and cfg["boss_models_dir"]:
        raise ValueError(
            "boss_model_path y boss_models_dir son mutuamente excluyentes."
        )
    if cfg.get("change_boss_steps", 0) <= 0:
        raise ValueError("change_boss_steps debe ser > 0.")

    bm = BossManager(
        rotation_mode=cfg.get("boss_rotation_mode", "cyclic"),
        deterministic=cfg.get("boss_deterministic", True),
        logger=logger,
    )

    # Modo A: un único boss fijo
    if cfg["boss_model_path"]:
        bm.add_external(cfg["boss_model_path"])

    # Modo B: carpeta con varios bosses
    if cfg["boss_models_dir"]:
        zips = sorted(glob.glob(os.path.join(cfg["boss_models_dir"], "*.zip")))
        paths = [p[:-4] for p in zips]   # quitar .zip para PPO.load
        if len(paths) > _MAX_EXTERNAL_BOSSES:
            logger.warning(
                f"[BossManager] {len(paths)} modelos en boss_models_dir; "
                f"se usan los {_MAX_EXTERNAL_BOSSES} más recientes."
            )
            paths = paths[-_MAX_EXTERNAL_BOSSES:]
        for p in paths:
            bm.add_external(p)

    return bm


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

class EscobaStatsCallback(BaseCallback):
    """
    Acumula estadísticas de juego por rollout y las vuelca a TensorBoard.

    Métricas globales (prefijo 'escoba/'):
      win_rate, draw_rate, loss_rate, point_diff, points_tu, points_op,
      escobas_tu, escobas_op, cards_rate, 7oro_rate

    Métricas por boss (prefijo 'escoba_boss_<id>/'):
      win_rate, episodes

    También mantiene self.history para generar gráficas al final.
    """

    # Máximo de episodios que se guardan en memoria para las gráficas finales.
    # Con ~18 pasos/episodio y 200M steps hay ~11M episodios; sin este cap la
    # lista crece indefinidamente y provoca OOM. 200k episodios ≈ 50 MB.
    _HISTORY_MAXLEN = 200_000

    def __init__(self, boss_manager: Optional[BossManager] = None, verbose: int = 0):
        super().__init__(verbose)
        self.boss_manager = boss_manager
        self._reset_rollout_buffers()
        self.history: deque = deque(maxlen=self._HISTORY_MAXLEN)

    def _reset_rollout_buffers(self):
        self._wins     = []
        self._draws    = []
        self._losses   = []
        self._pt_tu    = []
        self._pt_op    = []
        self._esc_tu   = []
        self._esc_op   = []
        self._cards_tu = []
        self._7oro     = []
        self._boss_buf: dict = {}   # boss_id → {"wins": [], "episodes": 0}

    def _on_step(self) -> bool:
        boss_id = self.boss_manager.boss_id if self.boss_manager else "none"

        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done or "result" not in info:
                continue
            result = info["result"]
            win    = int(result == "win")

            self._wins.append(win)
            self._draws.append(int(result == "draw"))
            self._losses.append(int(result == "loss"))
            self._pt_tu.append(info.get("puntos_tu", 0))
            self._pt_op.append(info.get("puntos_op", 0))
            self._esc_tu.append(info.get("escobas_tu", 0))
            self._esc_op.append(info.get("escobas_op", 0))
            self._cards_tu.append(info.get("cartas_tu", 0))
            self._7oro.append(info.get("tiene_7oro_tu", 0))

            # Per-boss buffer
            if boss_id not in self._boss_buf:
                self._boss_buf[boss_id] = {"wins": [], "episodes": 0}
            self._boss_buf[boss_id]["wins"].append(win)
            self._boss_buf[boss_id]["episodes"] += 1

            # Notificar al BossManager para weighted rotation
            if self.boss_manager:
                self.boss_manager.record_result(boss_id, result)

            # Historia global para gráficas
            self.history.append({
                "step":      self.num_timesteps,
                "boss_id":   boss_id,
                "result":    result,
                "puntos_tu": info.get("puntos_tu", 0),
                "puntos_op": info.get("puntos_op", 0),
            })
        return True

    def _on_rollout_end(self):
        if not self._wins:
            return
        pt_tu = np.array(self._pt_tu)
        pt_op = np.array(self._pt_op)

        # Métricas globales
        self.logger.record("escoba/win_rate",   np.mean(self._wins))
        self.logger.record("escoba/draw_rate",  np.mean(self._draws))
        self.logger.record("escoba/loss_rate",  np.mean(self._losses))
        self.logger.record("escoba/point_diff", np.mean(pt_tu - pt_op))
        self.logger.record("escoba/points_tu",  np.mean(pt_tu))
        self.logger.record("escoba/points_op",  np.mean(pt_op))
        self.logger.record("escoba/escobas_tu", np.mean(self._esc_tu))
        self.logger.record("escoba/escobas_op", np.mean(self._esc_op))
        self.logger.record("escoba/cards_rate", np.mean(self._cards_tu) / 40.0)
        self.logger.record("escoba/7oro_rate",  np.mean(self._7oro))

        # Métricas por boss
        for bid, buf in self._boss_buf.items():
            if buf["wins"]:
                prefix = f"escoba_boss_{bid}"
                self.logger.record(f"{prefix}/win_rate", np.mean(buf["wins"]))
                self.logger.record(f"{prefix}/episodes", buf["episodes"])

        self._reset_rollout_buffers()


class SelfCheckpointCallback(BaseCallback):
    """
    Guarda snapshots congelados del agente en entrenamiento y los añade
    al pool de bosses del BossManager.
    """

    def __init__(
        self,
        boss_manager: BossManager,
        checkpoint_dir: str,
        freq: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.boss_manager   = boss_manager
        self.checkpoint_dir = checkpoint_dir
        self.freq           = freq
        self._last_save     = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.freq:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.checkpoint_dir, f"self_{self.num_timesteps}")
            self.model.save(path)
            self.boss_manager.add_self_checkpoint(path)
            self._last_save = self.num_timesteps
        return True


class BossRotationCallback(BaseCallback):
    """
    Rota el boss activo cada `change_steps` pasos y actualiza el modelo
    oponente en el entorno de entrenamiento.
    """

    def __init__(
        self,
        boss_manager: BossManager,
        escoba_env: EscobaEnv,
        change_steps: int,
        stats_cb: Optional[EscobaStatsCallback] = None,
        logger: Optional[logging.Logger] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.boss_manager = boss_manager
        self.escoba_env   = escoba_env
        self.change_steps = change_steps
        self.stats_cb     = stats_cb
        self._log         = logger or logging.getLogger("BossRotation")
        self._last_change = 0

    def _on_step(self) -> bool:
        if not self.boss_manager.has_bosses():
            return True   # pool vacío todavía (esperando primer self-checkpoint)

        if self.num_timesteps - self._last_change >= self.change_steps:
            # Log métricas recientes antes de rotar
            bid = self.boss_manager.boss_id
            wr  = self.boss_manager.recent_win_rate(bid)
            wr_str = f"{wr:.2%}" if wr is not None else "N/A"
            self._log.info(
                f"[BossRotation] step={self.num_timesteps:,}  "
                f"boss={bid}  win_rate_reciente={wr_str}"
            )

            self.boss_manager.rotate()

            new_model = self.boss_manager.current_model
            self.escoba_env.opponent_model = new_model
            self._last_change = self.num_timesteps
            self._log.info(
                f"[BossRotation] Nuevo boss activo: {self.boss_manager.boss_id}"
            )

        return True


# ══════════════════════════════════════════════════════════════════════════════
#  GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(
    stats_cb: EscobaStatsCallback,
    boss_manager: Optional[BossManager],
    save_dir: str,
) -> None:
    """
    Genera y guarda:
      • training_global.png  — win/draw/loss rate con rolling mean, para todos los episodios.
      • training_boss_<id>.png — idem filtrando por boss.
    """
    if not stats_cb.history:
        return

    os.makedirs(save_dir, exist_ok=True)
    history = stats_cb.history

    boss_ids_in_history = sorted({h["boss_id"] for h in history})
    groups = {"_global_": history}
    for bid in boss_ids_in_history:
        groups[bid] = [h for h in history if h["boss_id"] == bid]

    for group_id, episodes in groups.items():
        if not episodes:
            continue
        steps   = [e["step"]    for e in episodes]
        wins    = [int(e["result"] == "win")  for e in episodes]
        draws   = [int(e["result"] == "draw") for e in episodes]
        losses  = [int(e["result"] == "loss") for e in episodes]
        pt_diff = [e["puntos_tu"] - e["puntos_op"] for e in episodes]

        w = 30  # rolling window
        def roll(arr):
            arr = np.array(arr, dtype=float)
            out = np.full_like(arr, np.nan)
            for i in range(len(arr)):
                out[i] = arr[max(0, i - w + 1): i + 1].mean()
            return out

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax = axes[0]
        ax.plot(steps, roll(wins),   label="Win rate",  color="green")
        ax.plot(steps, roll(draws),  label="Draw rate", color="gray",  linestyle="--")
        ax.plot(steps, roll(losses), label="Loss rate", color="red",   linestyle=":")
        ax.set_ylabel("Rate (MA30)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        title = "Global" if group_id == "_global_" else f"Boss: {group_id}"
        ax.set_title(f"Escoba self-play — {title}  ({len(episodes)} episodios)")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(steps, roll(pt_diff), label="Punto diff (MA30)", color="steelblue")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("puntos_tu − puntos_op")
        ax.set_xlabel("Timestep")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = "training_global.png" if group_id == "_global_" else f"training_boss_{group_id}.png"
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=120)
        plt.close(fig)

    # Resumen de bosses si existe
    if boss_manager:
        summary = boss_manager.summary()
        summary_path = os.path.join(save_dir, "boss_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _next_ppo_run_name() -> str:
    existing = glob.glob(os.path.join(LOG_DIR, "PPO_*"))
    nums = []
    for d in existing:
        try:
            nums.append(int(os.path.basename(d).split("_")[1]))
        except (IndexError, ValueError):
            pass
    return f"PPO_{max(nums, default=0) + 1}"


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def entrenar(cfg: dict = None):
    cfg = cfg or CONFIG

    os.makedirs(LOG_DIR,    exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    run_name  = _next_ppo_run_name()
    model_dir = os.path.join(MODELS_DIR, run_name)
    os.makedirs(model_dir, exist_ok=True)

    # ── Logger ──────────────────────────────────────────────────────────────
    logger = logging.getLogger("escoba_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(model_dir, "train.log"))
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # ── Configuración self-play ──────────────────────────────────────────────
    boss_manager = _build_boss_manager(cfg, logger)
    use_self_play = boss_manager is not None or cfg.get("use_frozen_self_checkpoints", False)

    if use_self_play:
        opp_type = "model"
        logger.info(f"[Self-play] activado  |  rotation={cfg['boss_rotation_mode']}")
    else:
        opp_type = cfg["opponent_type"]

    logger.info(
        f"Run: {run_name}  |  oponente: {opp_type}  |  "
        f"modelos → {model_dir}  |  logs → {LOG_DIR}/{run_name}"
    )

    # ── Guardar config ───────────────────────────────────────────────────────
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({**cfg, "run_name": run_name, "started_at": datetime.now().isoformat()}, f, indent=2)

    # ── Entornos ─────────────────────────────────────────────────────────────
    # Guardamos referencia directa al EscobaEnv para poder cambiar el boss
    escoba_env = EscobaEnv(
        render_mode=None,
        opponent_type=opp_type,
        boss_deterministic=cfg.get("boss_deterministic", True),
    )
    if use_self_play and boss_manager and boss_manager.has_bosses():
        escoba_env.opponent_model = boss_manager.current_model

    env = Monitor(escoba_env, filename=os.path.join(model_dir, "train_monitor.csv"))

    eval_env = Monitor(
        EscobaEnv(render_mode=None, opponent_type=cfg["opponent_type"]),
        filename=os.path.join(model_dir, "eval_monitor.csv"),
    )

    # ── Modelo ───────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=EscobaFeaturesExtractor,
        features_extractor_kwargs=dict(
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            features_dim=cfg["features_dim"],
        ),
    )
    model = PPO(
        cfg["policy"],
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=LOG_DIR,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        ent_coef=cfg["ent_coef"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=0,
    )

    # BossManager puede ser None si solo usamos self-checkpoints
    if use_self_play and boss_manager is None:
        boss_manager = BossManager(
            rotation_mode=cfg.get("boss_rotation_mode", "cyclic"),
            deterministic=cfg.get("boss_deterministic", True),
            logger=logger,
        )

    stats_callback = EscobaStatsCallback(boss_manager=boss_manager)

    callbacks = [eval_callback, stats_callback]

    if use_self_play:
        if cfg.get("use_frozen_self_checkpoints", False):
            ckpt_dir = os.path.join(model_dir, cfg.get("self_checkpoint_dir", "checkpoints"))
            callbacks.append(SelfCheckpointCallback(
                boss_manager=boss_manager,
                checkpoint_dir=ckpt_dir,
                freq=cfg.get("self_checkpoint_freq", 100_000),
            ))

        callbacks.append(BossRotationCallback(
            boss_manager=boss_manager,
            escoba_env=escoba_env,
            change_steps=cfg["change_boss_steps"],
            stats_cb=stats_callback,
            logger=logger,
        ))

    # ── Entrenar ─────────────────────────────────────────────────────────────
    logger.info(f"Iniciando entrenamiento: {cfg['total_timesteps']:,} pasos…")
    t0 = time.time()

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=CallbackList(callbacks),
        tb_log_name="PPO",
        progress_bar=True,
    )

    elapsed = time.time() - t0
    logger.info(f"Completado en {elapsed / 60:.1f} min.")

    # ── Guardar modelo final ──────────────────────────────────────────────────
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    logger.info(f"Modelo final  → {final_path}.zip")
    logger.info(f"Mejor modelo  → {os.path.join(model_dir, 'best_model.zip')}")
    logger.info(f"TensorBoard   → tensorboard --logdir {LOG_DIR}")

    # ── Gráficas ─────────────────────────────────────────────────────────────
    plot_results(stats_callback, boss_manager, model_dir)
    logger.info(f"Gráficas      → {model_dir}/training_*.png")

    # Log resumen de bosses
    if boss_manager and boss_manager.has_bosses():
        summary = boss_manager.summary()
        logger.info("[BossManager] Resumen final:")
        for bid, s in summary.items():
            wr = f"{s['win_rate']:.2%}" if s["win_rate"] is not None else "N/A"
            logger.info(f"  {bid}: episodes={s['episodes']}  win_rate={wr}  self={s['is_self']}")

    with open(config_path) as f:
        c = json.load(f)
    c["finished_at"]  = datetime.now().isoformat()
    c["elapsed_min"]  = round(elapsed / 60, 2)
    with open(config_path, "w") as f:
        json.dump(c, f, indent=2)

    return model, final_path


def probar_agente(model_path: str, n_episodios: int = 3):
    env   = EscobaEnv(render_mode="human")
    model = PPO.load(model_path)
    for ep in range(n_episodios):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.5)
        print(f"Episodio {ep + 1}: recompensa = {total_reward:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  TESTS BÁSICOS
# ══════════════════════════════════════════════════════════════════════════════

def _test_boss_rotation():
    """Verifica que BossManager rota correctamente en los tres modos."""
    print("test_boss_rotation...", end=" ")
    logger = logging.getLogger("test")

    for mode in ("cyclic", "random", "weighted"):
        bm = BossManager(rotation_mode=mode, logger=logger)
        # Añadimos entradas falsas (paths inexistentes a propósito, con model pre-cargado)
        for i in range(3):
            bm._entries.append({"id": f"fake_{i}", "path": "", "model": None, "is_self": False})
            bm._results[f"fake_{i}"] = [1, 0, 1, 0, 1]  # 60 % win rate

        seen = set()
        bm._idx = 0
        for _ in range(9):
            seen.add(bm.boss_id)
            bm.rotate()
        assert len(seen) >= 2, f"modo {mode}: no hubo rotación ({seen})"

    print("OK")


def _test_boss_stats():
    """Verifica que per-boss stats se registran correctamente."""
    print("test_boss_stats...", end=" ")

    bm = BossManager(rotation_mode="cyclic")
    bm._entries = [
        {"id": "a", "path": "", "model": None, "is_self": False},
        {"id": "b", "path": "", "model": None, "is_self": False},
    ]
    bm._results = {"a": [], "b": []}

    for _ in range(10):
        bm.record_result("a", "win")
    for _ in range(10):
        bm.record_result("b", "loss")

    assert bm.recent_win_rate("a") == 1.0, "boss a debe tener win_rate=1.0"
    assert bm.recent_win_rate("b") == 0.0, "boss b debe tener win_rate=0.0"

    s = bm.summary()
    assert s["a"]["episodes"] == 10
    assert s["b"]["win_rate"] == 0.0
    print("OK")


def _test_pool_capped_with_external_bosses():
    """
    3 bosses externos + self-checkpoints: verifica que el pool no crece
    indefinidamente y que el índice activo nunca apunta fuera de rango.
    """
    print("test_pool_capped_with_external_bosses...", end=" ")

    bm = BossManager(rotation_mode="cyclic")
    # Simular 3 bosses externos (sin path real)
    for i in range(3):
        bm._entries.append({"id": f"ext_{i}", "path": "", "model": None, "is_self": False})
        bm._results[f"ext_{i}"] = []

    # Añadir más self-checkpoints de los que permite _MAX_SELF_CHECKPOINTS
    for step in range((_MAX_SELF_CHECKPOINTS + 3) * 100_000, 0,
                      -(_MAX_SELF_CHECKPOINTS + 3) * 100_000 // (_MAX_SELF_CHECKPOINTS + 3)):
        # Inyectar directamente (bypass PPO.load para el test)
        bid = f"self_{step}"
        bm._entries.append({"id": bid, "path": "", "model": object(), "is_self": True})
        bm._results[bid] = []
        # Simular la lógica de expulsión manualmente (igual que add_self_checkpoint)
        self_entries = [e for e in bm._entries if e["is_self"]]
        while len(self_entries) > _MAX_SELF_CHECKPOINTS:
            oldest = self_entries[0]
            oldest_idx = next(i for i, e in enumerate(bm._entries) if e["id"] == oldest["id"])
            if bm._idx == oldest_idx:
                bm._idx = (bm._idx + 1) % len(bm._entries)
            bm._entries.pop(oldest_idx)
            bm._results.pop(oldest["id"], None)
            if oldest_idx < bm._idx:
                bm._idx -= 1
            self_entries = [e for e in bm._entries if e["is_self"]]

        # El índice siempre debe ser válido
        assert 0 <= bm._idx < len(bm._entries), (
            f"idx={bm._idx} fuera de rango para pool de {len(bm._entries)}"
        )

    total = len(bm._entries)
    self_count = sum(1 for e in bm._entries if e["is_self"])
    ext_count  = sum(1 for e in bm._entries if not e["is_self"])

    assert ext_count  == 3,                    f"externos esperados=3, got={ext_count}"
    assert self_count <= _MAX_SELF_CHECKPOINTS, f"self-ckpts cap superado: {self_count}"
    assert total == ext_count + self_count
    print(f"OK  (pool final={total}: {ext_count} ext + {self_count} self)")


def _test_self_play_config_validation():
    """Verifica validaciones de configuración."""
    print("test_config_validation...", end=" ")
    logger = logging.getLogger("test")

    # Mutuamente excluyentes
    cfg_bad = {**CONFIG, "self_play": True,
               "boss_model_path": "x", "boss_models_dir": "y",
               "change_boss_steps": 1000}
    try:
        _build_boss_manager(cfg_bad, logger)
        assert False, "debería haber lanzado ValueError"
    except ValueError:
        pass

    # change_boss_steps <= 0
    cfg_bad2 = {**CONFIG, "self_play": True,
                "boss_model_path": None, "boss_models_dir": None,
                "use_frozen_self_checkpoints": True,
                "change_boss_steps": 0}
    try:
        _build_boss_manager(cfg_bad2, logger)
        assert False, "debería haber lanzado ValueError"
    except ValueError:
        pass

    print("OK")


def run_tests():
    logging.disable(logging.CRITICAL)  # silenciar logs durante tests
    _test_boss_rotation()
    _test_boss_stats()
    _test_pool_capped_with_external_bosses()
    _test_self_play_config_validation()
    logging.disable(logging.NOTSET)
    print("Todos los tests pasaron.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        modelo, path = entrenar()
        probar_agente(path)
