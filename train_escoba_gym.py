import os
import glob
import json
import logging
import time
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from escoba_gym import EscobaEnv
from card_encoder import EscobaFeaturesExtractor

# ── Configuración centralizada ──────────────────────────────────────────────
CONFIG = {
    "total_timesteps":  200_000_000,
    "opponent_type":    "greedy",          # "random" | "greedy" | "model"
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
    # ── Encoder de cartas (EscobaFeaturesExtractor) ──────────────────────────
    "embed_dim":        16,   # dimensión del embedding por carta
    "hidden_dim":       32,   # dimensión de la MLP por carta
    "features_dim":     64,   # dimensión del vector latente final
}

LOG_DIR    = "logs"       # TensorBoard logs  →  logs/PPO_N/
MODELS_DIR = "models/PPO" # Modelos guardados →  models/PPO/PPO_N/


# ── Callback de métricas de juego ───────────────────────────────────────────
class EscobaStatsCallback(BaseCallback):
    """
    Acumula estadísticas de juego por rollout y las vuelca a TensorBoard.

    Métricas registradas (prefijo 'escoba/'):
      win_rate    – fracción de episodios ganados
      draw_rate   – fracción de empates
      loss_rate   – fracción de episodios perdidos
      point_diff  – diferencia media de puntos (jugador − oponente)
      points_tu   – puntos medios del jugador
      points_op   – puntos medios del oponente
      escobas_tu  – escobas medias por episodio (jugador)
      escobas_op  – escobas medias por episodio (oponente)
      cards_rate  – fracción de las 40 cartas capturadas por el jugador
      7oro_rate   – fracción de episodios en que se captura el 7 de oros
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._reset_buffers()

    def _reset_buffers(self):
        self._wins     = []
        self._draws    = []
        self._losses   = []
        self._pt_tu    = []
        self._pt_op    = []
        self._esc_tu   = []
        self._esc_op   = []
        self._cards_tu = []
        self._7oro     = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done or "result" not in info:
                continue
            result = info["result"]
            self._wins.append(int(result == "win"))
            self._draws.append(int(result == "draw"))
            self._losses.append(int(result == "loss"))
            self._pt_tu.append(info.get("puntos_tu", 0))
            self._pt_op.append(info.get("puntos_op", 0))
            self._esc_tu.append(info.get("escobas_tu", 0))
            self._esc_op.append(info.get("escobas_op", 0))
            self._cards_tu.append(info.get("cartas_tu", 0))
            self._7oro.append(info.get("tiene_7oro_tu", 0))
        return True

    def _on_rollout_end(self):
        if not self._wins:
            return
        pt_tu = np.array(self._pt_tu)
        pt_op = np.array(self._pt_op)
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
        self._reset_buffers()


def _next_ppo_run_name() -> str:
    """Devuelve el nombre del próximo run (PPO_N) que SB3 va a crear en logs/."""
    existing = glob.glob(os.path.join(LOG_DIR, "PPO_*"))
    nums = []
    for d in existing:
        try:
            nums.append(int(os.path.basename(d).split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_n = max(nums, default=0) + 1
    return f"PPO_{next_n}"


def entrenar():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Determinar nombre del run antes de entrenar ─────────────────────────
    # SB3 creará logs/PPO_N automáticamente; calculamos N de antemano
    # para poder nombrar el directorio de modelos igual.
    run_name   = _next_ppo_run_name()               # e.g. "PPO_9"
    model_dir  = os.path.join(MODELS_DIR, run_name) # models/PPO/PPO_9/
    os.makedirs(model_dir, exist_ok=True)

    # ── Logger solo a fichero ───────────────────────────────────────────────
    logger = logging.getLogger("escoba_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(model_dir, "train.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    logger.info(f"Run: {run_name}  |  oponente: {CONFIG['opponent_type']}  |  modelos → {model_dir}  |  logs → {LOG_DIR}/{run_name}")

    # ── Guardar config ──────────────────────────────────────────────────────
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({**CONFIG, "run_name": run_name, "started_at": datetime.now().isoformat()}, f, indent=2)

    # ── Entornos ────────────────────────────────────────────────────────────
    opp = CONFIG["opponent_type"]
    env = Monitor(
        EscobaEnv(render_mode=None, opponent_type=opp),
        filename=os.path.join(model_dir, "train_monitor.csv")
    )
    eval_env = Monitor(
        EscobaEnv(render_mode=None, opponent_type=opp),
        filename=os.path.join(model_dir, "eval_monitor.csv")
    )

    # ── Modelo ──────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=EscobaFeaturesExtractor,
        features_extractor_kwargs=dict(
            embed_dim=CONFIG["embed_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            features_dim=CONFIG["features_dim"],
        ),
    )

    model = PPO(
        CONFIG["policy"],
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=LOG_DIR,          # SB3 crea logs/PPO_N/
        learning_rate=CONFIG["learning_rate"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"],
        ent_coef=CONFIG["ent_coef"],
        gamma=CONFIG["gamma"],
        gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"],
    )

    # ── Callbacks ───────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,   # → models/PPO/PPO_N/best_model.zip
        log_path=model_dir,               # → models/PPO/PPO_N/evaluations.npz
        eval_freq=CONFIG["eval_freq"],
        n_eval_episodes=CONFIG["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=0,
    )
    stats_callback = EscobaStatsCallback()

    # ── Entrenar ────────────────────────────────────────────────────────────
    logger.info(f"Iniciando entrenamiento: {CONFIG['total_timesteps']:,} pasos…")
    t0 = time.time()

    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=[eval_callback, stats_callback],
        tb_log_name="PPO",               # SB3 crea logs/PPO_N (N = run_name)
        progress_bar=True,
    )

    elapsed = time.time() - t0
    logger.info(f"Completado en {elapsed / 60:.1f} min.")

    # ── Guardar modelo final ─────────────────────────────────────────────────
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    logger.info(f"Modelo final  → {final_path}.zip")
    logger.info(f"Mejor modelo  → {os.path.join(model_dir, 'best_model.zip')}")
    logger.info(f"TensorBoard   → tensorboard --logdir {LOG_DIR}")

    with open(config_path) as f:
        cfg = json.load(f)
    cfg["finished_at"] = datetime.now().isoformat()
    cfg["elapsed_min"]  = round(elapsed / 60, 2)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    return model, final_path


def probar_agente(model_path, n_episodios: int = 3):
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


if __name__ == "__main__":
    modelo, path = entrenar()
    probar_agente(path)
