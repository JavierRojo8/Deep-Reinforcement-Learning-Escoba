import os
import glob
import json
import logging
import time
from datetime import datetime

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from escoba_gym import EscobaEnv
from card_encoder import EscobaFeaturesExtractor

# ── Configuración centralizada ──────────────────────────────────────────────
CONFIG = {
    "bc_steps":         300_000,     # pasos de behavioral cloning antes de PPO
    "bc_epochs":        20,           # épocas BC por lote
    "bc_batch_size":    512,
    "bc_lr":            3e-4,
    "total_timesteps":  400_000_000,
    "opponent_type":    "mixed",          # Ahora usamos el oponente mixto
    "curriculum_schedule": [
        # (Paso de entrenamiento, Probabilidad de ser Greedy)
        (0,           0.2),   # 0M:  20% Greedy desde el inicio
        (10_000_000,  0.4),   # 10M: 40% Greedy
        (20_000_000,  0.6),   # 20M: 60% Greedy
        (35_000_000,  0.8),   # 35M: 80% Greedy
        (50_000_000,  1.0),   # 50M+: 100% Greedy
    ],
    "policy":           "MultiInputPolicy",
    "n_steps":          1024,
    "batch_size":       512,
    "n_epochs":         10,
    "gamma":            0.99,
    "gae_lambda":       0.95,
    "clip_range":       0.2,
    "eval_freq":        50_000,
    "n_eval_episodes":  20,
    # ── Encoder de cartas (EscobaFeaturesExtractor) ──────────────────────────
    "embed_dim":        16,   # dimensión del embedding por carta
    "hidden_dim":       64,   # dimensión de la MLP por carta
    "features_dim":     64,   # dimensión del vector latente final
    "num_envs":         10,    # número de entornos paralelos para entrenamiento (SubprocVecEnv)
    # ── Hiperparámetros Dinámicos (Schedulers) ─────────────────────────
    "learning_rate_init": 4e-4,
    "learning_rate_end":  1e-4,

    "ent_coef_init":      0.03,
    "ent_coef_end":       0.008,  # Decae a exploración mínima al final
    "ent_decay_start":    20_000_000,   # Empieza cuando rival es ~60% greedy
    "ent_decay_end":      150_000_000,  # Termina de bajar
}

LOG_DIR    = "logs"       # TensorBoard logs  →  logs/PPO_N/
MODELS_DIR = "models/PPO" # Modelos guardados →  models/PPO/PPO_N/

def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Devuelve una función que calcula un valor linealmente decreciente,
    ideal para el learning_rate en Stable-Baselines3.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining va de 1.0 (inicio) a 0.0 (fin)
        return progress_remaining * (initial_value - final_value) + final_value
    return func

def _mask_fn(env):
    return env.action_masks()

def make_env(env_id, opponent_type, seed=0):
    def _init():
        env = EscobaEnv(render_mode=None, opponent_type=opponent_type)
        env.reset(seed=seed + env_id)
        env = ActionMasker(env, _mask_fn)
        return env
    return _init


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


class CurriculumCallback(BaseCallback):
    """
    Sube gradualmente la probabilidad de que el oponente juegue en modo 'greedy'
    siguiendo un calendario de (timestep, prob_greedy).
    """
    def __init__(self, schedule: list, verbose: int = 0):
        super().__init__(verbose)
        # Ordenar por timestep para asegurar la secuencia correcta
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.current_phase_idx = 0

    def _on_step(self) -> bool:
        # Verificar si hemos cruzado el umbral del SIGUIENTE paso en el calendario
        if self.current_phase_idx < len(self.schedule) - 1:
            next_step, next_prob = self.schedule[self.current_phase_idx + 1]
            
            if self.num_timesteps >= next_step:
                self.current_phase_idx += 1
                if self.verbose > 0:
                    print(f"\n[Curriculum] Paso {self.num_timesteps:,} alcanzado.")
                    print(f"Subiendo dificultad: Oponente ahora es {next_prob*100:.0f}% Greedy.\n")
                
                # Actualizar la probabilidad en todos los entornos vectorizados
                self.training_env.env_method("set_prob_greedy", next_prob)
                
        return True

class DynamicEntCoefCallback(BaseCallback):
    """
    Baja el coeficiente de entropía linealmente a medida que avanza el entrenamiento.
    """
    def __init__(self, init_ent, end_ent, start_step, end_step, verbose=0):
        super().__init__(verbose)
        self.init_ent = init_ent
        self.end_ent = end_ent
        self.start_step = start_step
        self.end_step = end_step

    def _on_step(self) -> bool:
        # Calcular el valor actual según el timestep
        if self.num_timesteps <= self.start_step:
            current_ent = self.init_ent
        elif self.num_timesteps >= self.end_step:
            current_ent = self.end_ent
        else:
            progress = (self.num_timesteps - self.start_step) / (self.end_step - self.start_step)
            current_ent = self.init_ent - progress * (self.init_ent - self.end_ent)
        
        # Inyectarlo directamente en el modelo
        self.model.ent_coef = current_ent
        
        # Registrar en TensorBoard para poder ver la curva cayendo
        self.logger.record("config/ent_coef", current_ent)
        return True

def collect_bc_data(n_steps: int, seed: int = 42) -> dict:
    """Run greedy vs greedy, record player observations and greedy actions."""
    env = EscobaEnv(render_mode=None, opponent_type="greedy")
    obs, _ = env.reset(seed=seed)

    all_obs   = {k: [] for k in obs}
    all_masks = []
    all_acts  = []

    collected = 0
    while collected < n_steps:
        action = env.greedy_player_action()
        mask   = env.action_masks()
        for k, v in obs.items():
            all_obs[k].append(v.copy())
        all_masks.append(mask.copy())
        all_acts.append(action)
        collected += 1
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    return {
        "obs":     {k: np.stack(v) for k, v in all_obs.items()},
        "masks":   np.array(all_masks, dtype=bool),
        "actions": np.array(all_acts,  dtype=np.int64),
    }


def pretrain_bc(model, n_steps: int, n_epochs: int = 5,
                batch_size: int = 512, lr: float = 3e-4) -> None:
    """Behavioural cloning warm-start: trains model.policy in-place."""
    print(f"[BC] Collecting {n_steps:,} greedy demos…")
    demos  = collect_bc_data(n_steps)
    N      = len(demos["actions"])
    device = model.policy.device

    obs_tensors = {
        k: torch.tensor(v, dtype=torch.float32 if v.dtype != np.int64 else torch.long, device=device)
        for k, v in demos["obs"].items()
    }
    masks_t   = torch.tensor(demos["masks"],   dtype=torch.bool,  device=device)
    actions_t = torch.tensor(demos["actions"], dtype=torch.long,  device=device)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    print(f"[BC] Training {n_epochs} epoch(s) on {N:,} transitions…")
    for epoch in range(n_epochs):
        perm        = np.random.permutation(N)
        total_loss  = 0.0
        n_batches   = 0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            obs_b  = {k: v[idx] for k, v in obs_tensors.items()}
            mask_b = masks_t[idx]
            act_b  = actions_t[idx]

            _, log_prob, _ = model.policy.evaluate_actions(
                obs_b, act_b, action_masks=mask_b
            )
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        print(f"[BC] Epoch {epoch + 1}/{n_epochs}  loss={total_loss / n_batches:.4f}")

    print("[BC] Pretraining done.")


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




def entrenar(resume_from=None):
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

    # ── Entornos Paralelizados ──────────────────────────────────────────────
    opp = CONFIG["opponent_type"]
    num_envs = CONFIG["num_envs"]
    
    env = SubprocVecEnv([make_env(i, opp) for i in range(num_envs)])
    env = VecMonitor(env, os.path.join(model_dir, "train_monitor"))

    # Aplicar prob_greedy inicial del curriculum (la callback solo actualiza en cambios de fase)
    initial_prob = CONFIG["curriculum_schedule"][0][1]
    env.env_method("set_prob_greedy", initial_prob)

    # El evaluador SIEMPRE juega contra 100% greedy
    eval_env = DummyVecEnv([make_env(99, "greedy")])
    eval_env = VecMonitor(eval_env, os.path.join(model_dir, "eval_monitor"))

    # ── Modelo ──────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=EscobaFeaturesExtractor,
        features_extractor_kwargs=dict(
            embed_dim=CONFIG["embed_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            features_dim=CONFIG["features_dim"],
        ),
    )

    if resume_from is None:
        model = MaskablePPO(
            CONFIG["policy"],
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=LOG_DIR,
            learning_rate=linear_schedule(CONFIG["learning_rate_init"], CONFIG["learning_rate_end"]),
            n_steps=CONFIG["n_steps"],
            batch_size=CONFIG["batch_size"],
            n_epochs=CONFIG["n_epochs"],
            ent_coef=CONFIG["ent_coef_init"],
            gamma=CONFIG["gamma"],
            gae_lambda=CONFIG["gae_lambda"],
            clip_range=CONFIG["clip_range"],
        )
        if CONFIG["bc_steps"] > 0:
            pretrain_bc(
                model,
                n_steps=CONFIG["bc_steps"],
                n_epochs=CONFIG["bc_epochs"],
                batch_size=CONFIG["bc_batch_size"],
                lr=CONFIG["bc_lr"],
            )
        reset_num_timesteps = True
    else:
        model = MaskablePPO.load(resume_from, env=env)
        model.tensorboard_log = LOG_DIR
        reset_num_timesteps = False
        logger.info(f"Reanudando entrenamiento desde {resume_from}…")

    # ── Callbacks ───────────────────────────────────────────────────────────
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=CONFIG["eval_freq"],
        n_eval_episodes=CONFIG["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=0,
    )
    stats_callback = EscobaStatsCallback()
    
    # Instanciamos el nuevo callback con el calendario
    curriculum_callback = CurriculumCallback(
        schedule=CONFIG["curriculum_schedule"], 
        verbose=1
    )

    entropy_callback = DynamicEntCoefCallback(
        init_ent=CONFIG["ent_coef_init"],
        end_ent=CONFIG["ent_coef_end"],
        start_step=CONFIG["ent_decay_start"],
        end_step=CONFIG["ent_decay_end"]
    )

    # ── Entrenar ────────────────────────────────────────────────────────────
    logger.info(f"Iniciando entrenamiento: {CONFIG['total_timesteps']:,} pasos…")
    t0 = time.time()

    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=[eval_callback, stats_callback, curriculum_callback, entropy_callback],
        tb_log_name="PPO",
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
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
    model = MaskablePPO.load(model_path)

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
    resume_from = None
    # resume_from = "models/PPO/PPO_33/final_model.zip"
    modelo, path = entrenar(resume_from=resume_from)
    probar_agente(path)
