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

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from escoba_gym import EscobaEnv
from card_encoder import EscobaFeaturesExtractor

# ââ ConfiguraciÃ³n centralizada ââââââââââââââââââââââââââââââââââââââââââââââ
CONFIG = {
    "total_timesteps":  800_000_000,
    "opponent_type":    "mixed",          # Ahora usamos el oponente mixto
    "curriculum_schedule": [
        # (Paso de entrenamiento, Probabilidad de ser Greedy)
        (0,          0.0),   # 0 a 20M:   100% Random
        (40_000_000, 0.2),   # 20M a 40M: 20% Greedy
        (60_000_000, 0.4),   # 40M a 60M: 40% Greedy
        (80_000_000, 0.6),   # 60M a 80M: 60% Greedy
        (120_000_000, 0.8),   # 80M a 100M: 80% Greedy
        (160_000_000, 1.0),   # 100M+:      100% Greedy
    ],
    "policy":           "MultiInputPolicy",
    "n_steps":          1024,
    "batch_size":       512,
    "n_epochs":         10,
    "gamma":            0.99,
    "gae_lambda":       0.95,
    "clip_range":       0.2,
    "eval_freq":        200_000,
    "n_eval_episodes":  20,
    # ââ Encoder de cartas (EscobaFeaturesExtractor) ââââââââââââââââââââââââââ
    "embed_dim":        16,   # dimensiÃ³n del embedding por carta
    "hidden_dim":       32,   # dimensiÃ³n de la MLP por carta
    "features_dim":     32,   # dimensiÃ³n del vector latente final
    "num_envs":         64,    # nÃºmero de entornos paralelos para entrenamiento (SubprocVecEnv)
    # ââ HiperparÃ¡metros DinÃ¡micos (Schedulers) âââââââââââââââââââââââââ
    "learning_rate_init": 1e-4,
    "learning_rate_end":  1e-4,   # BajarÃ¡ poco a poco hasta casi cero
    
    "ent_coef_init":      0.008,
    "ent_coef_end":       0.008,  # Al final, jugarÃ¡ casi 100% de memoria, sin azar
    "ent_decay_start":    80_000_000,   # Empieza a bajar cuando el rival es 60% greedy
    "ent_decay_end":      220_000_000,  # Termina de bajar casi al final
}

LOG_DIR    = "logs"       # TensorBoard logs  â  logs/PPO_N/
MODELS_DIR = "models/PPO" # Modelos guardados â  models/PPO/PPO_N/

def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Devuelve una funciÃ³n que calcula un valor linealmente decreciente,
    ideal para el learning_rate en Stable-Baselines3.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining va de 1.0 (inicio) a 0.0 (fin)
        return progress_remaining * (initial_value - final_value) + final_value
    return func

def make_env(env_id, opponent_type, seed=0):
    """
    FunciÃ³n de utilidad para crear instancias independientes del entorno en diferentes procesos.
    """
    def _init():
        env = EscobaEnv(render_mode=None, opponent_type=opponent_type)
        env.reset(seed=seed + env_id)
        return env
    return _init


# ââ Callback de mÃ©tricas de juego âââââââââââââââââââââââââââââââââââââââââââ
class EscobaStatsCallback(BaseCallback):
    """
    Acumula estadÃ­sticas de juego por rollout y las vuelca a TensorBoard.

    MÃ©tricas registradas (prefijo 'escoba/'):
      win_rate    â fracciÃ³n de episodios ganados
      draw_rate   â fracciÃ³n de empates
      loss_rate   â fracciÃ³n de episodios perdidos
      point_diff  â diferencia media de puntos (jugador â oponente)
      points_tu   â puntos medios del jugador
      points_op   â puntos medios del oponente
      escobas_tu  â escobas medias por episodio (jugador)
      escobas_op  â escobas medias por episodio (oponente)
      cards_rate  â fracciÃ³n de las 40 cartas capturadas por el jugador
      7oro_rate   â fracciÃ³n de episodios en que se captura el 7 de oros
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
    Baja el coeficiente de entropÃ­a linealmente a medida que avanza el entrenamiento.
    """
    def __init__(self, init_ent, end_ent, start_step, end_step, verbose=0):
        super().__init__(verbose)
        self.init_ent = init_ent
        self.end_ent = end_ent
        self.start_step = start_step
        self.end_step = end_step

    def _on_step(self) -> bool:
        # Calcular el valor actual segÃºn el timestep
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

def _next_ppo_run_name() -> str:
    """Devuelve el nombre del prÃ³ximo run (PPO_N) que SB3 va a crear en logs/."""
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

    # ââ Determinar nombre del run antes de entrenar âââââââââââââââââââââââââ
    # SB3 crearÃ¡ logs/PPO_N automÃ¡ticamente; calculamos N de antemano
    # para poder nombrar el directorio de modelos igual.
    run_name   = _next_ppo_run_name()               # e.g. "PPO_9"
    model_dir  = os.path.join(MODELS_DIR, run_name) # models/PPO/PPO_9/
    os.makedirs(model_dir, exist_ok=True)

    # ââ Logger solo a fichero âââââââââââââââââââââââââââââââââââââââââââââââ
    logger = logging.getLogger("escoba_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(model_dir, "train.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    logger.info(f"Run: {run_name}  |  oponente: {CONFIG['opponent_type']}  |  modelos â {model_dir}  |  logs â {LOG_DIR}/{run_name}")

    # ââ Guardar config ââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({**CONFIG, "run_name": run_name, "started_at": datetime.now().isoformat()}, f, indent=2)

    # ââ Entornos Paralelizados ââââââââââââââââââââââââââââââââââââââââââââââ
    opp = CONFIG["opponent_type"]
    num_envs = CONFIG["num_envs"]
    
    # Entorno de entrenamiento paralelizado (arranca en "mixed")
    env = SubprocVecEnv([make_env(i, opp) for i in range(num_envs)])
    env = VecMonitor(env, os.path.join(model_dir, "train_monitor"))

    # IMPORTANTE: El evaluador SIEMPRE juega contra 100% greedy para tener 
    # una mÃ©trica de rendimiento "real" que no dependa de la fase curricular.
    eval_env = DummyVecEnv([make_env(99, "greedy")])
    eval_env = VecMonitor(eval_env, os.path.join(model_dir, "eval_monitor"))

    # ââ Modelo ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    policy_kwargs = dict(
        features_extractor_class=EscobaFeaturesExtractor,
        features_extractor_kwargs=dict(
            embed_dim=CONFIG["embed_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            features_dim=CONFIG["features_dim"],
        ),
    )

    if resume_from is None:
        model = PPO(
            CONFIG["policy"],
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=LOG_DIR,
            learning_rate=linear_schedule(CONFIG["learning_rate_init"], CONFIG["learning_rate_end"]), # <-- SCHEDULER LR
            n_steps=CONFIG["n_steps"],
            batch_size=CONFIG["batch_size"],
            n_epochs=CONFIG["n_epochs"],
            ent_coef=CONFIG["ent_coef_init"], # Arranca en el valor inicial
            gamma=CONFIG["gamma"],
            gae_lambda=CONFIG["gae_lambda"],
            clip_range=CONFIG["clip_range"],
        )
        reset_num_timesteps = True
    else:
        model = PPO.load(resume_from, env=env)
        model.tensorboard_log = LOG_DIR
        reset_num_timesteps = False
        logger.info(f"Reanudando entrenamiento desde {resume_from}â¦")

    # ââ Callbacks âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,   # â models/PPO/PPO_N/best_model.zip
        log_path=model_dir,               # â models/PPO/PPO_N/evaluations.npz
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

    # ââ Entrenar ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    logger.info(f"Iniciando entrenamiento: {CONFIG['total_timesteps']:,} pasosâ¦")
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

    # ââ Guardar modelo final âââââââââââââââââââââââââââââââââââââââââââââââââ
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    logger.info(f"Modelo final  â {final_path}.zip")
    logger.info(f"Mejor modelo  â {os.path.join(model_dir, 'best_model.zip')}")
    logger.info(f"TensorBoard   â tensorboard --logdir {LOG_DIR}")

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
    # resume_from = None
    resume_from = "final_model copy.zip"
    modelo, path = entrenar(resume_from=resume_from)
    probar_agente(path)
