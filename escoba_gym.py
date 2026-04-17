import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools
import pygame # Opcional, solo si vas a usar renderizado visual

# Definición de constantes para legibilidad
NUM_CARTAS = 40
HAND_SIZE = 3
MAX_TABLE_CARDS = 10
NUM_CARACTERISTICAS_CARTA = 3  # [Valor_juego, es_oros, es_siete] — usado en _codificar_carta_visible

# Mapeo de Posiciones (Codificación entera)
POS_MAZO = 0 
POS_MESA = 1
POS_MI_MANO = 2
POS_MIS_BAZAS = 3
POS_OP_BAZAS = 4
POS_MANO_OP = 5    # Cartas que se ha llevado el oponente

class EscobaEnv(gym.Env):
    """
    Entorno personalizado para el juego de la Escoba.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, config_param=None,
                 opponent_type="random", opponent_model=None,
                 boss_deterministic=True):
        # 1. Definir el espacio de observación (Inputs del agente)
        # Ejemplo: Un vector de 4 valores continuos (como posición y velocidad)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            # IDs enteros de carta: 0 = ranura vacía, 1-40 = carta real (índice+1)
            # El encoder aprendido convierte estos IDs en embeddings continuos.
            "hand": spaces.Box(
                low=0,
                high=NUM_CARTAS,
                shape=(HAND_SIZE,),
                dtype=np.int32
            ),
            "table": spaces.Box(
                low=0,
                high=NUM_CARTAS,
                shape=(MAX_TABLE_CARDS,),
                dtype=np.int32
            ),
            "globales": spaces.Box(
                low=0,
                high=np.array([
                    4, 4,       # sietes tu/op
                    40, 40,     # cartas tu/op
                    10, 10,     # oros tu/op
                    1, 1,       # tiene 7 de oros tu/op
                    20, 20,     # escobas tu/op
                    30,         # cartas restantes en mazo
                    10,         # num cartas en mesa
                    100         # suma mesa
                ]),
                shape=(13,),
                dtype=np.int16
            )
        })

        # 2. Definir el espacio de acción (Outputs del agente)
        # Ejemplo: 3 acciones discretas (3 cartas posibles de la mano)
        self.action_space = spaces.Discrete(3) 

        # Estado interno del juego (La "verdad" completa)
        self._estado_juego = None
        self._episode_result = None  # Relleno al terminar cada partida

        # Modo de renderizado
        self.render_mode = render_mode

        # ── NUEVO: tipo de oponente ──────────────────────────────────────────
        # opponent_type: "random" | "greedy" | "model"
        # opponent_model: instancia de PPO (u otro) cargada externamente,
        #                 solo se usa cuando opponent_type == "model"
        if opponent_type not in ("random", "greedy", "model"):
            raise ValueError(f"opponent_type debe ser 'random', 'greedy' o 'model', no '{opponent_type}'")
        self.opponent_type       = opponent_type
        self.opponent_model      = opponent_model
        self._boss_deterministic = boss_deterministic
            

    
    def _obtener_info_carta(self, indice_absoluto):
        """
        Convierte un índice (0-39) en características de la carta.
        Retorna: (numero_real, es_oros, valor_juego)
        """
        palo = indice_absoluto // 10  # 0: Oros, 1: Copas, 2: Espadas, 3: Bastos
        indice_relativo = indice_absoluto % 10 # 0 a 9
        
        es_oros = 1 if palo == 0 else 0
        
        # Mapeo de índices a números escritos en la carta
        # 0->1, 1->2, ..., 6->7, 7->10 (Sota), 8->11 (Caballo), 9->12 (Rey)
        numeros_carta = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
        numero_real = numeros_carta[indice_relativo]
        
        # Mapeo de índices a valor para sumar 15 (La Sota vale 8, Caballo 9, Rey 10)
        valores_juego = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        valor_juego = valores_juego[indice_relativo]
        
        return numero_real, es_oros, valor_juego

    def _codificar_carta_visible(self, indice_absoluto):
        """
        Devuelve una carta visible como [valor_juego, es_oros, es_siete].
        """
        numero_real, es_oros, valor_juego = self._obtener_info_carta(indice_absoluto)
        es_siete = 1 if numero_real == 7 else 0
        return np.array([valor_juego, es_oros, es_siete], dtype=np.int8)
    
    def _clave_orden_carta(self, indice_absoluto):
        """
        Clave de orden canónica para que la mano tenga siempre
        la misma representación independientemente del orden interno.
        """
        numero_real, es_oros, valor_juego = self._obtener_info_carta(indice_absoluto)
        es_siete = 1 if numero_real == 7 else 0
        return (valor_juego, es_oros, es_siete, indice_absoluto)

    def _inicializar_juego(self):
        # Creamos una lista barajada de índices [0..39]
        self.mazo = list(np.arange(NUM_CARTAS))
        self.np_random.shuffle(self.mazo)
        
        estado_cartas = np.zeros(NUM_CARTAS, dtype=int) + POS_MAZO
        
        # 1. Repartir 3 al Jugador
        for _ in range(3):
            c = self.mazo.pop()
            estado_cartas[c] = POS_MI_MANO
            
        # 2. Repartir 3 al Oponente
        for _ in range(3):
            c = self.mazo.pop()
            estado_cartas[c] = POS_MANO_OP # Interno
            
        # 3. Poner 4 en la mesa
        for _ in range(4):
            c = self.mazo.pop()
            estado_cartas[c] = POS_MESA
            
        return estado_cartas

    def _get_info(self):
        info = {
            "escobas_tu":   self.contadores["escobas_tu"],
            "escobas_op":   self.contadores["escobas_op"],
            "cartas_tu":    self.contadores["cartas_tu"],
            "cartas_op":    self.contadores["cartas_op"],
            "oros_tu":      self.contadores["oros_tu"],
            "sietes_tu":    self.contadores["sietes_tu"],
            "tiene_7oro_tu": self.contadores["tiene_7oro_tu"],
        }
        if self._episode_result is not None:
            info.update(self._episode_result)
        return info

    def reset(self, seed=None, options=None):
        # 1. Inicializar semilla de aleatoriedad
        super().reset(seed=seed)

        self.ultimo_en_bazar = None
        self._episode_result = None

        # 2. Configurar estado interno (Barajar y repartir)
        self.posicion_cartas = self._inicializar_juego()
        
        # 3. Reiniciar contadores globales (Puntos, escobas, etc.)
        self.contadores = {
            "escobas_tu": 0, "escobas_op": 0,
            "cartas_tu": 0, "cartas_op": 0, # Para contar al final quien tiene más
            "oros_tu": 0, "oros_op": 0,
            "sietes_tu": 0, "sietes_op": 0,
            "tiene_7oro_tu": 0, "tiene_7oro_op": 0
        }

        # 4. Generar la observación inicial
        observation = self._get_obs()
        info = self._get_info() # Info adicional (opcional)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_obs(self):
        # ----------------------------------------------------------------
        # HAND: IDs de carta (1-indexed; 0 = ranura vacía)
        # El encoder aprendido se encarga de la representación continua.
        # ----------------------------------------------------------------
        obs_hand = np.zeros(HAND_SIZE, dtype=np.int32)
        indices_mano = np.where(self.posicion_cartas == POS_MI_MANO)[0]
        indices_mano = sorted(indices_mano, key=self._clave_orden_carta)

        for slot, idx in enumerate(indices_mano[:HAND_SIZE]):
            obs_hand[slot] = int(idx) + 1  # 1-indexed

        # ----------------------------------------------------------------
        # TABLE: IDs de carta (1-indexed; 0 = ranura vacía)
        # ----------------------------------------------------------------
        obs_table = np.zeros(MAX_TABLE_CARDS, dtype=np.int32)
        indices_mesa = np.where(self.posicion_cartas == POS_MESA)[0]
        indices_mesa = sorted(indices_mesa, key=self._clave_orden_carta)

        suma_mesa = 0
        for slot, idx in enumerate(indices_mesa[:MAX_TABLE_CARDS]):
            obs_table[slot] = int(idx) + 1  # 1-indexed
            _, _, valor_juego = self._obtener_info_carta(idx)
            suma_mesa += valor_juego

        # -----------------------------
        # GLOBALS
        # -----------------------------
        obs_globales = np.array([
            self.contadores["sietes_op"],      # 0  sietes oponente
            self.contadores["sietes_tu"],      # 1  sietes jugador
            self.contadores["cartas_op"],      # 2  cartas oponente
            self.contadores["cartas_tu"],      # 3  cartas jugador
            self.contadores["oros_op"],        # 4  oros oponente
            self.contadores["oros_tu"],        # 5  oros jugador
            self.contadores["tiene_7oro_op"],  # 6  tiene 7 de oros oponente
            self.contadores["tiene_7oro_tu"],  # 7  tiene 7 de oros jugador
            self.contadores["escobas_op"],     # 8  escobas oponente
            self.contadores["escobas_tu"],     # 9  escobas jugador
            len(self.mazo),                    # 10 cartas en mazo
            len(indices_mesa),                 # 11 cartas en mesa
            suma_mesa                          # 12 suma de valores en mesa
        ], dtype=np.int16)

        return {
            "hand": obs_hand,
            "table": obs_table,
            "globales": obs_globales
        }

    # -----------------------------------------------------------
    # LÓGICA DE COMBINATORIA Y PUNTUACIÓN (GREEDY)
    # -----------------------------------------------------------

    def _buscar_mejor_jugada(self, valor_carta_jugada, cartas_mesa_indices):
        """
        Busca todas las combinaciones de cartas en la mesa que, sumadas a la carta
        jugada, den 15. Devuelve la mejor combinación según heurística.
        """
        target = 15 - valor_carta_jugada
        
        # Si la carta sola ya vale más de 15 (imposible en escoba normal) o target < 0
        if target < 0:
            return None

        candidatos = []

        # Buscar subconjuntos que sumen 'target'
        # Probamos combinaciones de tamaño 1 hasta el total de cartas en mesa
        for r in range(1, len(cartas_mesa_indices) + 1):
            for combo_indices in itertools.combinations(cartas_mesa_indices, r):
                # Calcular suma de esta combinación
                suma_combo = 0
                for idx in combo_indices:
                    _, _, v = self._obtener_info_carta(idx)
                    suma_combo += v
                
                if suma_combo == target:
                    candidatos.append(combo_indices)

        if not candidatos:
            return None

        # ELEGIR LA MEJOR COMBINACIÓN
        # Criterios: Escoba > 7 Velo (7 Oros) > Más 7s > Más Oros > Más Cartas
        mejor_score = -1
        mejor_combo = None

        for combo in candidatos:
            score = 0
            # Info detallada de las cartas que nos llevamos
            cartas_combo = [self._obtener_info_carta(i) for i in combo]
            
            # 1. ¿Es Escoba? (Si nos llevamos todas las de la mesa)
            if len(combo) == len(cartas_mesa_indices):
                score += 1000 
            
            # 2. Análisis de cartas individuales en el combo
            for (num, es_oro, _) in cartas_combo:
                if num == 7 and es_oro == 1: score += 150 # El 7 de oros vale mucho
                if num == 7: score += 50                  # Los 7s valen
                if es_oro: score += 20                    # Los oros valen
                score += 1                                # Cantidad de cartas vale
            
            if score > mejor_score:
                mejor_score = score
                mejor_combo = combo

        return mejor_combo

    # -----------------------------------------------------------
    # IMPLEMENTACIÓN DE STEP
    # -----------------------------------------------------------

    def step(self, action):
        """
        action: int (0, 1, 2) índice de la carta en la mano a jugar.
        """
        terminated = False
        truncated = False
        reward = 0.0

        # 1. IDENTIFICAR CARTA JUGADA
        # Buscamos las cartas que están actualmente en la mano del jugador (POS_MI_MANO = 2)
        indices_mano = np.where(self.posicion_cartas == POS_MI_MANO)[0]
        indices_mano = sorted(indices_mano, key=self._clave_orden_carta)
        
        if len(indices_mano) == 0:
            # Caso borde: No quedan cartas, no debería pasar si se controla bien el loop
            return self._get_obs(), 0, True, False, {}

        # Si la acción pide la carta 2 pero solo tengo 1 carta, uso la última disponible
        idx_real_accion = min(action, len(indices_mano) - 1)
        carta_jugada_idx = indices_mano[idx_real_accion]
        
        # Obtener datos de la carta jugada
        num_c, es_oro_c, valor_c = self._obtener_info_carta(carta_jugada_idx)

        # 2. LÓGICA DE PARTIDA (Buscar sumas)
        indices_mesa = np.where(self.posicion_cartas == POS_MESA)[0]
        
        # Usamos la función greedy para ver qué nos llevamos
        cartas_llevadas_indices = self._buscar_mejor_jugada(valor_c, indices_mesa)

        hizo_baza = False
        es_escoba = False

        if cartas_llevadas_indices is not None:
            # --- CASO: SUMAMOS 15 ---
            hizo_baza = True
            cartas_a_mover = list(cartas_llevadas_indices) + [carta_jugada_idx]
            
            # Actualizar posición a "MIS BAZAS" (3)
            self.posicion_cartas[cartas_a_mover] = POS_MIS_BAZAS
            
            # Chequear Escoba: Si la mesa quedó vacía (y no eran las cartas que acabamos de quitar)
            # Como ya movimos las cartas, verificamos si queda algo en POS_MESA
            quedan_en_mesa = np.count_nonzero(self.posicion_cartas == POS_MESA)
            if quedan_en_mesa == 0:
                es_escoba = True
                self.contadores["escobas_tu"] += 1
                reward += 1.0 # Recompensa directa por escoba

            # Actualizar contadores globales (para la observación)
            for idx in cartas_a_mover:
                n, oro, _ = self._obtener_info_carta(idx)
                self.contadores["cartas_tu"] += 1
                if oro: self.contadores["oros_tu"] += 1
                if n == 7: self.contadores["sietes_tu"] += 1
                if n == 7 and oro: self.contadores["tiene_7oro_tu"] = 1
                
                # Pequeña recompensa densa para ayudar al entrenamiento
                reward += 0.05 # Por cada carta
                if oro: reward += 0.1
                if n == 7: reward += 0.2

            # Guardar quién hizo la última baza (para el reparto final de cartas de mesa)
            self.ultimo_en_bazar = "jugador"

        else:
            # --- CASO: NO SUMAMOS 15 ---
            # La carta se queda en la mesa
            self.posicion_cartas[carta_jugada_idx] = POS_MESA
        
        # 3. TURNO DEL OPONENTE (Simulado/Aleatorio por ahora)
        # Aquí deberías implementar una IA simple para el rival o jugar una carta random.
        # Para este esqueleto, simplemente movemos una carta del rival a la mesa o hacemos que juegue.
        self._simular_turno_oponente()

        # 4. GESTIÓN DE RONDAS (REPARTIR)
        # Verificar si ambos jugadores se quedaron sin cartas en mano
        cartas_mano_jug = np.count_nonzero(self.posicion_cartas == POS_MI_MANO)
        cartas_mano_op = np.count_nonzero(self.posicion_cartas == POS_MANO_OP) # cartas actualmente en mano del oponente (estado interno)
        
        # Una forma más robusta de contar manos es tener variables separadas o rangos.
        # Simplificación: Si el jugador tiene 0 cartas, intentamos repartir.
        if cartas_mano_jug == 0 and cartas_mano_op == 0:
            # Buscar cartas en el mazo (que siguen en estado 0 y no son las del op)
            # NOTA: Esto requiere mejorar la gestión de estados para diferenciar "Mazo" de "Mano Rival".
            # Asumiremos que tenemos una función _repartir_si_es_necesario()
            mazo_vacio = self._repartir_nueva_ronda()
            
            if mazo_vacio:
                terminated = True # Fin del juego
                # Asignar cartas sobrantes de la mesa al último que bazó
                self._limpieza_final()
                
                # Calcular recompensa final (Ganar o Perder)
                reward += self._calcular_recompensa_final()

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    # -----------------------------------------------------------
    # TURNO DEL OPONENTE: dispatcher + estrategias
    # -----------------------------------------------------------

    def _simular_turno_oponente(self):
        """Delega al método correspondiente según self.opponent_type."""
        if self.opponent_type == "random":
            self._turno_oponente_random()
        elif self.opponent_type == "greedy":
            self._turno_oponente_greedy()
        elif self.opponent_type == "model":
            self._turno_oponente_model()

    # ── Helper compartido: ejecuta una captura del oponente ─────────────────
    def _ejecutar_captura_oponente(self, carta_idx, combo_indices):
        """Mueve cartas a POS_OP_BAZAS, actualiza contadores y chequea escoba."""
        cartas_a_mover = list(combo_indices) + [carta_idx]
        self.posicion_cartas[cartas_a_mover] = POS_OP_BAZAS

        if np.count_nonzero(self.posicion_cartas == POS_MESA) == 0:
            self.contadores["escobas_op"] += 1

        for idx in cartas_a_mover:
            n, oro, _ = self._obtener_info_carta(idx)
            self.contadores["cartas_op"] += 1
            if oro: self.contadores["oros_op"] += 1
            if n == 7: self.contadores["sietes_op"] += 1
            if n == 7 and oro: self.contadores["tiene_7oro_op"] = 1

        self.ultimo_en_bazar = "oponente"

    # ── Estrategia 1: RANDOM ─────────────────────────────────────────────────
    def _turno_oponente_random(self):
        """Elige una carta al azar; captura si puede (usando greedy de combos)."""
        indices_mano_op = np.where(self.posicion_cartas == POS_MANO_OP)[0]
        if len(indices_mano_op) == 0:
            return

        carta_idx = self.np_random.choice(indices_mano_op)
        _, _, valor_c = self._obtener_info_carta(carta_idx)
        indices_mesa  = np.where(self.posicion_cartas == POS_MESA)[0]
        combo         = self._buscar_mejor_jugada(valor_c, indices_mesa)

        if combo is not None:
            self._ejecutar_captura_oponente(carta_idx, combo)
        else:
            self.posicion_cartas[carta_idx] = POS_MESA

    # ── Estrategia 2: GREEDY ─────────────────────────────────────────────────
    def _turno_oponente_greedy(self):
        """
        Mira todas las cartas de su mano, evalúa la mejor captura posible
        para cada una y elige la jugada de mayor valor.
        Si ninguna carta captura, descarta la carta menos valiosa.

        Prioridades (mismo criterio que _buscar_mejor_jugada, aplicado
        también a la carta jugada):
          escoba > 7 de oros > sietes > oros > cantidad de cartas
        """
        indices_mano_op = np.where(self.posicion_cartas == POS_MANO_OP)[0]
        if len(indices_mano_op) == 0:
            return

        indices_mesa = np.where(self.posicion_cartas == POS_MESA)[0]

        mejor_score = -1
        mejor_carta = None
        mejor_combo = None

        for carta_idx in indices_mano_op:
            _, _, valor_c = self._obtener_info_carta(carta_idx)
            combo = self._buscar_mejor_jugada(valor_c, indices_mesa)
            if combo is None:
                continue
            score = self._puntuar_jugada_greedy(carta_idx, combo, indices_mesa)
            if score > mejor_score:
                mejor_score = score
                mejor_carta = carta_idx
                mejor_combo = combo

        if mejor_carta is not None:
            self._ejecutar_captura_oponente(mejor_carta, mejor_combo)
        else:
            # Ninguna carta puede capturar → descartar la menos valiosa
            carta_a_tirar = self._elegir_carta_a_tirar(indices_mano_op)
            self.posicion_cartas[carta_a_tirar] = POS_MESA

    def _puntuar_jugada_greedy(self, carta_idx, combo_indices, indices_mesa):
        """
        Puntúa la captura (carta jugada + combo de mesa).
        Mismos pesos que _buscar_mejor_jugada para consistencia.
        """
        score = 0
        todas = list(combo_indices) + [carta_idx]

        if len(combo_indices) == len(indices_mesa):  # escoba
            score += 1000

        for idx in todas:
            n, oro, _ = self._obtener_info_carta(idx)
            if n == 7 and oro: score += 150
            if n == 7:         score += 50
            if oro:            score += 20
            score += 1

        return score

    def _elegir_carta_a_tirar(self, indices_mano):
        """
        Cuando no hay captura posible, elige la carta menos valiosa para dejar
        en mesa: evita oros y sietes; entre el resto, prefiere valor bajo.
        """
        def prioridad(idx):
            n, oro, v = self._obtener_info_carta(idx)
            return oro * 100 + int(n == 7) * 50 + v  # menor → mejor para tirar

        return min(indices_mano, key=prioridad)

    # ── Helper: observación desde la perspectiva del oponente ───────────────
    def _get_obs_opponent(self):
        """
        Construye la observación desde la perspectiva del oponente (para self-play).
        Mismo formato que _get_obs() pero con mano = POS_MANO_OP y globales con
        tu↔op intercambiados, de modo que el boss recibe exactamente el mismo tipo
        de observación con el que fue entrenado.
        """
        obs_hand = np.zeros(HAND_SIZE, dtype=np.int32)
        indices_mano_op = np.where(self.posicion_cartas == POS_MANO_OP)[0]
        indices_mano_op = sorted(indices_mano_op, key=self._clave_orden_carta)
        for slot, idx in enumerate(indices_mano_op[:HAND_SIZE]):
            obs_hand[slot] = int(idx) + 1

        obs_table = np.zeros(MAX_TABLE_CARDS, dtype=np.int32)
        indices_mesa = np.where(self.posicion_cartas == POS_MESA)[0]
        indices_mesa = sorted(indices_mesa, key=self._clave_orden_carta)
        suma_mesa = 0
        for slot, idx in enumerate(indices_mesa[:MAX_TABLE_CARDS]):
            obs_table[slot] = int(idx) + 1
            _, _, valor_juego = self._obtener_info_carta(idx)
            suma_mesa += valor_juego

        # Globales con tu↔op invertidos: para el boss, POS_MANO_OP es "tu"
        obs_globales = np.array([
            self.contadores["sietes_tu"],       # 0  ("op" del boss = jugador principal)
            self.contadores["sietes_op"],       # 1  ("tu" del boss = él mismo)
            self.contadores["cartas_tu"],       # 2
            self.contadores["cartas_op"],       # 3
            self.contadores["oros_tu"],         # 4
            self.contadores["oros_op"],         # 5
            self.contadores["tiene_7oro_tu"],   # 6
            self.contadores["tiene_7oro_op"],   # 7
            self.contadores["escobas_tu"],      # 8
            self.contadores["escobas_op"],      # 9
            len(self.mazo),                     # 10
            len(indices_mesa),                  # 11
            suma_mesa,                          # 12
        ], dtype=np.int16)

        return {"hand": obs_hand, "table": obs_table, "globales": obs_globales}

    # ── Estrategia 3: MODEL ──────────────────────────────────────────────────
    def _turno_oponente_model(self):
        """Usa self.opponent_model para decidir la jugada del oponente."""
        if self.opponent_model is None:
            self._turno_oponente_random()
            return

        indices_mano_op = np.where(self.posicion_cartas == POS_MANO_OP)[0]
        if len(indices_mano_op) == 0:
            return

        obs_op = self._get_obs_opponent()
        action, _ = self.opponent_model.predict(obs_op, deterministic=self._boss_deterministic)
        action = int(action)
        action = min(action, len(indices_mano_op) - 1)

        indices_mano_op = sorted(indices_mano_op, key=self._clave_orden_carta)
        carta_idx = indices_mano_op[action]
        _, _, valor_c = self._obtener_info_carta(carta_idx)
        indices_mesa = np.where(self.posicion_cartas == POS_MESA)[0]
        combo = self._buscar_mejor_jugada(valor_c, indices_mesa)

        if combo is not None:
            self._ejecutar_captura_oponente(carta_idx, combo)
        else:
            self.posicion_cartas[carta_idx] = POS_MESA

    def _repartir_nueva_ronda(self):
        """
        Reparte 3 cartas a cada uno del mazo restante.
        Retorna True si el mazo estaba vacío (Fin de partida).
        """
        if len(self.mazo) == 0:
            return True # Mazo vacío, fin del juego
        
        # Repartir al jugador
        for _ in range(3):
            if self.mazo:
                c = self.mazo.pop()
                self.posicion_cartas[c] = POS_MI_MANO
                
        # Repartir al oponente
        for _ in range(3):
            if self.mazo:
                c = self.mazo.pop()
                self.posicion_cartas[c] = POS_MANO_OP
                
        return False

    def _limpieza_final(self):
        """Si quedan cartas en la mesa al final, se las lleva el último que bazó."""
        indices_mesa = np.where(self.posicion_cartas == POS_MESA)[0]
        
        if len(indices_mesa) == 0:
            return

        if self.ultimo_en_bazar is None:
            # Nadie hizo una baza en toda la partida: se las lleva quien tiene más cartas,
            # o el jugador principal en caso de empate (desempate arbitrario pero consistente).
            self.ultimo_en_bazar = (
                "oponente"
                if self.contadores["cartas_op"] > self.contadores["cartas_tu"]
                else "jugador"
            )
        
        destino = POS_MIS_BAZAS if self.ultimo_en_bazar == "jugador" else POS_OP_BAZAS
        self.posicion_cartas[indices_mesa] = destino
        
        # Actualizar contadores finales
        for idx in indices_mesa:
            n, oro, _ = self._obtener_info_carta(idx)
            sufijo = "tu" if destino == POS_MIS_BAZAS else "op"
            
            self.contadores[f"cartas_{sufijo}"] += 1
            if oro: self.contadores[f"oros_{sufijo}"] += 1
            if n == 7: self.contadores[f"sietes_{sufijo}"] += 1
            if n == 7 and oro: self.contadores[f"tiene_7oro_{sufijo}"] = 1

    def _calcular_recompensa_final(self):
        """
        Calcula los puntos de la Escoba (1 por cartas, 1 por oros, etc.)
        Retorna un reward grande si ganas, negativo si pierdes.
        """
        puntos_tu = self.contadores["escobas_tu"]
        puntos_op = self.contadores["escobas_op"]
        
        # 1. Puntos por Cartas (Más de 20)
        if self.contadores["cartas_tu"] > 20: puntos_tu += 1
        elif self.contadores["cartas_op"] > 20: puntos_op += 1
        
        # 2. Puntos por Oros (Más de 5)
        if self.contadores["oros_tu"] > 5: puntos_tu += 1
        elif self.contadores["oros_op"] > 5: puntos_op += 1
        
        # 3. Puntos por Sietes (Más de 2)
        if self.contadores["sietes_tu"] > 2: puntos_tu += 1
        elif self.contadores["sietes_op"] > 2: puntos_op += 1 # Empate a 2 no da punto
        
        # 4. El 7 de Oros
        if self.contadores["tiene_7oro_tu"]: puntos_tu += 1
        elif self.contadores["tiene_7oro_op"]: puntos_op += 1
        
        # Resultado final (solo imprime cuando render_mode == "human")
        if self.render_mode == "human":
            print(f"--- FIN PARTIDA: TU: {puntos_tu} | OP: {puntos_op} ---")

        if puntos_tu > puntos_op:
            self._episode_result = {"puntos_tu": puntos_tu, "puntos_op": puntos_op, "result": "win"}
            return 10.0
        elif puntos_tu < puntos_op:
            self._episode_result = {"puntos_tu": puntos_tu, "puntos_op": puntos_op, "result": "loss"}
            return -10.0
        else:
            self._episode_result = {"puntos_tu": puntos_tu, "puntos_op": puntos_op, "result": "draw"}
            return 0.0

    def render(self):
        if self.render_mode != "human":
            return

        # --- MAPEO VISUAL ---
        palos_str = ["Or", "Co", "Es", "Ba"] # Oros, Copas, Espadas, Bastos
        
        def carta_to_str(idx):
            n_real, _, _ = self._obtener_info_carta(idx)
            palo_idx = idx // 10
            
            # Formatear número (Sota, Caballo, Rey)
            if n_real == 10: n_str = "S"
            elif n_real == 11: n_str = "C"
            elif n_real == 12: n_str = "R"
            else: n_str = str(n_real)
            
            return f"[{n_str} {palos_str[palo_idx]}]"

        # --- RECOPILAR ESTADO ---
        mesa = []
        mi_mano = []
        op_mano = []
        
        for idx, pos in enumerate(self.posicion_cartas):
            if pos == POS_MESA:
                mesa.append(carta_to_str(idx))
            elif pos == POS_MI_MANO:
                mi_mano.append(carta_to_str(idx))
            elif pos == POS_MANO_OP: # Estado interno 5
                op_mano.append("[ * ]") # Oculto para el render

        # --- IMPRIMIR EN PANTALLA ---
        print("\n" + "="*40)
        print(f"TABLERO DE ESCOBA (Paso: {self.contadores.get('pasos', 0)})")
        print("-" * 40)

        # 1. RIVAL (Arriba)
        print(f"RIVAL ({len(op_mano)} cartas):")
        print("  " + " ".join(op_mano))
        print(f"  Escobas: {self.contadores['escobas_op']} | Bazas: {self.contadores['cartas_op']}")

        print("\n" + " "*15 + "MESA")
        print("  " + " ".join(mesa) if mesa else "  [ MESA LIMPIA ]")

        print(f"\nTU ({len(mi_mano)} cartas):")
        display_mano = [f"{c}" for c in mi_mano]
        print("  " + " ".join(display_mano))
        print(f"  Escobas: {self.contadores['escobas_tu']} | Bazas: {self.contadores['cartas_tu']}")
        print(f"  Sietes: {self.contadores['sietes_tu']} | Oros: {self.contadores['oros_tu']}")
        print("-" * 40)

    def _render_frame(self):
        # TODO: Implementar lógica de dibujo (usando Pygame, Matplotlib, etc.)
        pass

    def close(self):
        """
        Cierra recursos externos (ventanas de pygame, conexiones, etc.)
        """
        pass


import time
# from escoba_env import EscobaEnv # (Si lo tienes en otro archivo)

if __name__ == "__main__":
    # Importante: render_mode='human'
    env = EscobaEnv(render_mode="human") 
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    
    while not terminated:
        # Renderizamos el estado ANTES de jugar para ver qué decidir
        env.render()
        
        # Pausa para que puedas leer la consola (0.5 segundos)
        time.sleep(1.5) 
        
        # Acción aleatoria
        action = env.action_space.sample()
        # print(f"\n>>> JUGADOR tira carta índice {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print("\n!!! FIN DE LA PARTIDA !!!")
            env.render() # Ver estado final
            print(f"Recompensa Final: {reward}")