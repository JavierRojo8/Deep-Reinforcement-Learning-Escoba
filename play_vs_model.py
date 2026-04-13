import os
import sys
import pygame
import numpy as np
from stable_baselines3 import PPO

from escoba_gym import (
    EscobaEnv,
    HAND_SIZE,
    MAX_TABLE_CARDS,
    POS_MESA,
    POS_MI_MANO,
    POS_MIS_BAZAS,
    POS_OP_BAZAS,
    POS_MANO_OP,
)
# Necesario para que PPO.load() encuentre la clase del extractor
from card_encoder import EscobaFeaturesExtractor  # noqa: F401

MODEL_PATH = r"models\PPO\best_model2"
WINDOW_W, WINDOW_H = 1400, 900
FPS = 30

BG = (34, 110, 70)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
RED = (180, 50, 50)
GOLD = (212, 175, 55)
BLUE = (70, 120, 220)
GRAY = (200, 200, 200)
DARK_GRAY = (80, 80, 80)
GREEN = (40, 160, 90)
YELLOW = (240, 220, 70)

CARD_W, CARD_H = 90, 130
CARD_GAP = 16


def card_to_label(env: EscobaEnv, idx: int) -> str:
    numero_real, es_oros, valor_juego = env._obtener_info_carta(idx)
    palo_idx = idx // 10
    palos_str = ["Oros", "Copas", "Espadas", "Bastos"]
    palo = palos_str[palo_idx]
    n = {10: "Sota", 11: "Caballo", 12: "Rey"}.get(numero_real, str(numero_real))
    return f"{n}\n{palo}\n(v={valor_juego})"


def sorted_hand_indices(env: EscobaEnv, opponent: bool = False):
    pos = POS_MANO_OP if opponent else POS_MI_MANO
    indices = np.where(env.posicion_cartas == pos)[0]
    return sorted(indices, key=env._clave_orden_carta)


def sorted_table_indices(env: EscobaEnv):
    indices = np.where(env.posicion_cartas == POS_MESA)[0]
    return sorted(indices, key=env._clave_orden_carta)


def build_obs_for_side(env: EscobaEnv, side: str):
    """
    Construye la observación desde la perspectiva de 'player' u 'opponent'.
    Usa IDs de carta (1-indexed, 0=vacío) — idéntico a EscobaEnv._get_obs().
    Cuando side='opponent', se intercambian los roles tu/op en globales para
    que el modelo (entrenado como jugador) reciba la perspectiva correcta.
    """
    assert side in ("player", "opponent")

    hand_pos = POS_MI_MANO if side == "player" else POS_MANO_OP
    hand_indices = np.where(env.posicion_cartas == hand_pos)[0]
    hand_indices = sorted(hand_indices, key=env._clave_orden_carta)

    table_indices = sorted_table_indices(env)

    # ── Mano: IDs de carta (1-indexed) ─────────────────────────────────────
    obs_hand = np.zeros(HAND_SIZE, dtype=np.int32)
    for slot, idx in enumerate(hand_indices[:HAND_SIZE]):
        obs_hand[slot] = int(idx) + 1

    # ── Mesa: IDs de carta (1-indexed) ─────────────────────────────────────
    obs_table = np.zeros(MAX_TABLE_CARDS, dtype=np.int32)
    suma_mesa = 0
    for slot, idx in enumerate(table_indices[:MAX_TABLE_CARDS]):
        obs_table[slot] = int(idx) + 1
        _, _, valor_juego = env._obtener_info_carta(idx)
        suma_mesa += valor_juego

    # ── Globales: mismo orden que EscobaEnv._get_obs() ─────────────────────
    # Para 'player': "tu" = jugador principal, "op" = rival.
    # Para 'opponent': invertimos para que el modelo vea su propia perspectiva.
    c = env.contadores
    n_table = len(table_indices)

    if side == "player":
        obs_globales = np.array([
            c["sietes_op"],
            c["sietes_tu"],
            c["cartas_op"],
            c["cartas_tu"],
            c["oros_op"],
            c["oros_tu"],
            c["tiene_7oro_op"],
            c["tiene_7oro_tu"],
            c["escobas_op"],
            c["escobas_tu"],
            len(env.mazo),
            n_table,
            suma_mesa,
        ], dtype=np.int16)
    else:  # opponent: intercambiar tu↔op
        obs_globales = np.array([
            c["sietes_tu"],
            c["sietes_op"],
            c["cartas_tu"],
            c["cartas_op"],
            c["oros_tu"],
            c["oros_op"],
            c["tiene_7oro_tu"],
            c["tiene_7oro_op"],
            c["escobas_tu"],
            c["escobas_op"],
            len(env.mazo),
            n_table,
            suma_mesa,
        ], dtype=np.int16)

    return {
        "hand": obs_hand,
        "table": obs_table,
        "globales": obs_globales,
    }


def apply_move(env: EscobaEnv, side: str, action_slot: int):
    """
    Executes exactly one move for either the human ('player') or the model ('opponent'),
    without auto-playing the other side.
    """
    assert side in ("player", "opponent")

    if side == "player":
        hand_pos = POS_MI_MANO
        capture_pos = POS_MIS_BAZAS
        esc_key = "escobas_tu"
        cards_key = "cartas_tu"
        oros_key = "oros_tu"
        sevens_key = "sietes_tu"
        seven_gold_key = "tiene_7oro_tu"
        bazar_name = "jugador"
    else:
        hand_pos = POS_MANO_OP
        capture_pos = POS_OP_BAZAS
        esc_key = "escobas_op"
        cards_key = "cartas_op"
        oros_key = "oros_op"
        sevens_key = "sietes_op"
        seven_gold_key = "tiene_7oro_op"
        bazar_name = "oponente"

    hand_indices = np.where(env.posicion_cartas == hand_pos)[0]
    hand_indices = sorted(hand_indices, key=env._clave_orden_carta)

    if len(hand_indices) == 0:
        return {
            "done": False,
            "played_card": None,
            "captured": [],
            "escoba": False,
            "reward": 0.0,
        }

    real_slot = min(action_slot, len(hand_indices) - 1)
    card_idx = hand_indices[real_slot]
    _, _, value_card = env._obtener_info_carta(card_idx)

    table_indices = np.where(env.posicion_cartas == POS_MESA)[0]
    captured = env._buscar_mejor_jugada(value_card, table_indices)

    reward = 0.0
    escoba = False

    if captured is not None:
        cards_to_move = list(captured) + [card_idx]
        env.posicion_cartas[cards_to_move] = capture_pos

        if np.count_nonzero(env.posicion_cartas == POS_MESA) == 0:
            env.contadores[esc_key] += 1
            escoba = True
            if side == "player":
                reward += 1.0

        for idx in cards_to_move:
            n, oro, _ = env._obtener_info_carta(idx)
            env.contadores[cards_key] += 1
            if oro:
                env.contadores[oros_key] += 1
            if n == 7:
                env.contadores[sevens_key] += 1
            if n == 7 and oro:
                env.contadores[seven_gold_key] = 1

            if side == "player":
                reward += 0.05
                if oro:
                    reward += 0.1
                if n == 7:
                    reward += 0.2

        env.ultimo_en_bazar = bazar_name
        captured_list = list(captured)
    else:
        env.posicion_cartas[card_idx] = POS_MESA
        captured_list = []

    # end-of-round / end-of-game
    player_hand_n = np.count_nonzero(env.posicion_cartas == POS_MI_MANO)
    op_hand_n = np.count_nonzero(env.posicion_cartas == POS_MANO_OP)

    done = False
    if player_hand_n == 0 and op_hand_n == 0:
        mazo_vacio = env._repartir_nueva_ronda()
        if mazo_vacio:
            env._limpieza_final()
            final_reward = env._calcular_recompensa_final()
            done = True
            if side == "player":
                reward += final_reward
            else:
                reward -= final_reward

    return {
        "done": done,
        "played_card": card_idx,
        "captured": captured_list,
        "escoba": escoba,
        "reward": reward,
    }


class Button:
    def __init__(self, rect, text, color=BLUE):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color

    def draw(self, surface, font):
        pygame.draw.rect(surface, self.color, self.rect, border_radius=10)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=10)
        txt = font.render(self.text, True, WHITE)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


def draw_multiline_text(surface, text, x, y, font, color=BLACK, line_gap=4):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        img = font.render(line, True, color)
        surface.blit(img, (x, y + i * (font.get_height() + line_gap)))


def draw_card(surface, rect, label, font, selected=False, hidden=False, border=BLACK):
    color = YELLOW if selected else WHITE
    pygame.draw.rect(surface, color, rect, border_radius=10)
    pygame.draw.rect(surface, border, rect, 2, border_radius=10)
    if hidden:
        txt = font.render("?", True, BLACK)
        surface.blit(txt, txt.get_rect(center=rect.center))
    else:
        draw_multiline_text(surface, label, rect.x + 8, rect.y + 8, font)


def reset_game(env, seed):
    obs, info = env.reset(seed=seed)
    return {
        "state_msg": "Your turn",
        "done": False,
        "last_player_move": None,
        "last_op_move": None,
        "pending_op_action": None,
    }


def compute_opponent_action(model, env):
    hand = sorted_hand_indices(env, opponent=True)
    if not hand:
        return None
    obs_op = build_obs_for_side(env, "opponent")
    action, _ = model.predict(obs_op, deterministic=True)
    action = int(action)
    action = min(action, len(hand) - 1)
    return action


def main():
    if not os.path.exists(MODEL_PATH + ".zip") and not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Escoba vs PPO model")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 20)
    small_font = pygame.font.SysFont("arial", 16)
    big_font = pygame.font.SysFont("arial", 28, bold=True)

    model = PPO.load(MODEL_PATH)
    env = EscobaEnv(render_mode=None)

    current_seed = np.random.randint(0, 1_000_000)
    game = reset_game(env, current_seed)

    btn_restart = Button((1080, 40, 260, 45), "Restart same seed", GREEN)
    btn_shuffle = Button((1080, 95, 260, 45), "Reshuffle", BLUE)
    btn_op_move = Button((1080, 150, 260, 45), "Opponent move", RED)
    btn_toggle = Button((1080, 205, 260, 45), "Mode: Manual", DARK_GRAY)

    manual_opponent = True

    while True:
        clock.tick(FPS)

        screen.fill(BG)

        # compute pending opponent action if relevant
        if not game["done"]:
            player_has_cards = len(sorted_hand_indices(env, opponent=False)) > 0
            opp_has_cards = len(sorted_hand_indices(env, opponent=True)) > 0
            if opp_has_cards:
                game["pending_op_action"] = compute_opponent_action(model, env)
            else:
                game["pending_op_action"] = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = reset_game(env, current_seed)
                elif event.key == pygame.K_n:
                    current_seed = np.random.randint(0, 1_000_000)
                    game = reset_game(env, current_seed)
                elif event.key == pygame.K_o and not game["done"]:
                    if game["pending_op_action"] is not None:
                        result = apply_move(env, "opponent", game["pending_op_action"])
                        game["last_op_move"] = result
                        game["state_msg"] = "Your turn" if not result["done"] else "Game over"
                        game["done"] = result["done"]

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                if btn_restart.clicked(pos):
                    game = reset_game(env, current_seed)

                elif btn_shuffle.clicked(pos):
                    current_seed = np.random.randint(0, 1_000_000)
                    game = reset_game(env, current_seed)

                elif btn_op_move.clicked(pos) and not game["done"]:
                    if game["pending_op_action"] is not None:
                        result = apply_move(env, "opponent", game["pending_op_action"])
                        game["last_op_move"] = result
                        game["state_msg"] = "Your turn" if not result["done"] else "Game over"
                        game["done"] = result["done"]

                elif btn_toggle.clicked(pos):
                    manual_opponent = not manual_opponent
                    btn_toggle.text = f"Mode: {'Manual' if manual_opponent else 'Auto'}"

                elif not game["done"]:
                    hand = sorted_hand_indices(env, opponent=False)
                    hand_rects = []
                    start_x = (WINDOW_W - (len(hand) * CARD_W + max(0, len(hand) - 1) * CARD_GAP)) // 2
                    y = 720
                    for i, idx in enumerate(hand):
                        r = pygame.Rect(start_x + i * (CARD_W + CARD_GAP), y, CARD_W, CARD_H)
                        hand_rects.append((i, idx, r))

                    for slot, idx, r in hand_rects:
                        if r.collidepoint(pos):
                            result = apply_move(env, "player", slot)
                            game["last_player_move"] = result
                            game["done"] = result["done"]
                            if result["done"]:
                                game["state_msg"] = "Game over"
                            else:
                                game["state_msg"] = "Opponent thinking"

                            if (not manual_opponent) and (not game["done"]):
                                pending = compute_opponent_action(model, env)
                                game["pending_op_action"] = pending
                                if pending is not None:
                                    result_op = apply_move(env, "opponent", pending)
                                    game["last_op_move"] = result_op
                                    game["done"] = result_op["done"]
                                    game["state_msg"] = "Your turn" if not result_op["done"] else "Game over"
                            break

        # Top info
        title = big_font.render("Escoba - You vs PPO", True, WHITE)
        screen.blit(title, (40, 20))

        c = env.contadores
        info_lines = [
            f"State: {game['state_msg']}",
            f"Seed: {current_seed}",
            "",
            f"You     -> cartas:{c['cartas_tu']}  oros:{c['oros_tu']}  sietes:{c['sietes_tu']}  7O:{c['tiene_7oro_tu']}  escobas:{c['escobas_tu']}",
            f"Opponent-> cartas:{c['cartas_op']}  oros:{c['oros_op']}  sietes:{c['sietes_op']}  7O:{c['tiene_7oro_op']}  escobas:{c['escobas_op']}",
            f"Deck remaining: {len(env.mazo)}",
        ]
        y0 = 70
        for line in info_lines:
            img = font.render(line, True, WHITE)
            screen.blit(img, (40, y0))
            y0 += 26

        # Buttons
        btn_restart.draw(screen, font)
        btn_shuffle.draw(screen, font)
        btn_op_move.draw(screen, font)
        btn_toggle.draw(screen, font)

        # Opponent hand
        op_title = big_font.render("Opponent hand (revealed for debugging)", True, WHITE)
        screen.blit(op_title, (40, 240))
        op_hand = sorted_hand_indices(env, opponent=True)
        pending = game["pending_op_action"]

        op_x = 40
        op_y = 285
        for i, idx in enumerate(op_hand):
            rect = pygame.Rect(op_x + i * (CARD_W + CARD_GAP), op_y, CARD_W, CARD_H)
            selected = (pending == i)
            draw_card(screen, rect, card_to_label(env, idx), small_font, selected=selected, border=RED if selected else BLACK)

        if pending is not None and pending < len(op_hand):
            chosen_label = card_to_label(env, op_hand[pending]).replace("\n", " | ")
            txt = font.render(f"Predicted opponent move: slot {pending} -> {chosen_label}", True, WHITE)
            screen.blit(txt, (40, 430))

        # Table
        table_title = big_font.render("Table", True, WHITE)
        screen.blit(table_title, (40, 480))
        table = sorted_table_indices(env)
        tx = 40
        ty = 525
        for i, idx in enumerate(table):
            row = i // 10
            col = i % 10
            rect = pygame.Rect(tx + col * (CARD_W + 8), ty + row * (CARD_H + 8), CARD_W, CARD_H)
            draw_card(screen, rect, card_to_label(env, idx), small_font)

        # Your hand
        my_title = big_font.render("Your hand - click a card to play", True, WHITE)
        screen.blit(my_title, (40, 670))
        my_hand = sorted_hand_indices(env, opponent=False)
        start_x = (WINDOW_W - (len(my_hand) * CARD_W + max(0, len(my_hand) - 1) * CARD_GAP)) // 2
        my_y = 720
        for i, idx in enumerate(my_hand):
            rect = pygame.Rect(start_x + i * (CARD_W + CARD_GAP), my_y, CARD_W, CARD_H)
            draw_card(screen, rect, card_to_label(env, idx), small_font, border=BLUE)

        # Last move logs
        log_x = 1080
        log_y = 290
        pygame.draw.rect(screen, (25, 80, 55), (1060, 270, 300, 540), border_radius=12)
        pygame.draw.rect(screen, BLACK, (1060, 270, 300, 540), 2, border_radius=12)

        screen.blit(big_font.render("Move log", True, WHITE), (1080, 285))
        log_y = 330

        if game["last_player_move"] is not None:
            m = game["last_player_move"]
            played = "None" if m["played_card"] is None else card_to_label(env, m["played_card"]).replace("\n", " | ")
            lines = [
                "Last your move:",
                f"played: {played}",
                f"captured: {len(m['captured'])}",
                f"escoba: {m['escoba']}",
                f"reward: {m['reward']:.2f}",
                "",
            ]
            for line in lines:
                img = small_font.render(line, True, WHITE)
                screen.blit(img, (1080, log_y))
                log_y += 22

        if game["last_op_move"] is not None:
            m = game["last_op_move"]
            played = "None" if m["played_card"] is None else card_to_label(env, m["played_card"]).replace("\n", " | ")
            lines = [
                "Last opponent move:",
                f"played: {played}",
                f"captured: {len(m['captured'])}",
                f"escoba: {m['escoba']}",
                "",
            ]
            for line in lines:
                img = small_font.render(line, True, WHITE)
                screen.blit(img, (1080, log_y))
                log_y += 22

        hints = [
            "Controls:",
            "Click card = play",
            "O = opponent move",
            "R = restart same seed",
            "N = reshuffle",
        ]
        log_y += 20
        for line in hints:
            img = small_font.render(line, True, WHITE)
            screen.blit(img, (1080, log_y))
            log_y += 22

        pygame.display.flip()


if __name__ == "__main__":
    main()