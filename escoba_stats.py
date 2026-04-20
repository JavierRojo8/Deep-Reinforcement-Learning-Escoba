"""
Gr�ficas globales de estad�sticas de Escoba
Uso: python escoba_graficas.py [fichero.jsonl]
Si no se pasa fichero, usa los datos de ejemplo embebidos.
"""

import json
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ?? Paleta ????????????????????????????????????????????????????????????????????
BLUE   = "#378ADD"
CORAL  = "#D85A30"
GREEN  = "#1D9E75"
GRAY   = "#888780"
AMBER  = "#BA7517"
BG     = "#F8F8F6"
CARD   = "#EFEFEB"

# ?? Datos de ejemplo embebidos ?????????????????????????????????????????????????



# ?? Carga de datos ?????????????????????????????????????????????????????????????
def load_data(path=None):
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        raise ValueError("No se ha proporcionado un fichero de datos. Por favor, pasa un archivo JSONL con las estadísticas de las partidas.")
    return [json.loads(line) for line in text.strip().splitlines() if line.strip()]


# ?? Helpers ???????????????????????????????????????????????????????????????????
def avg(data, key):
    return sum(d[key] for d in data) / len(data)

def pct(data, key_tu, key_op):
    """Porcentaje medio de 'tu' sobre el total de ambos."""
    ratios = []
    for d in data:
        total = d[key_tu] + d[key_op]
        ratios.append(d[key_tu] / total * 100 if total else 50)
    return sum(ratios) / len(ratios)


# ?? Estilo global ??????????????????????????????????????????????????????????????
def set_style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.color": "#E0E0DC",
        "grid.linewidth": 0.6,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
    })


# ?? Gr�fica 1: Donut de resultados ????????????????????????????????????????????
def plot_resultados(ax, data):
    wins   = sum(1 for d in data if d["winner"] == "player")
    losses = sum(1 for d in data if d["winner"] == "opponent")
    draws  = sum(1 for d in data if d["winner"] == "draw")
    n = len(data)
    sizes  = [wins, losses, draws]
    colors = [GREEN, CORAL, GRAY]
    labels = [
        f"Victorias\n{wins} ({wins/n*100:.0f}%)",
        f"Derrotas\n{losses} ({losses/n*100:.0f}%)",
        f"Empates\n{draws} ({draws/n*100:.0f}%)",
    ]
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor=BG, linewidth=2),
    )
    ax.text(0, 0, f"{n}\npartidas", ha="center", va="center",
            fontsize=10, color="#444", fontweight="normal")
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, fontsize=8)
    ax.set_title("Resultados")


# ?? Gr�fica 2: Radar (cartas y oros en %, resto en valores medios) ????????????
def plot_radar(ax, data):
    categories = [
        "Puntos/partida",
        "% cartas",
        "% oros",
        "Escobas/partida",
        "7 de oros\n(partidas ganadas)",
    ]
    n_cat = len(categories)
    angles = [math.pi / 2 - 2 * math.pi * i / n_cat for i in range(n_cat)]
    angles += angles[:1]

    player_vals = [
        avg(data, "score_player"),
        pct(data, "cartas_tu", "cartas_op"),
        pct(data, "oros_tu", "oros_op"),
        avg(data, "escobas_tu"),
        avg(data, "tiene_7oro_tu") * 100,   # porcentaje de partidas con 7 de oros
    ]
    opp_vals = [
        avg(data, "score_opponent"),
        100 - player_vals[1],
        100 - player_vals[2],
        avg(data, "escobas_op"),
        avg(data, "tiene_7oro_op") * 100,
    ]

    # Normalizar cada eje 0-100 para que el radar sea uniforme
    maxvals = [
        max(player_vals[i], opp_vals[i]) or 1
        for i in range(n_cat)
    ]
    # Para puntos y escobas escalar a 100; % ya est�n en 0-100
    scale = [100 / maxvals[i] if i in (0, 3) else 1 for i in range(n_cat)]
    pn = [v * scale[i] for i, v in enumerate(player_vals)] + [player_vals[0] * scale[0]]
    on = [v * scale[i] for i, v in enumerate(opp_vals)]    + [opp_vals[0]   * scale[0]]

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks([math.pi / 2 - 2 * math.pi * i / n_cat for i in range(n_cat)])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="#999")
    ax.set_ylim(0, 100)
    ax.spines["polar"].set_color("#CCCCCC")
    ax.grid(color="#E0E0DC", linewidth=0.6)

    angles_rad = [math.pi / 2 - 2 * math.pi * i / n_cat for i in range(n_cat)]
    angles_rad_closed = angles_rad + angles_rad[:1]

    ax.plot(angles_rad_closed, pn, color=BLUE, linewidth=1.8)
    ax.fill(angles_rad_closed, pn, color=BLUE, alpha=0.18)
    ax.plot(angles_rad_closed, on, color=CORAL, linewidth=1.8)
    ax.fill(angles_rad_closed, on, color=CORAL, alpha=0.18)

    ax.set_title("Radar de rendimiento\n(cartas y oros en %)", pad=14)

    patch_p = mpatches.Patch(color=BLUE,  label="Tú")
    patch_o = mpatches.Patch(color=CORAL, label="Rival")
    ax.legend(handles=[patch_p, patch_o], loc="lower left",
              bbox_to_anchor=(-0.15, -0.12))


# ?? Gr�fica 3: Barras agrupadas ? cartas y oros totales ???????????????????????
def plot_cartas_oros(ax, data):
    cats    = ["Cartas\ntotales", "Media\ncartas", "Oros\ntotales", "Media\noros"]
    p_vals  = [
        sum(d["cartas_tu"] for d in data),
        avg(data, "cartas_tu"),
        sum(d["oros_tu"] for d in data),
        avg(data, "oros_tu"),
    ]
    o_vals  = [
        sum(d["cartas_op"] for d in data),
        avg(data, "cartas_op"),
        sum(d["oros_op"] for d in data),
        avg(data, "oros_op"),
    ]
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w/2, p_vals, w, color=BLUE,  label="Tú",    zorder=3)
    ax.bar(x + w/2, o_vals, w, color=CORAL, label="Rival", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_title("Cartas y oros (totales y media)")
    ax.legend()
    ax.grid(axis="x", visible=False)

    for bar in ax.patches:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.3,
            f"{h:.1f}" if h < 10 else f"{h:.0f}",
            ha="center", va="bottom", fontsize=8
        )


# ?? Gr�fica 4: Fuentes de puntuaci�n (partidas ganadas en cada categor�a) ?????
def plot_fuentes(ax, data):
    cats = ["Mayoría\ncartas", "Mayoría\noros", "7 de oros", "Escobas\ntotales"]
    p_vals = [
        sum(1 for d in data if d["cartas_tu"] > d["cartas_op"]),
        sum(1 for d in data if d["oros_tu"] > d["oros_op"]),
        sum(d["tiene_7oro_tu"] for d in data),
        sum(d["escobas_tu"]    for d in data),
    ]
    o_vals = [
        sum(1 for d in data if d["cartas_op"] > d["cartas_tu"]),
        sum(1 for d in data if d["oros_op"] > d["oros_tu"]),
        sum(d["tiene_7oro_op"] for d in data),
        sum(d["escobas_op"]    for d in data),
    ]
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w/2, p_vals, w, color=BLUE,  label="Tú",    zorder=3)
    ax.bar(x + w/2, o_vals, w, color=CORAL, label="Rival", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_title("Fuentes de puntuación")
    ax.legend()
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for bar in ax.patches:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.05,
            f"{h:.0f}", ha="center", va="bottom", fontsize=8
        )


# ?? Gr�fica 5: Puntuaci�n media por resultado ?????????????????????????????????
def plot_score_por_resultado(ax, data):
    grupos = {
        "Victoria": [d for d in data if d["winner"] == "player"],
        "Derrota":  [d for d in data if d["winner"] == "opponent"],
        "Empate":   [d for d in data if d["winner"] == "draw"],
    }
    labels_g = [k for k, v in grupos.items() if v]
    p_avgs   = [avg(v, "score_player")   for v in grupos.values() if v]
    o_avgs   = [avg(v, "score_opponent") for v in grupos.values() if v]
    colors_g = [GREEN, CORAL, GRAY][:len(labels_g)]

    x = np.arange(len(labels_g))
    w = 0.35
    bars_p = ax.bar(x - w/2, p_avgs, w, color=BLUE,  label="Tú",    zorder=3)
    bars_o = ax.bar(x + w/2, o_avgs, w, color=CORAL, label="Rival", zorder=3)

    # Color del borde seg�n resultado
    for bar, col in zip(bars_p, colors_g):
        bar.set_edgecolor(col)
        bar.set_linewidth(1.5)
    for bar, col in zip(bars_o, colors_g):
        bar.set_edgecolor(col)
        bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_g)
    ax.set_title("Puntuación media según resultado")
    ax.legend()
    ax.grid(axis="x", visible=False)

    for bar in list(bars_p) + list(bars_o):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.05,
            f"{h:.1f}", ha="center", va="bottom", fontsize=8
        )


# ?? Layout principal ???????????????????????????????????????????????????????????
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'escoba_stats.jsonl'
    data = load_data(path)
    n    = len(data)
    print(f"Partidas cargadas: {n}")

    set_style()

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    fig.suptitle(
        f"Estadísticas globales de Escoba para {n} partidas",
        fontsize=13, y=0.98, color="#333"
    )

    # Grid: 2 filas � 3 columnas, radar ocupa celda polar
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38,
                          left=0.06, right=0.97, top=0.92, bottom=0.06)

    ax_donut  = fig.add_subplot(gs[0, 0])
    ax_radar  = fig.add_subplot(gs[0, 1], polar=True)
    ax_cartas = fig.add_subplot(gs[0, 2])
    ax_fuent  = fig.add_subplot(gs[1, 0:2])
    ax_score  = fig.add_subplot(gs[1, 2])

    plot_resultados(ax_donut, data)
    plot_radar(ax_radar, data)
    plot_cartas_oros(ax_cartas, data)
    plot_fuentes(ax_fuent, data)
    plot_score_por_resultado(ax_score, data)

    out = Path("escoba_stats.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Gr�fica guardada en: {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()