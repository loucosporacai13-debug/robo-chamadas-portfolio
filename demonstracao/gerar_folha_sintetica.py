"""Gerador de folha de presença SINTÉTICA / Synthetic attendance-sheet generator.

Autocontido: depende só de numpy e OpenCV. Não importa nenhum módulo do projeto,
não lê fotografias nem arquivos externos. Todo o conteúdo é fictício.
Self-contained: only numpy and OpenCV. No project imports, no external files.

Gera uma "foto" simulada (perspectiva, sombra, ruído, JPEG) de uma lista de
presença genérica com 17 tipos de marca no quadrado "P", e um gabarito JSON
com a marca desenhada e a classe esperada de cada linha.

Uso / Usage:
    python gerar_folha_sintetica.py --saida saida [--condicao base|escura|perspectiva_forte|todas]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ---- layout genérico ---------------------------------------------------------
LARGURA, ALTURA = 1240, 1754
X_COL = [70, 115, 552, 705, 815, 902, 1170]  # Nº | Participante | CPF | ID | P | Assinatura
TOPO_TABELA, ALT_CAB, ALT_LINHA, LADO_QUADRADO = 450, 40, 48, 26
FONTE = cv2.FONT_HERSHEY_SIMPLEX
CANETA = (120, 50, 25)  # azul-escuro (BGR)
CANETA_FRACA = (190, 165, 150)
CANETA_MUITO_FRACA = (215, 205, 195)

P, SEM, REV = "P", "SEM_MARCACAO", "REVISAR"

# tipo de marca -> (classe esperada de um leitor cuidadoso, descrição)
MARCAS = {
    "vazio": (SEM, "quadrado vazio"),
    "x_forte": (P, "X firme"),
    "x_torto": (P, "X torto"),
    "x_pequeno": (P, "X pequeno"),
    "x_fraco": (P, "X em caneta clara"),
    "x_muito_fraco": (REV, "X quase apagado"),
    "risco": (REV, "um único traço"),
    "fora": (SEM, "X fora do quadrado"),
    "assinatura_celula": (SEM, "assinatura invade a célula, não o quadrado"),
    "x_com_assinatura": (P, "X + assinatura na célula"),
    "assinatura_no_quadrado": (REV, "assinatura cruza o quadrado"),
    "rasurado": (REV, "rasura"),
    "cruz": (REV, "cruz (+)"),
    "ponto": (REV, "ponto"),
    "curva": (REV, "arco"),
    "v": (REV, "V"),
    "visto": (REV, "visto"),
}
CONDICOES = {
    "base": {},
    "escura": {"luz": 0.55},
    "perspectiva_forte": {"keystone": 0.07, "rotacao_px": -25},
}


def _txt(img, s, x, y, escala=0.55, cor=60, esp=1):
    cv2.putText(img, s, (x, y), FONTE, escala, (cor, cor, cor), esp, cv2.LINE_AA)


def _traco(img, p0, p1, cor, esp, rng, tremor=0.6):
    """Traço "à mão": polilinha com pequenos desvios perpendiculares."""
    p0, p1 = np.float32(p0), np.float32(p1)
    pts = p0 + (p1 - p0) * np.linspace(0, 1, 8)[:, None]
    normal = np.float32([-(p1 - p0)[1], (p1 - p0)[0]])
    normal /= max(np.linalg.norm(normal), 1e-6)
    pts += normal * rng.normal(0, tremor, (8, 1)).astype(np.float32)
    cv2.polylines(img, [np.round(pts).astype(np.int32)], False, cor, esp, cv2.LINE_AA)


def _assinatura(img, xa, xb, y, rng, amp=6.0):
    xs = np.linspace(xa, xb, 40)
    ys = y + amp * np.sin(np.linspace(0, 5 * np.pi, 40)) + rng.normal(0, 0.8, 40)
    cv2.polylines(img, [np.int32(np.column_stack([xs, ys]))], False, CANETA, 2, cv2.LINE_AA)


def _x(img, cx, cy, r, cor, esp, rng):
    _traco(img, (cx - r, cy - r), (cx + r, cy + r), cor, esp, rng)
    _traco(img, (cx + r, cy - r), (cx - r, cy + r), cor, esp, rng)


def desenhar_marca(img, tipo, cx, cy, rng):
    if tipo in ("x_forte", "x_com_assinatura"):
        _x(img, cx, cy, 8, CANETA, 2, rng)
    elif tipo == "x_torto":
        _traco(img, (cx + 7, cy - 8), (cx - 7, cy + 8), CANETA, 2, rng)
        _traco(img, (cx - 4, cy - 9), (cx + 2, cy + 8), CANETA, 2, rng)
    elif tipo == "x_pequeno":
        _x(img, cx, cy, 5, CANETA, 2, rng)
    elif tipo == "x_fraco":
        _x(img, cx, cy, 8, CANETA_FRACA, 1, rng)
    elif tipo == "x_muito_fraco":
        _x(img, cx, cy, 8, CANETA_MUITO_FRACA, 1, rng)
    elif tipo == "risco":
        _traco(img, (cx + 8, cy - 8), (cx - 8, cy + 8), CANETA, 2, rng)
    elif tipo == "fora":
        _x(img, cx + 29, cy, 7, CANETA, 2, rng)
    elif tipo == "rasurado":
        for k in range(-9, 10, 3):
            _traco(img, (cx - 10, cy + k), (cx + 10, cy + k + 3), CANETA, 1, rng)
        _traco(img, (cx - 10, cy - 10), (cx + 10, cy + 10), CANETA, 2, rng)
    elif tipo == "cruz":
        _traco(img, (cx - 8, cy), (cx + 8, cy), CANETA, 2, rng)
        _traco(img, (cx, cy - 8), (cx, cy + 8), CANETA, 2, rng)
    elif tipo == "ponto":
        cv2.circle(img, (cx + 1, cy), 2, CANETA, -1, cv2.LINE_AA)
    elif tipo == "v":
        _traco(img, (cx - 8, cy - 8), (cx, cy + 8), CANETA, 2, rng)
        _traco(img, (cx, cy + 8), (cx + 8, cy - 8), CANETA, 2, rng)
    elif tipo == "visto":
        _traco(img, (cx - 8, cy), (cx - 3, cy + 7), CANETA, 2, rng)
        _traco(img, (cx - 3, cy + 7), (cx + 9, cy - 9), CANETA, 2, rng)
    elif tipo == "curva":
        cv2.ellipse(img, (cx + 4, cy), (8, 9), 0, 110, 250, CANETA, 2, cv2.LINE_AA)
    if tipo in ("assinatura_celula", "x_com_assinatura"):
        _assinatura(img, X_COL[5] + 120, cx + 18, cy + 4, rng)
    elif tipo == "assinatura_no_quadrado":
        _assinatura(img, X_COL[5] + 120, cx - 30, cy + 1, rng, amp=2.5)


def desenhar_pagina(tipos: list[str]) -> np.ndarray:
    n = len(tipos)
    pg = np.full((ALTURA, LARGURA, 3), 255, np.uint8)
    x0, x1 = X_COL[0], X_COL[-1]

    _txt(pg, "LISTA DE PRESENCA - DEMO", x0, 120, 1.0, 90, 2)
    cv2.line(pg, (x0, 150), (x1, 150), (120, 120, 120), 2)
    cv2.rectangle(pg, (x0, 180), (x1, 420), (205, 205, 205), 1)
    cv2.line(pg, (620, 180), (620, 420), (215, 215, 215), 1)
    for y in (240, 300, 360):
        cv2.line(pg, (x0, y), (x1, y), (225, 225, 225), 1)
    _txt(pg, "CURSO", x0 + 10, 205, 0.45, 110)
    _txt(pg, "Curso Ficticio de Demonstracao", x0 + 10, 230)
    _txt(pg, "CODIGO DA TURMA", 630, 205, 0.45, 110)
    _txt(pg, "DEMO", 630, 230)

    y0, y1 = TOPO_TABELA, TOPO_TABELA + ALT_CAB
    cv2.rectangle(pg, (x0, y0), (x1, y1), (225, 225, 225), -1)
    ys = [y0, y1] + [y1 + (i + 1) * ALT_LINHA for i in range(n)]
    for i in range(1, n, 2):
        ya = y1 + i * ALT_LINHA
        cv2.rectangle(pg, (x0, ya), (x1, ya + ALT_LINHA), (246, 246, 246), -1)
    for y in ys:
        cv2.line(pg, (x0, y), (x1, y), (200, 200, 200), 1)
    for x in X_COL:
        cv2.line(pg, (x, y0), (x, ys[-1]), (190, 190, 190), 1)
    for j, r in enumerate(["No", "Participante", "CPF", "ID", "P", "Assinatura"]):
        cx = (X_COL[j] + X_COL[j + 1]) // 2
        (w, _), _ = cv2.getTextSize(r, FONTE, 0.5, 1)
        _txt(pg, r, X_COL[j] + 8 if j < 2 else cx - w // 2, y1 - 13, 0.5, 70)

    rng = np.random.default_rng(1234)
    for i, tipo in enumerate(tipos):
        ya = y1 + i * ALT_LINHA
        yt, cy = ya + ALT_LINHA // 2 + 6, ya + ALT_LINHA // 2
        cx = (X_COL[4] + X_COL[5]) // 2
        _txt(pg, str(i + 1), X_COL[0] + 12, yt, 0.5, 100)
        _txt(pg, f"PARTICIPANTE FICTICIO {i + 1:02d}", X_COL[1] + 8, yt, 0.5, 70)
        _txt(pg, "000******00", X_COL[2] + 30, yt, 0.5, 90)
        _txt(pg, str(int(rng.integers(100, 99999))), X_COL[3] + 30, yt, 0.5, 90)
        m = LADO_QUADRADO // 2
        cv2.rectangle(pg, (cx - m, cy - m), (cx + m, cy + m), (110, 110, 110), 2)
        desenhar_marca(pg, tipo, cx, cy, rng)

    y_rod = ys[-1] + 130
    for a, b in ((x0, 340), (390, 660), (710, x1)):
        cv2.line(pg, (a, y_rod), (b, y_rod), (110, 110, 110), 2)
    _txt(pg, "Data", 180, y_rod + 25, 0.45, 130)
    return pg


def simular_foto(pagina, keystone=0.04, rotacao_px=15, seed=0, luz=1.0, sombra=0.22):
    """Perspectiva + fundo + sombra + ruído + JPEG."""
    rng = np.random.default_rng(seed)
    H, W = 2000, 1500
    fundo = np.zeros((H, W, 3), np.uint8)
    fundo[:] = (60, 100, 140)
    ph, pw = pagina.shape[:2]
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    k = keystone * pw
    dst = np.float32(
        [[130 + k, 120 + rotacao_px], [1370 - k, 120 - rotacao_px], [1400, 1880], [100, 1880 + rotacao_px]]
    )
    foto = cv2.warpPerspective(
        pagina, cv2.getPerspectiveTransform(src, dst), (W, H), dst=fundo.copy(), borderMode=cv2.BORDER_TRANSPARENT
    )
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    fator = luz * ((1 - sombra) + sombra * (1 - gx / W) * (0.6 + 0.4 * gy / H))
    foto = np.clip(foto.astype(np.float32) * fator[..., None] + rng.normal(0, 3, foto.shape), 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", foto, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gera folhas de presença sintéticas.")
    ap.add_argument("--saida", default="saida")
    ap.add_argument("--condicao", default="todas", choices=[*CONDICOES, "todas"])
    a = ap.parse_args(argv)
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)

    tipos = [*MARCAS, "vazio"]  # uma marca de cada tipo + linha final vazia
    pagina = desenhar_pagina(tipos)
    gabarito = [
        {"linha": i + 1, "marca_desenhada": t, "descricao": MARCAS[t][1], "classe_esperada": MARCAS[t][0]}
        for i, t in enumerate(tipos)
    ]
    (saida / "gabarito.json").write_text(json.dumps(gabarito, ensure_ascii=False, indent=1), encoding="utf-8")
    for nome in CONDICOES if a.condicao == "todas" else [a.condicao]:
        ok, buf = cv2.imencode(".jpg", simular_foto(pagina, **CONDICOES[nome]))
        buf.tofile(str(saida / f"folha_sintetica_{nome}.jpg"))
        print(f"gerado: {saida / f'folha_sintetica_{nome}.jpg'}")
    print(f"gabarito: {saida / 'gabarito.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
