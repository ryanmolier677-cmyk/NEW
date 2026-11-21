import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle, Arc, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

OUT = "/home/user/NEW/floorplan"
INK = "#222222"
WALL = "#3a3a3a"
SOFT = "#9a9a9a"
ACC = "#c0392b"
BLUE = "#2b6cb0"

# Page: A4 landscape, scale 1:50  (1 m = 20 mm)
PW, PH = 297, 210
S = 20.0
X0, Y0 = -2.4, -2.6
XL, YL = (X0, X0 + PW / S), (Y0, Y0 + PH / S)

# Estimated geometry (metres). x: west->east, y: street(south)->north
W, D = 9.0, 4.5          # interior width along street, depth
T = 0.3                  # wall thickness
BLK_X, BLK_Y = 7.5, 2.5  # enclosed block over the lower stair run (NE corner)
ST_W = 1.0               # stair width
ST_X0 = 5.5              # top of stair (west end)
WIN = (4.6, 6.4)         # shop window
DOOR = (7.95, 8.85)      # entrance door


def page():
    fig = plt.figure(figsize=(PW / 25.4, PH / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def grid(ax, step=0.5, color="#e4e9f0"):
    import numpy as np
    for x in np.arange(-1.0, W + 1.01, step):
        ax.plot([x, x], [-1.0, D + 1.0], color=color, lw=0.4, zorder=0)
    for y in np.arange(-1.0, D + 1.01, step):
        ax.plot([-1.0, W + 1.0], [y, y], color=color, lw=0.4, zorder=0)


def walls(ax):
    # floor
    floor = [(0, 0), (W, 0), (W, BLK_Y), (BLK_X, BLK_Y), (BLK_X, D), (0, D)]
    ax.add_patch(Polygon(floor, closed=True, fc="white", ec="none", zorder=1))
    # solid walls (thick)
    def wall(x, y, w, h):
        ax.add_patch(Rectangle((x, y), w, h, fc=WALL, ec=WALL, lw=0.5, zorder=5))
    wall(-T, -T, T, D + 2 * T)              # west
    wall(-T, D, W + 2 * T, T)               # north
    wall(W, -T, T, D + T)                   # east
    # south facade with openings
    wall(-T, -T, WIN[0] + T, T)
    wall(WIN[1], -T, DOOR[0] - WIN[1], T)
    wall(DOOR[1], -T, W + T - DOOR[1], T)
    # enclosed block (stair passes underneath) - hatched
    ax.add_patch(Rectangle((BLK_X, BLK_Y), W - BLK_X, D - BLK_Y, fc="#eeeeee",
                           ec=WALL, lw=1.4, hatch="////", zorder=3))
    # window glazing
    for dy in (-0.2, -0.1):
        ax.plot(WIN, [dy, dy], color=INK, lw=0.8, zorder=6)
    ax.plot([WIN[0], WIN[0]], [-T, 0], color=INK, lw=0.8, zorder=6)
    ax.plot([WIN[1], WIN[1]], [-T, 0], color=INK, lw=0.8, zorder=6)
    # door: leaf + swing
    dw = DOOR[1] - DOOR[0]
    ax.plot([DOOR[0], DOOR[0]], [0, dw], color=INK, lw=1.2, zorder=6)
    ax.add_patch(Arc((DOOR[0], 0), 2 * dw, 2 * dw, theta1=0, theta2=90,
                     color=INK, lw=0.7, ls=(0, (3, 2)), zorder=6))


def stairs(ax, labels=True):
    y0, y1 = D - ST_W, D
    # open part (visible treads)
    n = int((W - ST_X0) / 0.25)
    for i in range(n + 1):
        x = ST_X0 + i * 0.25
        ls = "-" if x <= BLK_X else (0, (2, 2))
        ax.plot([x, x], [y0, y1], color=INK if x <= BLK_X else SOFT,
                lw=0.6, ls=ls, zorder=7)
    ax.plot([BLK_X, W], [y0, y0], color=SOFT, lw=0.6, ls=(0, (2, 2)), zorder=7)
    # railing
    ax.plot([ST_X0, BLK_X], [y0, y0], color=INK, lw=2.6, zorder=8)
    # direction arrow
    ax.add_patch(FancyArrowPatch((ST_X0 + 0.15, (y0 + y1) / 2), (W - 0.2, (y0 + y1) / 2),
                                 arrowstyle="-|>", mutation_scale=9, color=ACC, lw=0.9, zorder=9))
    if labels:
        ax.text(ST_X0 + 0.25, y0 + 0.62, "DOWN to basement", fontsize=6.5, color=ACC,
                zorder=10, va="center", bbox=dict(fc="white", ec="none", pad=0.5))


def fixtures(ax, labels=True, faint=False):
    c = SOFT if faint else INK
    fs = 6
    # shelves on east wall
    for off, lbl in ((0.0, None),):
        ax.add_patch(Rectangle((W - 0.28, 0.3), 0.28, 2.0, fc="none", ec=c, lw=0.8,
                               ls=(0, (3, 1.5)), zorder=7))
    # AC
    ax.add_patch(Rectangle((W - 0.22, 0.6), 0.22, 0.9, fc=("#dbe8f5" if not faint else "none"),
                           ec=BLUE if not faint else c, lw=0.8, zorder=8))
    # pendants
    for (px, py) in ((1.15, 2.75), (1.35, 2.55), (1.05, 2.45)):
        ax.add_patch(Circle((px, py), 0.14, fc="none", ec=c, lw=0.7, zorder=8))
        ax.plot([px - 0.08, px + 0.08], [py - 0.08, py + 0.08], color=c, lw=0.4, zorder=8)
        ax.plot([px - 0.08, px + 0.08], [py + 0.08, py - 0.08], color=c, lw=0.4, zorder=8)
    # track spots
    for sx in (1.1, 1.5, 1.9):
        ax.add_patch(Circle((sx, 3.35), 0.05, fc=c, ec=c, zorder=8))
    # duct box + vent NW
    ax.add_patch(Rectangle((0, D - 0.45), 0.45, 0.45, fc="#f2f2f2", ec=c, lw=0.6, hatch="xx", zorder=4))
    # extinguisher
    ax.add_patch(Circle((0.62, D - 0.12), 0.1, fc=ACC if not faint else "none", ec=ACC, lw=0.8, zorder=8))
    # sockets
    for (sx, sy, rot) in ((0.06, 3.7, 0), (0.06, 0.8, 0), (7.2, 0.06, 0),
                           (BLK_X - 0.06, 2.75, 0), (W - 0.06, 0.15, 0)):
        ax.add_patch(Rectangle((sx - 0.06, sy - 0.06), 0.12, 0.12, fc="white", ec=c, lw=0.6, zorder=9))
        ax.plot([sx - 0.06, sx + 0.06], [sy, sy], color=c, lw=0.5, zorder=9)
    # leaning white board
    ax.add_patch(Rectangle((BLK_X - 0.12, 2.55), 0.08, 0.9, fc="#ffffff", ec=c, lw=0.6, zorder=8))
    # door mat
    ax.add_patch(Rectangle((7.6, 0.35), 0.55, 0.4, fc="#eeeeee", ec=c, lw=0.4, zorder=2))

    if not labels:
        return
    def lab(x, y, s, **k):
        k.setdefault("fontsize", fs); k.setdefault("color", INK)
        ax.text(x, y, s, zorder=12, **k)
    def lead(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color=SOFT, lw=0.5, zorder=11)
    lab(W + 0.45, 1.3, "Floating shelves x2\n(LED under, el. box below)", va="center")
    lead(W, 1.6, W + 0.42, 1.45)
    lab(W + 0.45, 0.65, "AC unit (high)", va="center", color=BLUE)
    lead(W, 0.9, W + 0.42, 0.7)
    lab(W + 0.45, 3.5, "Enclosed block —\nstair runs underneath\n(\"What's down there?\"\nlettering on west face)",
        va="center")
    lead(8.3, 3.5, W + 0.42, 3.5)
    lab(1.7, 2.35, "Glass pendant\ncluster + track spots", va="top")
    lab(0.62, D - 0.38, "Fire ext.", fontsize=5.5, color=ACC, ha="left", va="top")
    lab(0.5, D - 0.65, "Duct box / vent (NW corner)", fontsize=5.5, color=SOFT, ha="left", va="top")
    lab(BLK_X - 0.2, 2.55, "Loose white board\n(leaning)", fontsize=5.5, ha="right", va="bottom")
    lab(ST_X0 + 1.0, D - ST_W - 0.12, "railing", fontsize=5.5, ha="center", va="top")
    lab(7.6, 0.85, "mat", fontsize=5, color=SOFT)


def street(ax):
    ax.add_patch(Rectangle((-1.0, -1.25), W + 2.0, 0.8, fc="#f4f1ec", ec="none", zorder=0))
    ax.text(W / 2, -0.85, "S T R E E T", ha="center", va="center", fontsize=9, color="#a08f74",
            zorder=1, fontweight="bold")
    ax.text((WIN[0] + WIN[1]) / 2, -0.38, "Shop window", ha="center", va="top", fontsize=6, color=INK, zorder=12)
    ax.text((DOOR[0] + DOOR[1]) / 2, -0.38, "Entrance", ha="center", va="top", fontsize=6, color=INK, zorder=12)
    ax.text(7.18, 0.12, "SALIDA sign above", fontsize=4.8, color=SOFT, ha="center", va="bottom", zorder=12)


def beams(ax):
    import numpy as np
    for x in np.arange(0.35, W, 0.55):
        top = D if x < BLK_X else BLK_Y
        ax.plot([x, x], [0, top], color="#d9c7b4", lw=1.4, zorder=2, alpha=0.55)


def dims(ax):
    def d(x1, y1, x2, y2, txt, off=(0, 0), rot=0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<|-|>", mutation_scale=6,
                                     color=SOFT, lw=0.6, zorder=11))
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], txt, fontsize=6.5, color="#666",
                ha="center", va="center", rotation=rot, zorder=11,
                bbox=dict(fc="white", ec="none", pad=0.6))
    d(0, D + 0.6, W, D + 0.6, "≈ 9.0 m  (measure)")
    d(-0.75, 0, -0.75, D, "≈ 4.5 m  (measure)", rot=90)


def compass(ax, x, y):
    ax.add_patch(FancyArrowPatch((x, y - 0.3), (x, y + 0.35), arrowstyle="-|>",
                                 mutation_scale=10, color=INK, lw=1))
    ax.text(x, y + 0.45, "N", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.text(x, y - 0.42, "(street = S,\nassumed)", ha="center", va="top", fontsize=4.8, color=SOFT)


def scalebar(ax, x, y):
    for i in range(4):
        ax.add_patch(Rectangle((x + i * 0.5, y), 0.5, 0.08, fc=INK if i % 2 == 0 else "white", ec=INK, lw=0.5))
    for i in (0, 1, 2):
        ax.text(x + i * 1.0, y - 0.08, f"{i} m", fontsize=5.5, ha="center", va="top")


def header(ax, title, sub):
    ax.text(X0 + 0.4, YL[1] - 0.35, title, fontsize=15, fontweight="bold", va="top", color=INK)
    ax.text(X0 + 0.4, YL[1] - 0.95, sub, fontsize=7, va="top", color="#666")


with PdfPages(f"{OUT}/office-floorplan.pdf") as pdf:
    # ---------- Page 1: annotated existing plan ----------
    fig, ax = page()
    header(ax, "Ground floor — existing layout",
           "Top-down plan reconstructed from photos. Dimensions are ESTIMATES (≈1:50 on A4) — "
           "tape-measure the walls and correct before final planning.")
    street(ax); beams(ax); walls(ax); stairs(ax); fixtures(ax); dims(ax)
    ax.text(3.2, 1.6, "OPEN FLOOR\n≈ 36 m² usable", ha="center", va="center", fontsize=11,
            color="#bbbbbb", fontweight="bold", zorder=3)
    compass(ax, -1.6, 2.2)
    scalebar(ax, 7.0, -2.0)
    # legend
    lx, ly = 0.0, -1.7
    items = [("sock", "Socket"), ("beam", "Timber ceiling beams (above)"),
             ("hatch", "Enclosed / not usable"), ("ext", "Fire extinguisher")]
    for i, (k, t) in enumerate(items):
        x = lx + i * 2.35
        if k == "sock":
            ax.add_patch(Rectangle((x, ly - 0.06), 0.14, 0.14, fc="white", ec=INK, lw=0.6))
        elif k == "beam":
            ax.plot([x, x + 0.25], [ly, ly], color="#d9c7b4", lw=2)
        elif k == "hatch":
            ax.add_patch(Rectangle((x, ly - 0.08), 0.25, 0.18, fc="#eee", ec=WALL, hatch="////", lw=0.6))
        else:
            ax.add_patch(Circle((x + 0.08, ly + 0.01), 0.08, fc=ACC, ec=ACC))
        ax.text(x + 0.35, ly, t, fontsize=6, va="center")
    pdf.savefig(fig); fig.savefig(f"{OUT}/1-existing-plan.png", dpi=200); plt.close(fig)

    # ---------- Page 2: clean plan on grid for sketching ----------
    fig, ax = page()
    header(ax, "Layout sketch sheet",
           "Grid = 50 cm squares. Fixed elements shown faint. Draw your zones and furniture on top "
           "(or lasso-copy the blocks from page 3).")
    grid(ax); street(ax); walls(ax); stairs(ax, labels=True); fixtures(ax, labels=False, faint=True)
    compass(ax, -1.6, 2.2); scalebar(ax, 7.0, -2.0)
    # notes area
    ax.text(0, -1.55, "Notes:", fontsize=7, fontweight="bold", va="center")
    for i in range(2):
        ax.plot([0.8, W], [-1.6 - i * 0.45, -1.6 - i * 0.45], color="#ddd", lw=0.6)
    pdf.savefig(fig); fig.savefig(f"{OUT}/2-sketch-sheet.png", dpi=200); plt.close(fig)

    # ---------- Page 3: furniture blocks, same scale ----------
    fig, ax = page()
    header(ax, "Furniture blocks — same scale as plan",
           "Lasso → copy → paste onto page 2. Sizes in cm (W × D).")
    blocks = [
        ("Desk 160×80", 1.6, 0.8, 2), ("Desk 140×70", 1.4, 0.7, 3), ("Desk 120×60", 1.2, 0.6, 2),
        ("Meeting table 200×100", 2.0, 1.0, 1), ("Meeting table 160×80", 1.6, 0.8, 1),
        ("Sofa 200×85", 2.0, 0.85, 1), ("Armchair 80×80", 0.8, 0.8, 2),
        ("Reception counter 160×60", 1.6, 0.6, 1), ("Bookcase 80×35", 0.8, 0.35, 4),
        ("Filing cab. 45×60", 0.45, 0.6, 3), ("Printer stand 60×50", 0.6, 0.5, 1),
        ("Coffee table 100×50", 1.0, 0.5, 1),
    ]
    x, y, rowh = -1.0, 6.4, 0
    for name, w, d, n in blocks:
        need = n * (w + 0.25)
        if x + need > 11.6:
            x = -1.0; y -= rowh + 0.65; rowh = 0
        for i in range(n):
            ax.add_patch(Rectangle((x + i * (w + 0.25), y - d), w, d, fc="#f7f3ea", ec=INK, lw=0.8))
        ax.text(x, y + 0.08, name, fontsize=6, va="bottom")
        x += need + 0.35; rowh = max(rowh, d)
    # round items
    y -= rowh + 1.0; x = -0.8
    for name, r, n in (("Office chair Ø60", 0.3, 6), ("Round table Ø100", 0.5, 2),
                       ("Round table Ø80", 0.4, 1), ("Plant Ø40", 0.2, 4)):
        ax.text(x - 0.2, y + r + 0.1, name, fontsize=6, va="bottom")
        for i in range(n):
            ax.add_patch(Circle((x + i * (2 * r + 0.15), y), r, fc="#f7f3ea", ec=INK, lw=0.8))
        x += n * (2 * r + 0.15) + 0.6
    ax.text(-1.0, Y0 + 0.4, "Clearance rules of thumb: 90–100 cm behind a desk chair · 80 cm walkways · "
            "60–75 cm around meeting tables · keep stair top and entrance clear.",
            fontsize=6.5, color="#666")
    scalebar(ax, 9.2, 7.6)
    pdf.savefig(fig); fig.savefig(f"{OUT}/3-furniture-blocks.png", dpi=200); plt.close(fig)
print("ok")
