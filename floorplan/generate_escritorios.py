import cv2, numpy as np
R = np.random.default_rng(3)
base = cv2.imread("/home/user/NEW/floorplan/oficina-molduras.jpg").astype(np.float32)
colw = cv2.imread("col_w.png")
Hh, Ww = base.shape[:2]
f, cx, cy, h = 834., 1008., 770., 1.5
C = np.array([0.56, 4.81]); e1 = np.array([-0.8, -0.6]); e2 = np.array([0.6, -0.8])
CAM = np.array([3.334, 3.512])

def P(p, q, z=0.0):
    X, Z = C + p * e1 + q * e2
    return np.array([cx + f * X / Z, cy + f * (h - z) / Z])
def depth(p, q):
    return (C + p * e1 + q * e2)[1]

SS = 3
def polymask(pts):
    m = np.zeros((Hh * SS, Ww * SS), np.uint8)
    cv2.fillPoly(m, [np.round(np.array(pts) * SS).astype(np.int32)], 255, cv2.LINE_AA)
    return cv2.resize(m, (Ww, Hh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

def warp_tex(tex, quad):
    th, tw = tex.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), np.float32(quad))
    return cv2.warpPerspective(tex.astype(np.float32), M, (Ww, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def comp(img, layer, alpha):
    a = alpha[..., None]
    return img * (1 - a) + layer * a

# ---------- textures ----------
def walnut(w, hgt, seed, along="x"):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:hgt, 0:w].astype(np.float32)
    lf = cv2.resize(r.random((6, 10)).astype(np.float32), (w, hgt), interpolation=cv2.INTER_CUBIC)
    wav = yy + 10 * lf + 3 * np.sin(xx / w * 9 + r.random() * 6)
    per = 26.0
    stripes = 0.5 + 0.5 * np.sin(wav / per * 2 * np.pi + np.sin(wav / 37) * 2)
    band = cv2.resize(r.random((max(2, hgt // 18), 3)).astype(np.float32), (w, hgt), interpolation=cv2.INTER_CUBIC)
    fib = cv2.blur(r.normal(0, 1, (hgt, w)).astype(np.float32), (61, 1))
    t = np.clip(0.45 + 0.10 * stripes * (0.5 + band) + 0.10 * fib + 0.22 * (lf - 0.5) + 0.12 * (band - 0.5), 0, 1)[..., None]
    dark = np.array([28, 44, 66], np.float32); light = np.array([66, 100, 140], np.float32)
    tex = dark * (1 - t) + light * t
    return tex if along == "x" else np.ascontiguousarray(tex.transpose(1, 0, 2))

def mirror_tile(chunk, nx, ny):
    row = []
    for j in range(ny):
        cols = []
        for i in range(nx):
            c = chunk
            if i % 2: c = c[:, ::-1]
            if j % 2: c = c[::-1, :]
            cols.append(c)
        row.append(np.concatenate(cols, 1))
    return np.concatenate(row, 0)

def rug_from(chunk, nx, ny, border_col, seed):
    t = mirror_tile(chunk.astype(np.float32), nx, ny)
    hh, ww = t.shape[:2]
    b = max(6, int(min(hh, ww) * 0.06))
    out = t.copy()
    band = np.zeros((hh, ww), np.float32)
    band[:b, :] = 1; band[-b:, :] = 1; band[:, :b] = 1; band[:, -b:] = 1
    inner = np.zeros_like(band); g = b + max(3, b // 3)
    inner[b:g, b:ww - b] = 1; inner[hh - g:hh - b, b:ww - b] = 1; inner[b:hh - b, b:g] = 1; inner[b:hh - b, ww - g:ww - b] = 1
    lum = t.mean(axis=2, keepdims=True) / 160
    out = out * (1 - band[..., None] * 0.75) + (np.array(border_col, np.float32) * lum) * band[..., None] * 0.75
    out = out * (1 - inner[..., None] * 0.35) + (t * 1.15) * inner[..., None] * 0.35
    return np.clip(out, 0, 255)

M = cv2.getRotationMatrix2D((898.7, 1264.3), -18.56, 1.0)
rot = cv2.warpAffine(colw, M, (2016, 1700))
rug1_chunk = rot[1165:1318, 985:1100]
rug2_chunk = rot[1305:1465, 710:855]
rug1 = rug_from(rug1_chunk, 6, 4, (120, 140, 150), 1)
rug2 = rug_from(rug2_chunk, 6, 3, (60, 80, 100), 2)

# chair sprite from the collage
ch = colw[826:1014, 826:992].astype(np.float32)
pm = cv2.imread("paste_mask.png", 0)[826:1014, 826:992].astype(np.float32) / 255
pm[130:, 125:] = 0   # drop laptop corner
pm = cv2.erode((pm > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
band = pm - cv2.erode(pm.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
hs_ch = cv2.cvtColor(ch.astype(np.uint8), cv2.COLOR_BGR2HSV)
pm[(band > 0) & (hs_ch[..., 2] > 185) & (hs_ch[..., 1] < 40)] = 0
pm = cv2.GaussianBlur(pm, (3, 3), 0)
ys = np.where(pm.max(axis=1) > 0.5)[0]
chair_top, chair_cut = ys.min(), ys.max()

img = base.copy()
lumimg = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

def floor_quad(p0, p1, q0, q1):
    return [P(p1, q0), P(p0, q0), P(p0, q1), P(p1, q1)]

def place_rug(img, tex, p0, p1, q0, q1):
    quad = floor_quad(p0, p1, q0, q1)
    lay = warp_tex(tex, quad)
    m = polymask(quad)
    L = np.clip(cv2.GaussianBlur(lumimg, (0, 0), 25) / 150.0, 0.75, 1.2)
    return comp(img, lay * L[..., None], m)

def shadow(img, pts, strength=0.4, blur=25):
    m = polymask(pts)
    m = cv2.GaussianBlur(m, (0, 0), blur)
    return img * (1 - strength * m[..., None])

Lw = np.array([0.45, 0.55, 0.7]); Lw /= np.linalg.norm(Lw)
def box(img, p0, p1, q0, q1, z0, z1, kind="wood", seed=0, along="p", alpha=None):
    corners = {}
    faces = [((0, 0, 1), [(p1, q0, z1), (p0, q0, z1), (p0, q1, z1), (p1, q1, z1)]),
             ((0, 1, 0), [(p1, q1, z1), (p0, q1, z1), (p0, q1, z0), (p1, q1, z0)]),
             ((0, -1, 0), [(p0, q0, z1), (p1, q0, z1), (p1, q0, z0), (p0, q0, z0)]),
             ((1, 0, 0), [(p1, q0, z1), (p1, q1, z1), (p1, q1, z0), (p1, q0, z0)]),
             ((-1, 0, 0), [(p0, q1, z1), (p0, q0, z1), (p0, q0, z0), (p0, q1, z0)])]
    vis = []
    for n, pts in faces:
        c = np.mean(pts, axis=0)
        camv = np.array([CAM[0] - c[0], CAM[1] - c[1], h - c[2]])
        if np.dot(n, camv) > 0:
            vis.append((n, pts, -depth(c[0], c[1])))
    vis.sort(key=lambda t: t[2])
    for n, pts, _ in vis:
        quad = [P(*pt) for pt in pts]
        sh = 0.55 + 0.6 * max(np.dot(n, Lw), 0)
        if kind == "wood":
            # grain along the longer horizontal extent
            tex = walnut(900, 300, seed + int(abs(n[0]) * 7 + abs(n[1]) * 3))
            if (along == "q" and n[2] == 1) or (abs(n[0]) == 1 and along == "q"):
                tex = np.ascontiguousarray(tex.transpose(1, 0, 2)) if n[2] == 1 else tex
            lay = warp_tex(tex, quad) * sh
            m = polymask(quad)
            img = comp(img, lay, m)
            # crisp edge highlight on top face
            if n[2] == 1:
                q = np.round(np.array(quad)).astype(np.int32)
                edge = np.zeros((Hh, Ww), np.float32)
                cv2.polylines(edge, [q], True, 1.0, 1, cv2.LINE_AA)
                img = comp(img, np.full_like(img, 150), edge * 0.35)
        elif kind == "glass":
            m = polymask(quad)
            tint = np.full_like(img, 0); tint[:] = (225, 232, 226)
            img = comp(img, tint * (0.85 + 0.15 * sh), m * 0.22)
            q = np.round(np.array(quad)).astype(np.int32)
            edge = np.zeros((Hh, Ww), np.float32)
            cv2.polylines(edge, [q], True, 1.0, 2, cv2.LINE_AA)
            img = comp(img, np.full_like(img, 240), edge * 0.55)
    return img

def extrude(img, poly, z0, z1, col, spec=0.0):
    faces = [((0, 0, 1), [(pp, qq, z1) for pp, qq in poly])]
    n = len(poly)
    for i in range(n):
        (pa, qa), (pb, qb) = poly[i], poly[(i + 1) % n]
        nx, ny = (qb - qa), -(pb - pa)
        L = np.hypot(nx, ny) + 1e-9; nx, ny = nx / L, ny / L
        cpq = np.mean(poly, axis=0)
        mid = ((pa + pb) / 2, (qa + qb) / 2)
        if (mid[0] - cpq[0]) * nx + (mid[1] - cpq[1]) * ny < 0: nx, ny = -nx, -ny
        faces.append(((nx, ny, 0), [(pa, qa, z1), (pb, qb, z1), (pb, qb, z0), (pa, qa, z0)]))
    vis = []
    for nrm, pts in faces:
        c = np.mean(pts, axis=0)
        camv = np.array([CAM[0] - c[0], CAM[1] - c[1], h - c[2]])
        if np.dot(nrm, camv) > 0: vis.append((nrm, pts, -depth(c[0], c[1])))
    vis.sort(key=lambda t: t[2])
    for nrm, pts, _ in vis:
        quad = [P(*pt) for pt in pts]
        sh = 0.5 + 0.6 * max(np.dot(nrm, Lw), 0) + spec * max(nrm[2], 0)
        lay = np.zeros_like(img); lay[:] = np.array(col, np.float32) * sh
        img = comp(img, lay, polymask(quad))
    return img

def chair_base(img, p, q, leather):
    import math
    # star base legs + wheels
    for k in range(5):
        a = k * 2 * math.pi / 5 + 0.3
        dx, dy = math.cos(a), math.sin(a); ox, oy = -dy * 0.022, dx * 0.022
        leg = [(p + ox, q + oy), (p + dx * 0.32 + ox, q + dy * 0.32 + oy), (p + dx * 0.32 - ox, q + dy * 0.32 - oy), (p - ox, q - oy)]
        img = extrude(img, leg, 0.07, 0.10, (175, 175, 178), spec=0.4)
        wx, wy = p + dx * 0.32, q + dy * 0.32
        circ = [(wx + 0.03 * math.cos(t), wy + 0.03 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
        img = extrude(img, circ, 0.0, 0.06, (30, 30, 32))
    col = [(p - .025, q - .025), (p + .025, q - .025), (p + .025, q + .025), (p - .025, q + .025)]
    img = extrude(img, col, 0.10, 0.50, (190, 190, 192), spec=0.5)
    seat = [(p + 0.23 * math.cos(t), q + 0.22 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    img = extrude(img, seat, 0.50, 0.57, leather)
    return img

def chair(img, p, q, height=1.22, mirror=False, cut_z=0.62):
    spr, msk = ch.copy(), pm.copy()
    if mirror: spr, msk = spr[:, ::-1], msk[:, ::-1]
    top = P(p, q, height); cut = P(p, q, cut_z)
    s = (cut[1] - top[1]) / (chair_cut - chair_top)
    colc = (msk.sum(axis=0) * np.arange(msk.shape[1])).sum() / msk.sum()
    x0 = top[0] - colc * s; y0 = top[1] - chair_top * s
    A = np.float32([[s, 0, x0], [0, s, y0]])
    lay = cv2.warpAffine(spr, A, (Ww, Hh), flags=cv2.INTER_CUBIC)
    m = cv2.warpAffine(msk, A, (Ww, Hh), flags=cv2.INTER_LINEAR)
    return comp(img, lay, np.clip(m, 0, 1))

leather = (48.0, 58.0, 72.0)
hsv = cv2.cvtColor(colw, cv2.COLOR_BGR2HSV)
PM = cv2.imread("paste_mask.png", 0)
def sprite(x0, y0, x1, y1, cond):
    spr = colw[y0:y1, x0:x1].astype(np.float32)
    hs = hsv[y0:y1, x0:x1].astype(np.float32)
    m = ((PM[y0:y1, x0:x1] > 0) & cond(hs)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        k = 1 + np.argmax(st[1:, 4]); m = (lab == k).astype(np.uint8)
    m = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)
    return spr, m
laptop = sprite(945, 948, 1056, 1036, lambda hs: hs[..., 1] < 55)
plant = sprite(1118, 955, 1168, 1012, lambda hs: (hs[..., 1] < 60) | ((hs[..., 0] > 30) & (hs[..., 0] < 95)))
pens = sprite(838, 1005, 900, 1062, lambda hs: hs[..., 1] < 60)
def put(img, spr_m, p, q, width_m, z=0.75, mirror=False):
    spr, m = spr_m
    if mirror: spr, m = spr[:, ::-1], m[:, ::-1]
    ys, xs = np.where(m > 0.5)
    if len(xs) == 0: return img
    wpx = xs.max() - xs.min() + 1
    base_pt = P(p, q, z); Zd = depth(p, q)
    s = (width_m * f / Zd) / wpx
    cxs = (xs.min() + xs.max()) / 2; byy = ys.max()
    A = np.float32([[s, 0, base_pt[0] - cxs * s], [0, s, base_pt[1] - byy * s]])
    lay = cv2.warpAffine(spr, A, (Ww, Hh), flags=cv2.INTER_CUBIC)
    mm = cv2.warpAffine(m, A, (Ww, Hh))
    return comp(img, lay, np.clip(mm, 0, 1))

# ---------------- scene ----------------
# rugs (layered)
img = place_rug(img, rug1, 0.62, 2.25, 0.30, 2.05)
img = place_rug(img, rug2, 1.85, 3.06, 0.20, 1.90)

# desk A: against back wall, facing camera (glass legs + walnut modesty panel), chair behind
A = dict(p0=0.62, p1=1.80, q0=0.85, q1=1.60)
img = shadow(img, [P(A["p1"] + .05, A["q0"] - .05), P(A["p0"] - .05, A["q0"] - .05), P(A["p0"] - .05, A["q1"] + .08), P(A["p1"] + .05, A["q1"] + .08)], 0.35, 16)
img = chair_base(img, 1.21, 0.42, leather)
img = chair(img, 1.21, 0.42)
img = box(img, A["p0"] + .02, A["p0"] + .034, A["q0"] + .03, A["q1"] - .03, 0.0, 0.70, kind="glass")
img = box(img, A["p1"] - .034, A["p1"] - .02, A["q0"] + .03, A["q1"] - .03, 0.0, 0.70, kind="glass")
img = box(img, A["p0"] + .04, A["p1"] - .04, A["q1"] - .07, A["q1"] - .05, 0.28, 0.70, kind="wood", seed=11)
img = box(img, A["p0"], A["p1"], A["q0"], A["q1"], 0.70, 0.75, kind="wood", seed=5)
import math
def rect(pc, qc, w, d, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    return [(pc + ca * x - sa * y, qc + sa * x + ca * y) for x, y in ((-w/2, -d/2), (w/2, -d/2), (w/2, d/2), (-w/2, d/2))]
def books(img, pc, qc, ang, cols, z=0.75):
    for i, c in enumerate(cols):
        img = extrude(img, rect(pc, qc, 0.27 - i * 0.02, 0.20 - i * 0.015, ang + i * 0.12), z + i * 0.03, z + (i + 1) * 0.03 - 0.002, c)
    return img
def vase(img, pc, qc, r=0.065, hgt=0.22, col=(58, 62, 68), z=0.75):
    poly = [(pc + r * math.cos(t), qc + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 20, endpoint=False)]
    return extrude(img, poly, z, z + hgt, col)
img = extrude(img, rect(1.25, 1.12, 0.32, 0.22, 0.08), 0.75, 0.768, (178, 180, 184), spec=0.3)
img = books(img, 0.86, 1.05, -0.15, [(198, 214, 226), (78, 96, 92)])

# chair B tucked toward desk B (drawn first so the desk hides its lower part)
img = shadow(img, [P(2.55, 0.62), P(2.0, 0.62), P(2.0, 1.22), P(2.55, 1.22)], 0.25, 14)
img = chair_base(img, 2.27, 0.92, leather)
img = chair(img, 2.27, 0.92, mirror=True)

# desk B (left side, walnut waterfall), user faces the left wall
B = dict(p0=2.46, p1=3.04, q0=0.30, q1=1.55)
img = shadow(img, [P(B["p1"], B["q0"] - .04), P(B["p0"] - .05, B["q0"] - .04), P(B["p0"] - .05, B["q1"] + .05), P(B["p1"], B["q1"] + .05)], 0.35, 14)
img = box(img, B["p0"], B["p1"], B["q0"], B["q0"] + .04, 0.0, 0.70, kind="wood", seed=21, along="q")
img = box(img, B["p0"], B["p1"], B["q1"] - .04, B["q1"], 0.0, 0.70, kind="wood", seed=22, along="q")
img = box(img, B["p0"] + 0.05, B["p1"] - 0.02, B["q0"] + .04, B["q1"] - .04, 0.54, 0.70, kind="wood", seed=23, along="q")
img = box(img, B["p0"], B["p1"], B["q0"], B["q1"], 0.70, 0.75, kind="wood", seed=24, along="q")
img = books(img, 2.76, 1.22, 0.2, [(70, 98, 150), (200, 214, 224), (60, 70, 80)])
img = vase(img, 2.80, 0.55)

out = np.clip(img, 0, 255).astype(np.uint8)
cv2.imwrite("/home/user/NEW/floorplan/despachos-escritorios.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite("desks_small.jpg", cv2.resize(out, (1344, 1008), interpolation=cv2.INTER_AREA))
print("ok", chair_top, chair_cut)
