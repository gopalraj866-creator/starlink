#!/usr/bin/env python3
"""
Starlink constellation live simulation (~10,000 satellites) + Earth + Moon.

Pure numpy + Pillow software renderer. Raw frames are piped into ffmpeg, so it
runs on a plain GitHub Actions runner (no GPU, no browser).

Usage:
  python starlink_sim.py --test-frame out.png        # one PNG frame
  python starlink_sim.py --frames 150 --out test.mp4 # short local clip
  python starlink_sim.py --rtmp "$RTMP_URL"          # live stream (loops forever)
"""
import argparse
import datetime as dt
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

W, H = 1280, 720
FPS = int(os.environ.get("FPS", "30"))
TIME_SCALE = float(os.environ.get("TIME_SCALE", "45"))  # sim seconds per real second
N_TOTAL = 10000
CX, CY = 628, 352          # Earth centre on screen
R = 172                    # Earth radius in px
TILT = math.radians(22)    # camera looks slightly from the north
SPIN_DEG_PER_SEC = 0.25 * TIME_SCALE / 60.0 * 6   # Earth spin (visual), deg/s

rng = np.random.default_rng(7)

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for p in (FONT_DIR + name, name):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ----------------------------------------------------------------------------
# Rotation helpers
# ----------------------------------------------------------------------------
def rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


RT = rx(TILT)          # world -> view
RT_INV = RT.T          # view -> world

# ----------------------------------------------------------------------------
# Constellation (shells sum to exactly N_TOTAL)
# name: (planes, sats_per_plane, inclination_deg, display_radius, active_fraction)
# Display radii are exaggerated (real altitude ~550 km = 1.086 R) so the shells
# are visible around the globe, like the reference image.
# ----------------------------------------------------------------------------
SHELLS = [
    (72, 80, 53.0, 1.30, 0.72),   # 5760
    (36, 50, 43.0, 1.26, 0.70),   # 1800
    (36, 40, 70.0, 1.34, 0.70),   # 1440
    (20, 50, 97.6, 1.38, 0.70),   # 1000
]
assert sum(p * s for p, s, *_ in SHELLS) == N_TOTAL


def build_constellation():
    om, inc, rad, u0, nrate, active = [], [], [], [], [], []
    for planes, per, incl, r, act in SHELLS:
        for p in range(planes):
            raan = 2 * math.pi * p / planes
            phase = (p * 0.37) % 1.0
            for k in range(per):
                om.append(raan)
                inc.append(math.radians(incl))
                rad.append(r + rng.normal(0, 0.006))
                u0.append(2 * math.pi * ((k + phase) / per))
                # ~95 min period (real), tiny variation per shell
                nrate.append(2 * math.pi / (95.0 * 60.0) * (1 + 0.01 * (incl - 53) / 40))
                active.append(rng.random() < act)
    a = lambda x: np.array(x, dtype=np.float32)
    return dict(om=a(om), inc=a(inc), r=a(rad), u0=a(u0), n=a(nrate),
                active=np.array(active, dtype=bool))


SATS = build_constellation()
SATS["cos_i"] = np.cos(SATS["inc"])
SATS["sin_i"] = np.sin(SATS["inc"])
SATS["cos_o"] = np.cos(SATS["om"])
SATS["sin_o"] = np.sin(SATS["om"])


def sat_positions(t):
    """Satellite positions in VIEW coords (Earth radius = 1). Returns x, y, z."""
    s = SATS
    u = s["u0"] + s["n"] * t
    xo = s["r"] * np.cos(u)
    yo = s["r"] * np.sin(u)
    x = xo * s["cos_o"] - yo * s["cos_i"] * s["sin_o"]
    y = xo * s["sin_o"] + yo * s["cos_i"] * s["cos_o"]
    z = yo * s["sin_i"]
    ct, st = math.cos(TILT), math.sin(TILT)
    return x, y * ct - z * st, y * st + z * ct


# ----------------------------------------------------------------------------
# Static background: stars, Earth atmosphere glow, orbit paths, Moon, HUD
# ----------------------------------------------------------------------------
def make_stars():
    img = np.zeros((H, W, 3), np.float32)
    base = np.array([2, 3, 8], np.float32)
    img += base
    n = 1700
    xs = rng.integers(0, W, n)
    ys = rng.integers(0, H, n)
    b = rng.random(n) ** 3 * 200 + 25
    tint = rng.random((n, 3)) * 0.25 + 0.85
    img[ys, xs] += b[:, None] * tint
    # faint nebula-like noise
    noise = rng.random((H // 8, W // 8)).astype(np.float32)
    noise = np.array(Image.fromarray((noise * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC),
                     np.float32) / 255
    img += (noise[..., None] ** 3) * np.array([6, 8, 16], np.float32)
    return img


def add_glow(img):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - CX, yy - CY) / R
    g = np.exp(-(d - 1.0) * 7.0) * (d >= 1.0)
    g = g * 0.55
    img += g[..., None] * np.array([30, 90, 230], np.float32)
    # wide soft halo
    img += (np.exp(-(d - 1.0) * 2.2) * (d >= 1.0) * 0.10)[..., None] * np.array([10, 40, 120], np.float32)


def earth_geometry():
    size = 2 * R + 2
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    u = (xx - R) / R
    v = (R - yy) / R
    r2 = u * u + v * v
    mask = r2 <= 1.0
    w = np.sqrt(np.clip(1 - r2, 0, 1))
    view = np.stack([u, -w, v], -1)               # camera at -y
    world = view @ RT_INV.T
    lon = np.arctan2(world[..., 1], world[..., 0])
    lat = np.arcsin(np.clip(world[..., 2], -1, 1))
    sun_view = np.array([0.62, -0.70, 0.30])
    sun_view /= np.linalg.norm(sun_view)
    ndl = (view * sun_view).sum(-1)
    return dict(mask=mask, w=w, lon=lon, lat=lat, ndl=ndl, size=size)


EG = earth_geometry()
def _load(name, w=1024):
    im = Image.open(os.path.join(ASSETS, name)).convert("RGB").resize((w, w // 2), Image.LANCZOS)
    return np.array(im, np.float32)


TEX_DAY = _load("earth_atmos_2048.jpg")
TEX_NIGHT = _load("earth_lights_2048.png")
TH, TW = TEX_DAY.shape[:2]
# night-side texture precombined once: faint land + warm city lights
TEX_NIGHTC = TEX_DAY * 0.05 + TEX_NIGHT * np.array([1.25, 1.0, 0.65], np.float32) * 1.15

_smooth = lambda e0, e1, x: (lambda t: t * t * (3 - 2 * t))(np.clip((x - e0) / (e1 - e0), 0, 1))
_M = EG["mask"]
_ndl = EG["ndl"][_M]
_w = EG["w"][_M]
ROW_M = np.clip(((0.5 - EG["lat"][_M] / math.pi) * TH).astype(np.int32), 0, TH - 1)
LON_M = EG["lon"][_M].astype(np.float32)
DAYW_M = _smooth(-0.12, 0.30, _ndl).astype(np.float32)[:, None]
DIFF_M = ((0.30 + 0.85 * np.clip(_ndl, 0, 1)) * (0.60 + 0.40 * _w ** 0.4)).astype(np.float32)[:, None]
RIM_M = (((1 - _w) ** 3.0)[:, None] * (0.35 + 0.65 * DAYW_M) * np.array([40, 110, 255], np.float32)).astype(np.float32)
DAY_K = DIFF_M * DAYW_M
NIGHT_K = 1.0 - DAYW_M


def render_earth_masked(theta):
    col = (((LON_M - theta) * (TW / (2 * math.pi)) + TW / 2).astype(np.int32)) % TW
    out = TEX_DAY[ROW_M, col] * DAY_K + TEX_NIGHTC[ROW_M, col] * NIGHT_K + RIM_M
    return np.clip(out, 0, 255).astype(np.uint8)


def make_orbit_lines():
    """Thin blue orbit paths (static in inertial frame) as an additive layer."""
    P = int(1.6 * R) * 2
    layer = Image.new("RGB", (P, P), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    ox = oy = P // 2
    segs = []
    for planes, per, incl, r, act in SHELLS:
        step = 2 if planes > 40 else 1
        for p in range(0, planes, 6 if planes > 40 else 4):
            raan = 2 * math.pi * p / planes
            ang = np.linspace(0, 2 * math.pi, 220)
            xo, yo = r * np.cos(ang), r * np.sin(ang)
            ci, si = math.cos(math.radians(incl)), math.sin(math.radians(incl))
            x = xo * math.cos(raan) - yo * ci * math.sin(raan)
            y = xo * math.sin(raan) + yo * ci * math.cos(raan)
            z = yo * si
            ct, st = math.cos(TILT), math.sin(TILT)
            yv, zv = y * ct - z * st, y * st + z * ct
            sx = ox + x * R
            sy = oy - zv * R
            vis = ~((yv > 0) & (x * x + zv * zv < 1.0))
            front = yv <= 0
            for i in range(len(ang) - 1):
                if vis[i] and vis[i + 1]:
                    c = (40, 105, 235) if front[i] else (18, 50, 125)
                    d.line([(sx[i], sy[i]), (sx[i + 1], sy[i + 1])], fill=c, width=1)
    arr = np.array(layer.filter(ImageFilter.GaussianBlur(0.4)), np.float32) * 0.55
    return arr, P


LINES, LP = make_orbit_lines()


def dashed_curve(draw, pts, dash=7, gap=6, fill=(110, 125, 150)):
    acc = 0.0
    on = True
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if on:
            draw.line([(x0, y0), (x1, y1)], fill=fill, width=1)
        acc += seg
        if acc >= (dash if on else gap):
            acc = 0
            on = not on


def make_moon(size=70):
    tex = Image.open(os.path.join(ASSETS, "moon_1024.jpg")).convert("RGB")
    t = np.array(tex, np.float32)
    th, tw = t.shape[:2]
    n = size * 2 + 1
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    u = (xx - size) / size
    v = (size - yy) / size
    r2 = u * u + v * v
    m = r2 <= 1
    w = np.sqrt(np.clip(1 - r2, 0, 1))
    lon = np.arctan2(u, w)
    lat = np.arcsin(np.clip(v, -1, 1))
    c = ((lon / (2 * math.pi) + 0.5) * tw).astype(int) % tw
    r_ = np.clip(((0.5 - lat / math.pi) * th).astype(int), 0, th - 1)
    img = t[r_, c]
    sun = np.array([-0.55, 0.35, 0.75]); sun /= np.linalg.norm(sun)
    ndl = np.clip(u * sun[0] + v * sun[1] + w * sun[2], 0, 1)
    img = img * (0.12 + 1.05 * ndl)[..., None]
    rgba = np.dstack([np.clip(img, 0, 255), (m * 255)]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def build_background():
    img = make_stars()
    add_glow(img)
    bg = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    d = ImageDraw.Draw(bg)

    # --- Moon + dashed orbit arc (top right) ---
    mx, my, mr = 1085, 98, 34
    pts = [(mx + (t - 0.5) * 1.0 * 760 - 40, my - 20 + 330 * ((t - 0.5) ** 2) * 1.6 + (t - 0.5) * 120)
           for t in np.linspace(0.0, 1.0, 120)]
    pts = [(x, y) for x, y in pts if 0 <= x < W and 0 <= y < H]
    dashed_curve(d, pts)
    moon = make_moon(mr)
    bg.alpha_composite(moon, (mx - mr, my - mr))
    d.line([(mx + mr - 2, my + 6), (mx + mr + 26, my - 14), (mx + mr + 112, my - 14)], fill=(60, 120, 230), width=1)
    d.text((mx + mr + 30, my - 40), "Moon", font=font(13, True), fill=(235, 240, 250))
    d.text((mx + mr + 30, my - 10), "384,400 km", font=font(11), fill=(110, 160, 255))

    # --- Title block (top left) ---
    d.text((30, 28), "STARLINK CONSTELLATION", font=font(25, True), fill=(245, 248, 255))
    d.text((30, 66), "~10,000 SATELLITES (SIMULATION)", font=font(16, True), fill=(235, 240, 250))
    stats = [
        ("Total Satellites", "10,000 (approx.)"),
        ("Active (approx.)", "7,000+"),
        ("Orbital Altitude", "~550 km (LEO)"),
        ("Orbital Planes", "~72"),
        ("Inclination", "~53°"),
    ]
    y = 112
    for i, (k, v) in enumerate(stats):
        # simple icon: ring
        ic = (150, 175, 220)
        d.ellipse([32, y + 4, 48, y + 20], outline=ic, width=2)
        d.ellipse([37, y + 9, 43, y + 15], fill=ic)
        d.text((62, y), k, font=font(13), fill=(150, 175, 225))
        d.text((62, y + 17), v, font=font(14, True), fill=(240, 244, 252))
        y += 52

    # --- Legend (bottom left) ---
    d.rounded_rectangle([24, 556, 228, 692], radius=6, fill=(6, 10, 20, 215), outline=(40, 52, 80))
    ly = 568
    rows = [("dot_w", "Starlink Satellite (active)"), ("dot_b", "Starlink Satellite (inactive)"),
            ("line", "Orbital Path"), ("dash", "Moon Orbit (approx.)"), ("earth", "Earth")]
    for kind, label in rows:
        if kind == "dot_w":
            d.ellipse([36, ly + 3, 44, ly + 11], fill=(255, 255, 255))
        elif kind == "dot_b":
            d.ellipse([36, ly + 3, 44, ly + 11], fill=(40, 120, 255))
        elif kind == "line":
            d.line([(34, ly + 7), (50, ly + 7)], fill=(50, 120, 245), width=2)
        elif kind == "dash":
            dashed_curve(d, [(34, ly + 7), (52, ly + 7)], 4, 3, (130, 145, 170))
        else:
            d.ellipse([34, ly, 50, ly + 16], fill=(30, 90, 180), outline=(90, 160, 240))
        d.text((62, ly), label, font=font(11), fill=(225, 232, 245))
        ly += 25

    # --- Live panel frame (bottom right); values drawn per frame ---
    d.rounded_rectangle([PANEL[0], PANEL[1], PANEL[2], PANEL[3]], radius=6, fill=(6, 10, 20, 225), outline=(60, 72, 100))
    d.line([(PANEL[0], PANEL[1] + 36), (PANEL[2], PANEL[1] + 36)], fill=(50, 60, 85))
    d.ellipse([PANEL[0] + 14, PANEL[1] + 11, PANEL[0] + 28, PANEL[1] + 25], fill=(235, 40, 40))
    d.text((PANEL[0] + 36, PANEL[1] + 8), "LIVE", font=font(16, True), fill=(235, 50, 50))
    d.text((PANEL[0] + 14, PANEL[1] + 46), "Satellites in View", font=font(12), fill=(225, 232, 245))
    d.text((PANEL[0] + 14, PANEL[1] + 72), "Orbital Speed (avg.)", font=font(12), fill=(225, 232, 245))
    d.text((PANEL[0] + 14, PANEL[1] + 98), "Altitude Range", font=font(12), fill=(225, 232, 245))
    d.text((PANEL[2] - 14, PANEL[1] + 98), "~ 540 – 570 km", font=font(12, True), fill=(245, 248, 255), anchor="ra")
    return np.array(bg.convert("RGB"), np.uint8)


PANEL = (955, 548, 1252, 668)
BG = build_background()

F12, F12B, F11 = font(12), font(12, True), font(11)


def draw_live(frame, n_view, now):
    x0, y0, x1, y1 = PANEL
    reg = Image.fromarray(frame[y0 + 2:y0 + 36, x0 + 80:x1 - 2].copy())
    d = ImageDraw.Draw(reg)
    d.rectangle([0, 0, reg.width, reg.height], fill=(6, 10, 20))
    ts = now.strftime("UTC   %Y-%m-%d %H:%M:%S")
    d.text((reg.width - 8, 17), ts, font=F11, fill=(205, 215, 235), anchor="rm")
    frame[y0 + 2:y0 + 36, x0 + 80:x1 - 2] = np.array(reg)
    # values
    reg2 = Image.fromarray(frame[y0 + 42:y0 + 96, x0 + 150:x1 - 4].copy())
    d = ImageDraw.Draw(reg2)
    d.rectangle([0, 0, reg2.width, reg2.height], fill=(6, 10, 20))
    d.text((reg2.width - 8, 12), f"~ {n_view:,}", font=F12B, fill=(245, 248, 255), anchor="rm")
    d.text((reg2.width - 8, 38), "~ 7.6 km/s", font=F12B, fill=(245, 248, 255), anchor="rm")
    frame[y0 + 42:y0 + 96, x0 + 150:x1 - 4] = np.array(reg2)


# ----------------------------------------------------------------------------
# Per-frame render
# ----------------------------------------------------------------------------
PS = int(1.6 * R) * 2          # constellation patch size (matches LINES)
PX0, PY0 = CX - PS // 2, CY - PS // 2
EX0, EY0 = CX - R, CY - R
WHITE = np.array([255, 255, 255], np.float32)
BLUE = np.array([60, 140, 255], np.float32)


def render_frame(t, now):
    frame = BG.copy()
    theta = math.radians(SPIN_DEG_PER_SEC) * t

    # Earth
    reg = frame[EY0:EY0 + EG["size"], EX0:EX0 + EG["size"]]
    reg[EG["mask"]] = render_earth_masked(theta)

    # Patch (orbit lines + satellites) added additively
    patch = LINES.copy()
    x, y, z = sat_positions(t)
    sx = np.rint(PS / 2 + x * R).astype(np.int32)
    sy = np.rint(PS / 2 - z * R).astype(np.int32)
    behind = (y > 0)
    occl = behind & (x * x + z * z < 1.0)
    vis = (~occl) & (sx > 1) & (sx < PS - 2) & (sy > 1) & (sy < PS - 2)
    n_view = int(vis.sum())
    act = SATS["active"]
    bright = np.where(behind, 0.55, 1.0).astype(np.float32)

    idx = sy * PS + sx
    cw = np.bincount(idx[vis & act], weights=bright[vis & act], minlength=PS * PS).reshape(PS, PS).astype(np.float32)
    cb = np.bincount(idx[vis & ~act], weights=bright[vis & ~act], minlength=PS * PS).reshape(PS, PS).astype(np.float32)
    for cnt, colour in ((cw, WHITE), (cb, BLUE)):
        g = cnt.copy()
        g[1:] += 0.28 * cnt[:-1]; g[:-1] += 0.28 * cnt[1:]
        g[:, 1:] += 0.28 * cnt[:, :-1]; g[:, :-1] += 0.28 * cnt[:, 1:]
        np.minimum(g, 1.2, out=g)
        patch += g[..., None] * colour

    pr = frame[PY0:PY0 + PS, PX0:PX0 + PS].astype(np.float32) + patch
    frame[PY0:PY0 + PS, PX0:PX0 + PS] = np.clip(pr, 0, 255).astype(np.uint8)

    draw_live(frame, n_view, now)
    return frame


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def ffmpeg_cmd(args):
    base = ["ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-re", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if args.rtmp:
        base += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
        return base + ["-c:v", "libx264", "-preset", os.environ.get("X264_PRESET", "veryfast"),
                       "-pix_fmt", "yuv420p", "-b:v", "3000k", "-maxrate", "3500k", "-bufsize", "7000k",
                       "-g", str(FPS * 2), "-keyint_min", str(FPS * 2), "-sc_threshold", "0",
                       "-c:a", "aac", "-b:a", "96k", "-ar", "44100",
                       "-f", "flv", args.rtmp]
    base.remove("-re")
    return base + ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-y", args.out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frame")
    ap.add_argument("--frames", type=int, default=0, help="0 = run forever")
    ap.add_argument("--out", default="starlink_test.mp4")
    ap.add_argument("--rtmp", default=os.environ.get("RTMP_URL"))
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (live mode)")
    args = ap.parse_args()

    if args.test_frame:
        f = render_frame(120.0, dt.datetime.now(dt.timezone.utc))
        Image.fromarray(f).save(args.test_frame)
        print("saved", args.test_frame)
        return

    proc = subprocess.Popen(ffmpeg_cmd(args), stdin=subprocess.PIPE)
    start = time.time()
    n = 0
    try:
        while True:
            if args.frames and n >= args.frames:
                break
            if args.seconds and time.time() - start > args.seconds:
                break
            f = render_frame(n / FPS, dt.datetime.now(dt.timezone.utc))
            proc.stdin.write(f.tobytes())
            n += 1
            if n % (FPS * 10) == 0:
                print(f"{n} frames, {n / (time.time() - start):.1f} fps avg", flush=True)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main() or 0)
