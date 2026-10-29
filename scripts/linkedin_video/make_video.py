"""LinkedIn video for TLEIO, rendered frame by frame:
title (staggered entrances) -> pipeline demo on a card, whose two panels then shrink into the method diagram
(events at its input, trajectory at its output) and keep playing while it builds -> closing card.
Sections overlap by XF seconds and dissolve into each other with a slight zoom; each drifts forward slowly.

  .venv/bin/python scripts/linkedin_video/make_video.py            # full res to ~/tleio_linkedin_renders, 480p preview to outputs/linkedin
  .venv/bin/python scripts/linkedin_video/make_video.py --final    # also copy the full-res video to outputs/linkedin
"""
from pathlib import Path
import hashlib, inspect, json, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import imageio.v2 as iio
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; ASSETS = ROOT/'website/assets'
OUT = ROOT/'outputs/linkedin'; OUT.mkdir(parents=True, exist_ok=True)   # git-ignored
RENDERS = Path.home()/'tleio_linkedin_renders'; BUILD = RENDERS/'build'; BUILD.mkdir(parents=True, exist_ok=True)   # kept off the output folder
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 30
INK, MUTED, ACCENT = (38, 38, 38, 255), (107, 107, 107, 255), (33, 102, 172, 255)
FONTS = Path('/usr/share/fonts/truetype/noto')
DEMO_LEN = 23.6                    # the page teaser is a 37 s replay of an 11.8 s sequence; shown at 2x slow motion
DEMO_SPEEDUP = 37.01/DEMO_LEN
XF = 1.3                           # section overlap: outgoing fades over its first 60%, incoming over its last 60%
PACE = 1.25                        # stretches the title and closing animations (not the demo playback)
SECTIONS = [('title', 7.7*PACE), ('live', DEMO_LEN - .2), ('end', 4.2*PACE)]   # live ends while the demo is still playing


# ---------- helpers ----------
def font(size, bold=False): return ImageFont.truetype(str(FONTS/('NotoSans-Bold.ttf' if bold else 'NotoSans-Regular.ttf')), size)
def clamp(x): return max(0., min(1., x))
def ease_out(x): x = clamp(x); return 1 - (1 - x)**3
def ease_io(x): x = clamp(x); return 4*x**3 if x < .5 else 1 - (-2*x + 2)**3/2


def layer(): return Image.new('RGBA', (W, H), (0, 0, 0, 0))


def text_layer(text, f, fill, y):
    """A full-frame RGBA layer holding one centered line of text."""
    im = layer(); d = ImageDraw.Draw(im); w = d.textlength(text, font=f)
    d.text(((W - w)/2, y), text, font=f, fill=fill); return im


def fit(img, box_w, box_h):
    s = min(box_w/img.width, box_h/img.height); return img.resize((round(img.width*s), round(img.height*s)), Image.LANCZOS)


def rounded_mask(w, h, radius=22):
    m = Image.new('L', (w, h), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius, fill=255); return m


def place_card(base, x, y, w, h, radius=22, blur=28, offset=12, alpha=60):
    """Draw a white rounded card with a soft drop shadow onto base."""
    pad = blur*3; sh = Image.new('L', (w + 2*pad, h + 2*pad), 0)
    ImageDraw.Draw(sh).rounded_rectangle([pad, pad + offset, pad + w, pad + h + offset], radius, fill=alpha)
    base.paste((20, 40, 70, 255), (x - pad, y - pad), sh.filter(ImageFilter.GaussianBlur(blur)))
    base.paste((255, 255, 255, 255), (x, y), rounded_mask(w, h, radius))


def with_alpha(im, a):
    if a >= 1: return im
    im = im.copy(); im.putalpha(im.getchannel('A').point(lambda v: int(v*a))); return im


def transform(im, scale, dy=0.):
    """Scale about the frame centre (and shift vertically) in one resample."""
    if abs(scale - 1) < 1e-4 and abs(dy) < .01: return im
    cx, cy = W/2, H/2; s = 1/scale
    return im.transform((W, H), Image.AFFINE, (s, 0, cx - s*cx, 0, s, cy - s*(cy + dy)), resample=Image.BICUBIC)


def entrance(im, t, start, dur=.7, rise=28):
    """Fade in while rising into place; None before it starts."""
    k = ease_out((t - start)/dur)
    return None if k <= 0 else with_alpha(transform(im, 1, rise*(1 - k)), k)


def compose(out, items, t, **kw):
    for im, start in items:
        e = entrance(im, t, start, **kw)
        if e is not None: out.alpha_composite(e)


def background():
    """Soft vertical wash, like the page header."""
    g = np.linspace(0, 1, H)[:, None]
    rgb = np.array([238, 242, 247])*(1 - g) + np.array([250, 251, 252])*g
    return Image.fromarray(np.broadcast_to(rgb[:, None, :], (H, W, 3)).astype(np.uint8), 'RGB').convert('RGBA')


def tag_layer():
    """Small persistent wordmark, bottom left."""
    im = layer(); d = ImageDraw.Draw(im); x, y = 64, 1012
    d.text((x, y), 'TLEIO', font=font(24, True), fill=INK); x += d.textlength('TLEIO', font=font(24, True)) + 14
    d.text((x, y + 1), 'Advances in Event-Based Computer Vision @ ACCV 2026', font=font(22), fill=MUTED); return im


def logos_layer(y, eth_h, goo_h, gap, tags=True):
    im = layer(); d = ImageDraw.Draw(im); tag = font(30); tw = 26 if tags else 0
    eth = fit(Image.open(HERE/'assets/eth.png').convert('RGBA'), 600, eth_h)
    goo = fit(Image.open(HERE/'assets/google.png').convert('RGBA'), 600, goo_h)
    x = (W - (tw + eth.width + gap + tw + goo.width))//2
    if tags: d.text((x, y - 12), '1', font=tag, fill=INK)
    x += tw; im.alpha_composite(eth, (x, y + (goo_h - eth.height)//2)); x += eth.width + gap
    if tags: d.text((x, y - 12), '2', font=tag, fill=INK)
    x += tw; im.alpha_composite(goo, (x, y)); return im


def eth_layer(y, h):
    """The ETH Zurich logo alone, centred."""
    im = layer(); eth = fit(Image.open(HERE/'assets/eth.png').convert('RGBA'), 600, h); im.alpha_composite(eth, ((W - eth.width)//2, y)); return im


# ---------- sections: each returns its content layer at local time t ----------
class Title:
    def __init__(self):
        name, sup = font(36), font(22)
        self.head = [(text_layer('TLEIO: Tight Learned', font(88, True), INK, 150), .15),
                     (text_layer('Event-Inertial Odometry', font(88, True), INK, 258), .15)]
        rule = layer(); ImageDraw.Draw(rule).rounded_rectangle([W/2 - 60, 395, W/2 + 60, 399], 2, fill=INK)
        self.items = [(rule, 2.2), (text_layer('Advances in Event-Based Computer Vision @ ACCV 2026', font(40, True), ACCENT, 432), 2.5)]
        rows = [[('Matteo Cascone', '1*'), ('David Jensen', '1*'), ('Alessandro Pirini', '1*'), ('Angelo Stefanini', '1*')],
                [('Pietro Venè', '1*'), ('Marta Tintoré Gazulla', '2'), ('Javier Hidalgo Carriò', '2')]]
        for k, row in enumerate(rows):
            im = layer(); d = ImageDraw.Draw(im); runs = []
            for i, (n, s) in enumerate(row): runs += [(n, name, 0), (s, sup, 12)] + ([('      ', name, 0)] if i < len(row) - 1 else [])
            x = (W - sum(d.textlength(t, font=f) for t, f, _ in runs))/2
            for t, f, r in runs: d.text((x, 530 + k*60 - r), t, font=f, fill=INK); x += d.textlength(t, font=f)
            self.items.append((im, 2.9))
        self.items += [(logos_layer(745, 70, 92, 130), 3.6), (text_layer('* Equal contribution', font(26), MUTED, 900), 4.0)]

    def __call__(self, t):
        t /= PACE; head = layer(); compose(head, self.head, t)
        # the title starts alone at the centre of the frame, then glides up to make room for the rest
        out = transform(head, 1, 255*(1 - ease_io((t - 1.7)/.8))); compose(out, self.items, t)
        return out


class LivePipeline:
    """The demo on a card; then its two panels shrink into the method diagram (events into the Event Camera slot,
    trajectory after the filter output) and keep playing while the diagram builds around them."""
    MORPH_AT, MORPH = 10.5, 2.1
    RGB = ROOT/'data/tartanair/competition/rgb/mono/MH001'     # the demo's sequence (TartanAir competition MH001), 30 Hz
    RGB_LAG = -1                                                # frame offset that best aligns RGB edges with the events
    SWAP_AT, SWAP_DUR = 5.0, 1.4                                # events | RGB side by side; then the RGB leaves, the events grow
                                                                # into place and the trajectory comes up on the right
    HEADER = (0, 50, 1280, 89)                                  # the demo's own panel headings, replaced by our labels
    TR_LABEL_X = 995                                            # centre of the demo's 'Estimated 3D trajectory' heading
    EV_SRC, TR_SRC = (8, 89, 670, 503), (745, 92, 530, 508)    # panels in the 1280x606 demo frame
    TR_PLOT = (765, 168, 510, 432)                              # trajectory crop once landed: the plot, without info box and legend,
                                                                # at the events' 4:3 so both landed windows are the same size
    BOX = (227, 237, 255, 255)                                  # the diagram's light-blue box fill
    EV_CUT, ARROW_Y, OUT_Y = 638, 370, 1207                      # 3400 px diagram: cut right after its Event Camera box (keeping the
                                                                # arrow out of it), that arrow's y, y of the Update -> outputs arrow
    DS = .34                                                    # diagram scale; the two live windows share the remaining width

    def __init__(self):
        clip = BUILD/'demo_rt.mp4'
        subprocess.run([FF, '-v', 'error', '-y', '-i', ASSETS/'tleio-demo.mp4', '-an', '-vf',
                        f'crop=1280:606:0:57,setpts=PTS/{DEMO_SPEEDUP},fps={FPS}', '-c:v', 'libx264', '-crf', '14', clip], check=True)
        self.n = iio.get_reader(clip).count_frames(); self.reader = iio.get_reader(clip); self.idx, self.frame = -1, None
        # demo card
        self.cw = 1600; s = self.cw/1280; top = 38                         # the card starts below the demo's (removed) title line
        self.top = top; self.ch = round((606 - top)*s); self.cx, self.cy = (W - self.cw)//2, 135; oy = self.cy - top*s   # oy: demo-frame origin
        pad = 28; self.pad = pad                                       # white margin between the demo content and the card edge
        self.demo_chrome = layer(); place_card(self.demo_chrome, self.cx - pad, self.cy - pad, self.cw + 2*pad, self.ch + 2*pad)
        self.mask = rounded_mask(self.cw, self.ch, 8)
        self.label_mid = (self.cy - pad + oy + self.EV_SRC[1]*s)/2       # labels centred between card edge and panel top
        self.caption = layer(); d = ImageDraw.Draw(self.caption); f34 = font(34); a, b, gap, arrow = 'Events + IMU', '6-DoF trajectory', 28, 90
        x = (W - (d.textlength(a, font=f34) + 2*gap + arrow + d.textlength(b, font=f34)))/2; y = self.cy + self.ch + pad + 36
        top, bottom = d.textbbox((x, y), 'E', font=f34)[1::2]; ay = (top + bottom)/2            # input -> output
        d.text((x, y), a, font=f34, fill=INK); x += d.textlength(a, font=f34) + gap
        d.line([(x, ay), (x + arrow - 14, ay)], fill=INK, width=3); d.polygon([(x + arrow, ay), (x + arrow - 16, ay - 9), (x + arrow - 16, ay + 9)], fill=INK)
        d.text((x + arrow + gap, y), b, font=f34, fill=INK)
        # method card: the diagram, then an arrow from its output bracket to the live trajectory panel
        png = BUILD/'model_arch'
        subprocess.run(['pdftocairo', '-png', '-singlefile', '-scale-to-x', '3400', '-scale-to-y', '-1', HERE/'assets/model_arch.pdf', png], check=True)
        arch = Image.open(f'{png}.png').convert('RGBA'); arch = arch.crop((self.EV_CUT, 0, arch.width, arch.height))
        ds = self.DS; arch = arch.resize((round(arch.width*ds), round(arch.height*ds)), Image.LANCZOS)
        inset, card_w = 40, 1800; x0, y0 = (W - card_w)//2 + inset, 240 + inset; strip_w = card_w - 2*inset
        tw = (strip_w - arch.width - 144)//2; th = round(tw*self.EV_SRC[3]/self.EV_SRC[2]); ew = tw + 24   # event box: same image size plus padding
        tx, ty = x0 + ew + arch.width + 112, y0 + round(self.OUT_Y*ds) - th//2
        self.method_chrome = layer(); place_card(self.method_chrome, x0 - inset, y0 - inset, card_w, arch.height + 2*inset)
        strip = Image.new('RGBA', (strip_w, arch.height), (0, 0, 0, 0)); strip.alpha_composite(arch, (ew, 0)); d = ImageDraw.Draw(strip)
        lf = font(28, True); self.boxes = layer(); bd = ImageDraw.Draw(self.boxes)   # the two live boxes fade in on their own

        def box(x, y, w, h, label):                     # a light-blue box with a bold label, as in the diagram (strip coordinates)
            x, y = x + x0, y + y0; bd.rounded_rectangle([x, y, x + w, y + h], 6, fill=self.BOX)
            bd.text((x + (w - bd.textlength(label, font=lf))/2, y + 6), label, font=lf, fill=(0, 0, 0, 255))
        # a larger Event Camera box in place of the diagram's own, feeding the diagram's arrow
        pad, label_h = 12, 48; ew_img = ew - 2*pad; eh_img = round(ew_img*self.EV_SRC[3]/self.EV_SRC[2]); eb_h = label_h + eh_img + pad
        eb_top = max(0, round(self.ARROW_Y*ds) - round(.4*eb_h)); box(0, eb_top, ew, eb_h, 'Event Camera')
        box(tx - x0 - 8, ty - y0 - 50, tw + 16, th + 58, 'Trajectory')
        # arrow to the trajectory: the diagram's own Update -> outputs arrow (3400 px coordinates), lengthened, at the same height
        src = Image.open(f'{png}.png').convert('RGBA'); head, shaft = src.crop((3120, 1188, 3172, 1227)), src.crop((3040, 1188, 3041, 1227))
        a0, a1 = ew + round((3356 - self.EV_CUT)*ds) + 14, tx - x0 - 8 - 14; L = round((a1 - a0)/ds)
        arrow = Image.new('RGBA', (L, head.height)); arrow.paste(shaft.resize((L - head.width, head.height)), (0, 0)); arrow.paste(head, (L - head.width, 0))
        arrow = arrow.resize((a1 - a0, round(head.height*ds)), Image.LANCZOS); strip.alpha_composite(arrow, (a0, round(self.OUT_Y*ds) - arrow.height//2))
        self.strip, self.strip_xy, self.strip_a, self.ew = strip, (x0, y0), np.asarray(strip.getchannel('A'), np.float32), ew
        self.heading = text_layer('How TLEIO works: EventsFormer + SCEKF', font(54, True), INK, 75); self.tag = tag_layer()
        self.takeaway = text_layer('Learned motion from events + filter-based fusion', font(34), INK, y0 + arch.height + inset + 42)
        # panel flights: (rect on the demo card, rect in the diagram) and the source crop at each end
        self.ev = ((self.cx + 8*s, oy + 89*s, 670*s, 503*s), (x0 + pad, y0 + eb_top + label_h, ew_img, eh_img))
        self.ev_src = (self.EV_SRC, self.EV_SRC)
        self.tr = ((self.cx + 745*s, oy + 92*s, 530*s, 508*s), (tx + (tw - round(th*self.TR_PLOT[2]/self.TR_PLOT[3]))//2, ty, round(th*self.TR_PLOT[2]/self.TR_PLOT[3]), th))
        self.tr_src = (self.TR_SRC, self.TR_PLOT)                     # whole plot, fitted by height into the same-size box
        fade = lambda n: np.linspace(1, 0, n)[:, None]**1.5
        self.fade_b, self.fade_r = fade(34)[:, :, None], fade(22).T[:, :, None]   # the source plot is clipped at these edges
        lf2 = font(26, True); self.lbl = {}
        for text in ('RGB camera', 'Event camera', 'Estimated 3D trajectory'):
            d = ImageDraw.Draw(Image.new('RGBA', (1, 1))); w = round(d.textlength(text, font=lf2))
            im = Image.new('RGBA', (w + 4, 40), (0, 0, 0, 0)); ImageDraw.Draw(im).text((2, 0), text, font=lf2, fill=INK)
            top, bottom = d.textbbox((2, 0), text, font=lf2)[1::2]; self.lbl[text] = (im, (top + bottom)/2)
        self.rgb_i, self.rgb = -1, None

    def rgb_at(self, j):
        i = min(max(round(j*11.8/DEMO_LEN) + self.RGB_LAG, 0), 356)
        if i != self.rgb_i: self.rgb_i, self.rgb = i, np.asarray(Image.open(self.RGB/f'{i:06d}.png').convert('RGB'), np.float32)
        return self.rgb

    def soften(self, f):
        """Fade the trajectory plot to white at the bottom and right, where the source render clips its axis labels."""
        f = f.astype(np.float32); b, rr = self.fade_b, self.fade_r
        f[-len(b):, 700:] = 255 - (255 - f[-len(b):, 700:])*b; f[:, -rr.shape[1]:] = 255 - (255 - f[:, -rr.shape[1]:])*rr
        return f.astype(np.uint8)

    def stage(self, f, t):
        """Demo-frame content for the card, and its panel labels as (text, centre x, alpha) in demo-frame pixels."""
        p = ease_io((t - self.SWAP_AT)/self.SWAP_DUR); x, y, w, h = self.EV_SRC
        g = f.copy(); x0, y0, x1, y1 = self.HEADER; g[y0:y1, x0:x1] = 255; g[:45] = 255   # drop the demo's own title line
        if p >= 1: return g, [('Event camera', x + w/2, 1.), ('Estimated 3D trajectory', self.TR_LABEL_X, 1.)]
        # side by side, events and RGB are both full size: the card is wider, then narrows back as the RGB leaves
        # and the trajectory comes up, while the events stay put
        ev = f[y:y + h, x:x + w]; g[40:, :x + w + 10] = 255          # left half rebuilt below; the title row stays
        p_rgb, p_tr = ease_io(p/.6), ease_io((p - .45)/.55)          # the RGB leaves first, then the trajectory comes up
        gap = 20; rgb_x = x + w + gap; wide = rgb_x + w + x; wf = round(wide + (1280 - wide)*p)
        c = np.full((606, wf, 3), 255, np.float32); c[40:, :1280] = 255*(1 - p_tr) + g[40:].astype(np.float32)*p_tr
        c[y:y + h, x:x + w] = ev
        rgb = np.asarray(Image.fromarray(self.rgb_at(int(t*FPS)).astype(np.uint8)).resize((w, h), Image.LANCZOS), np.float32)
        vis = min(w, wf - rgb_x)
        if vis > 0: c[y:y + h, rgb_x:rgb_x + vis] = c[y:y + h, rgb_x:rgb_x + vis]*p_rgb + rgb[:, :vis]*(1 - p_rgb)
        return c.astype(np.uint8), [('RGB camera', rgb_x + w/2, 1 - p_rgb), ('Event camera', x + w/2, 1.),
                                    ('Estimated 3D trajectory', self.TR_LABEL_X, p_tr)]

    def frame_at(self, i):
        """Frames are only ever requested in order, so read sequentially."""
        i = min(i, self.n - 1)
        while self.idx < i: self.frame = self.reader.get_next_data(); self.idx += 1
        return self.frame

    @staticmethod
    def lerp(a, b, m): return tuple(p + (q - p)*m for p, q in zip(a, b))

    def panel(self, out, f, rects, srcs, m):
        x, y, w, h = self.lerp(*rects, m); sx, sy, sw, sh = self.lerp(*srcs, m); w, h = max(2, round(w)), max(2, round(h))
        im = Image.fromarray(f[round(sy):round(sy + sh), round(sx):round(sx + sw)]).resize((w, h), Image.LANCZOS)
        out.paste(im, (round(x), round(y)), rounded_mask(w, h, round(6 + 10*(1 - m))))

    def __call__(self, t):
        f = self.soften(self.frame_at(int(t*FPS))); m = ease_io((t - self.MORPH_AT)/self.MORPH); out = layer()
        if m > 0: out.alpha_composite(with_alpha(self.method_chrome, ease_io((m - .3)/.7)))
        if m < 1:
            fc, labels = self.stage(f, t); s = self.cw/1280; cw = round(fc.shape[1]*s); cx = (W - cw)//2
            if cw == self.cw: card = self.demo_chrome.copy()
            else: card = layer(); place_card(card, cx - self.pad, self.cy - self.pad, cw + 2*self.pad, self.ch + 2*self.pad)
            card.paste(Image.fromarray(fc[self.top:]).resize((cw, self.ch), Image.LANCZOS), (cx, self.cy), rounded_mask(cw, self.ch, 8))
            for text, lx, a in labels:
                if a > 0: im, mid = self.lbl[text]; card.alpha_composite(with_alpha(im, a), (round(cx + lx*s - im.width/2), round(self.label_mid - mid)))
            if m > 0:                                   # the panels are flying on their own now: blank their spots on the card
                d = ImageDraw.Draw(card)
                for x, y, w, h in (self.ev[0], self.tr[0]): d.rectangle([x, y, x + w, y + h], fill='white')
            out.alpha_composite(with_alpha(card, 1 - ease_io(m/.4)))   # card leaves early: mid-flight only the panels show
            cap = entrance(self.caption, t, .9, rise=16)
            if cap is not None: out.alpha_composite(with_alpha(cap, 1 - ease_io((t - self.MORPH_AT + .2)/.5)))
        if m > 0:
            out.alpha_composite(with_alpha(self.boxes, ease_io((m - .15)/.85)))   # boxes fade in around the shrinking panels
            k = ease_io((t - self.MORPH_AT - 1.2)/2.9)      # then the pipeline between them builds left to right
            if k > 0:
                w, e = self.strip.width, self.ew; ramp = np.clip(((w - e + 120)*k - (np.arange(w) - e))/120, 0, 1)
                strip = self.strip.copy(); strip.putalpha(Image.fromarray((self.strip_a*ramp[None, :]).astype(np.uint8), 'L'))
                out.alpha_composite(strip, self.strip_xy)
            self.panel(out, f, self.ev, self.ev_src, m); self.panel(out, f, self.tr, self.tr_src, m)
            h = entrance(self.heading, t, self.MORPH_AT + 1.4, dur=.9)
            if h is not None: out.alpha_composite(h)
            k2 = entrance(self.takeaway, t, self.MORPH_AT + 3.6, dur=.9, rise=16)   # once the diagram has built
            if k2 is not None: out.alpha_composite(k2)
        tag = entrance(self.tag, t, 1.2, rise=16)
        if tag is not None: out.alpha_composite(tag)
        return out


class End:
    def __init__(self):
        self.items = [(text_layer('TLEIO', font(120, True), INK, 250), .1),
                      (text_layer('Tight Learned Event-Inertial Odometry', font(44), INK, 420), .3),
                      (text_layer('Advances in Event-Based Computer Vision @ ACCV 2026', font(40, True), ACCENT, 560), .6),
                      (eth_layer(800, 58), 1.0)]

    def __call__(self, t):
        out = layer(); compose(out, self.items, t/PACE); return out


# ---------- timeline ----------
MAKERS = {'title': Title, 'live': LivePipeline, 'end': End}
CHUNKS = RENDERS/'chunks'


def fingerprints():
    """One hash per section class, and one for everything else in this file (helpers, timing, compositing)."""
    shared, per = Path(__file__).read_text(), {}
    for name, cls in MAKERS.items():
        src = inspect.getsource(cls); per[name] = hashlib.sha1(src.encode()).hexdigest(); shared = shared.replace(src, '')
    return hashlib.sha1(shared.encode()).hexdigest(), per


def main():
    """The timeline is cut at every section start and end; each piece is cached under a hash of the code it depends on,
    so an edit re-renders only the pieces it touches, and the pieces are joined without re-encoding."""
    shared, per = fingerprints(); CHUNKS.mkdir(parents=True, exist_ok=True)
    secs, start = [], 0.
    for name, dur in SECTIONS: secs.append((name, start, dur)); start += dur - XF
    total = start + XF; n = round(total*FPS); bg = background(); white = Image.new('RGBA', (W, H), 'white')
    cuts = sorted({0, n} | {min(n, round(v*FPS)) for _, s0, dur in secs for v in (s0, s0 + dur)})
    made, parts = {}, []
    for i0, i1 in zip(cuts, cuts[1:]):
        active = [k for k, (_, s0, dur) in enumerate(secs) if s0*FPS <= i1 - 1 and (s0 + dur)*FPS >= i0]
        key = hashlib.sha1(json.dumps([shared, i0, i1, [(k, *secs[k], per[secs[k][0]]) for k in active]]).encode()).hexdigest()[:12]
        part = CHUNKS/f'{i0:05d}-{i1:05d}_{key}.mp4'; parts.append(part)
        if part.exists(): print(f'{i0/FPS:5.1f}-{i1/FPS:5.1f} s  cached', flush=True); continue
        print(f'{i0/FPS:5.1f}-{i1/FPS:5.1f} s  rendering', flush=True)
        for k in active:
            if k not in made: made[k] = MAKERS[secs[k][0]]()
        writer = iio.get_writer(str(part), fps=FPS, codec='libx264', quality=None, macro_block_size=None, pixelformat='yuv420p',
                                ffmpeg_params=['-crf', '18', '-preset', 'slow'])
        for i in range(i0, i1):
            T = i/FPS; frame = bg.copy()
            for k in active:
                _, s0, dur = secs[k]; t = T - s0
                if t < 0 or t > dur: continue
                a_in = 1. if k == 0 else ease_io((t/XF - .4)/.6)
                a_out = 1. if k == len(secs) - 1 else ease_io(((dur - t)/XF - .4)/.6)
                if min(a_in, a_out) <= 0: continue
                # slow push-in over the section; incoming content settles from slightly small, outgoing grows slightly
                scale = 1 + .025*t/dur - .03*(1 - a_in) + .03*(1 - a_out)
                frame.alpha_composite(with_alpha(transform(made[k](t), scale), min(a_in, a_out)))
            if T < .5: frame = Image.blend(white, frame, ease_io(T/.5))                       # open from white
            writer.append_data(np.asarray(frame.convert('RGB')))
        writer.close()
    for old in CHUNKS.glob('*.mp4'):
        if old not in parts: old.unlink()
    listing = CHUNKS/'list.txt'; listing.write_text(''.join(f"file '{p}'\n" for p in parts))
    out = RENDERS/'tleio_linkedin.mp4'
    subprocess.run([FF, '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', listing, '-c', 'copy', '-movflags', '+faststart', out], check=True)
    print(f'{out}  ({total:.1f} s)')
    preview = OUT/'tleio_linkedin_preview.mp4'                       # small enough to stream over a hotspot
    subprocess.run([FF, '-v', 'error', '-y', '-i', out, '-vf', 'scale=854:480:flags=lanczos', '-c:v', 'libx264', '-crf', '30',
                    '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', preview], check=True)
    print(f'{preview}  ({preview.stat().st_size/1e6:.1f} MB)')
    if '--final' in sys.argv: shutil.copy(out, OUT/'tleio_linkedin.mp4'); print(OUT/'tleio_linkedin.mp4')


if __name__ == '__main__': main()
