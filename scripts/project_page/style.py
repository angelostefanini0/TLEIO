"""Shared figure style for the project page: one palette and typeface (Noto Sans, as on the page)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

BLUE, GT, RED, INK, MUTED, GRID = '#2166ac', '#5a5a5a', '#c0392b', '#2b2b2b', '#777777', '#e6e6e6'
BASELINES = ['#9aa5b1', '#bcc4cc', '#d8dde2']          # neutral greys, darkest = strongest baseline
POS, NEG = (33, 102, 172), (192, 57, 43)               # event polarity, same hues as BLUE / RED
AXIS_RGB = ('#c0392b', '#2e8b57', '#2166ac')           # body frame x, y, z

for f in ['NotoSans-Regular.ttf', 'NotoSans-Bold.ttf', 'NotoSansMono-Regular.ttf']:
    p = Path('/usr/share/fonts/truetype/noto')/f
    if p.exists(): font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family': ['Noto Sans', 'DejaVu Sans'], 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#b5b5b5', 'axes.labelcolor': '#444',
                     'xtick.color': '#555', 'ytick.color': '#555', 'axes.titlesize': 11, 'axes.titleweight': 'normal',
                     'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'legend.frameon': False})


def save(fig, path, svg=True):
    """Save PNG (and SVG with text kept as text, trailing whitespace stripped)."""
    path = Path(path)
    if svg:
        fig.savefig(path.with_suffix('.svg'))
        p = path.with_suffix('.svg'); p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines()) + '\n')
    fig.savefig(path.with_suffix('.png'), dpi=180); plt.close(fig)
