from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def save_fig(
    fig: plt.Figure,
    name: str,
    figs_dir: str | Path = "figs",
) -> Path:
    """
    Save a matplotlib figure as an SVG file in figs_dir.

    Text is kept as editable text elements (svg.fonttype='none') rather than
    being converted to paths, so labels and titles can be edited directly in
    Inkscape.  The figs_dir is created if it does not already exist.

    Params:
        fig: plt.Figure : figure to save
        name: str : output filename without extension
        figs_dir: str | Path : directory to write into (default 'figs/')
    Returns:
        Path : path to the written .svg file
    """
    out_dir = Path(figs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.svg"
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig.savefig(out_path, format="svg", bbox_inches="tight")
    return out_path
