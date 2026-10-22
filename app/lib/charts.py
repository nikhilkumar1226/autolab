"""
The discovery chart (Altair): measurements, the GP's mean and ±2σ band, the best model's curve,
and rejected or flagged requests. Colours: GP blue, best model orange (validated categorical
slots 1 and 2), measurements in ink, rejected or flagged in the status red, each also shown with
its own mark and named in the legend.
"""
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd

GP, MODEL, INK, FLAG, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#d03b3b", "#898781", "#e1e0d9"


def discovery_chart(points: List[Dict[str, Any]], gp: Optional[Dict[str, List[float]]], curve: pd.DataFrame,
                    flagged: List[Dict[str, Any]], x_title: str, y_title: str, x_domain: List[float],
                    height: int = 380) -> alt.LayerChart:
    x = alt.X("x:Q", title=x_title, scale=alt.Scale(domain=x_domain, nice=False))
    layers = []
    legend = alt.Legend(title=None, orient="top")
    series = alt.Color("series:N", scale=alt.Scale(
        domain=["GP mean ± 2σ", "Best model", "Measured", "Rejected or flagged"], range=[GP, MODEL, INK, FLAG]), legend=legend)
    if gp:
        band = pd.DataFrame({"x": gp["x"], "mean": gp["mean"], "std": gp["std"]})
        band["lo"], band["hi"], band["series"] = band["mean"] - 2 * band["std"], band["mean"] + 2 * band["std"], "GP mean ± 2σ"
        layers.append(alt.Chart(band).mark_area(opacity=0.15, color=GP).encode(x=x, y="lo:Q", y2="hi:Q"))
        layers.append(alt.Chart(band).mark_line(strokeWidth=2).encode(
            x=x, y=alt.Y("mean:Q", title=y_title), color=series,
            tooltip=[alt.Tooltip("x:Q", format=".4g"), alt.Tooltip("mean:Q", title="GP mean", format=".4g"),
                     alt.Tooltip("std:Q", title="GP σ", format=".3g")]))
    if not curve.empty:
        c = curve.assign(series="Best model")
        layers.append(alt.Chart(c).mark_line(strokeWidth=2, strokeDash=[6, 3]).encode(
            x=x, y=alt.Y("y:Q", title=y_title), color=series,
            tooltip=[alt.Tooltip("model:N", title="model"), alt.Tooltip("x:Q", format=".4g"), alt.Tooltip("y:Q", format=".4g")]))
    if points:
        p = pd.DataFrame(points).assign(series="Measured")
        layers.append(alt.Chart(p).mark_circle(size=70, opacity=0.95, stroke="#fcfcfb", strokeWidth=1.5).encode(
            x=x, y=alt.Y("y:Q", title=y_title), color=series,
            tooltip=[alt.Tooltip("step:Q", title="attempt"), alt.Tooltip("x:Q", format=".4g"),
                     alt.Tooltip("y:Q", format=".4g"), alt.Tooltip("source:N")]))
    marks = [f for f in flagged if f.get("x") is not None]
    if marks:
        f = pd.DataFrame(marks)
        if "y" not in f or f["y"].isna().all():
            f["y"] = (pd.DataFrame(points)["y"].min() if points else 0.0)
        f["y"] = f["y"].fillna(pd.DataFrame(points)["y"].min() if points else 0.0)
        f["series"] = "Rejected or flagged"
        layers.append(alt.Chart(f).mark_point(shape="cross", size=110, filled=True).encode(
            x=x, y=alt.Y("y:Q", title=y_title), color=series,
            tooltip=[alt.Tooltip("step:Q", title="attempt"), alt.Tooltip("x:Q", format=".4g"), alt.Tooltip("kind:N")]))
    if not layers:
        layers.append(alt.Chart(pd.DataFrame({"x": x_domain, "y": [0, 0]})).mark_point(opacity=0).encode(
            x=x, y=alt.Y("y:Q", title=y_title)))
    return (alt.layer(*layers).properties(height=height)
            .configure_axis(gridColor=GRID, labelColor=MUTED, titleColor="#52514e", domainColor="#c3c2b7")
            .configure_view(strokeWidth=0)
            .configure_legend(labelColor="#52514e"))
