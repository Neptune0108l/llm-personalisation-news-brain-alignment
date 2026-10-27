"""Reorganise the saved v8 evidence into Figures 5a–c and 6a–c.

Python/Matplotlib renders saved estimates and significance labels. PyMuPDF
preserves the unchanged cortical artwork. No statistics or RSA are recomputed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pymupdf

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "release.json").exists())
RECORD = ROOT / "outputs/figures_05_06"
RECORD.mkdir(parents=True, exist_ok=True)
(RECORD / "qa").mkdir(exist_ok=True)

COLOURS = {"A": "#7B818A", "S": "#B68132", "T": "#256B95"}
MARKERS = {"A": "o", "S": "^", "T": "D"}
FOCUS = "#A26F4A"
ROIS = ["value", "self_referential", "mentalizing"]
ROI_NAMES = {"value": "Value", "self_referential": "Self", "mentalizing": "Mentalizing"}
ENDPOINT = {"A": "article_only", "S": "shuffled", "T": "true_history"}
READ_FILES = {
    "brain_behaviour": "figure_05/brain_behaviour_summary.csv",
    "model_behaviour": "figure_05/MainFigure1_c_model_behavior.csv",
    "layers": "figure_05/MainFigure1_d_layers.csv",
    "behaviour_tests": "figure_05/Qwen_RSA_exploratory_significance.csv",
    "sa_tests": "figure_05/Qwen_ABC_AS_significance.csv",
    "brain_absolute": "figure_06/MainFigure2_c_absolute.csv",
    "brain_tests": "figure_06/MainFigure2_c_significance.csv",
    "semantic": "figure_06/MainFigure2_d_semantic.csv",
    "semantic_tests": "figure_06/MainFigure2_d_raw_vs_control_significance.csv",
}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def point(ax, row, y, colour, marker="o"):
    lo, mean, hi = row["ci_low"], row["mean"], row["ci_high"]
    assert lo <= mean <= hi
    line, = ax.plot([lo, hi], [y, y], color=colour, lw=1, zorder=2)
    dot, = ax.plot(mean, y, marker=marker, color=colour, ms=3.8, ls="none", zorder=3)
    np.testing.assert_allclose(line.get_xdata(), [lo, hi])
    np.testing.assert_allclose(dot.get_xdata(), [mean])


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def heading(ax, letter, title, width, y=1.12):
    ax.text(-13 / width, y, letter, transform=ax.transAxes, va="bottom",
            fontweight="bold", fontsize=10)
    ax.text(0, y, title, transform=ax.transAxes, va="bottom", fontsize=8)


def alignment(fig, axes, ids, name):
    """Record axes bounds; the original local skill QA dependency is not bundled."""
    fig.canvas.draw()
    bounds = {key: list(ax.get_position().bounds) for key, ax in zip(ids, axes)}
    (RECORD / f"qa/{name}.axes.json").write_text(json.dumps(bounds, indent=2))


def export_matplotlib(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(RECORD / f"qa/{stem.stem}_preview.png", dpi=300)
    plt.close(fig)


def plot5(data, output):
    width, height = 183 / 25.4 * 72, 250
    fig = plt.figure(figsize=(width / 72, height / 72))
    axes = [fig.add_axes([x / width, 55 / height, 100 / width, 145 / height])
            for x in (58, 221, 384)]
    a, b, c = axes
    for ax, letter, title in zip(axes, "abc",
            ["Brain–behaviour RSA", "Model–behaviour RSA", "Layer-wise RSA"]):
        heading(ax, letter, title, 100)
        ax.tick_params(length=2.5, pad=3)
    order = ROIS + ["memory", "language", "emotion", "arousal", "attention", "visual"]
    bb = data["brain_behaviour"].set_index("roi")
    assert set(bb.index) == set(order)
    for y, roi in enumerate(order):
        point(a, bb.loc[roi], y, FOCUS if roi in ROIS else "#8D959B")
    a.set(yticks=range(9), yticklabels=[ROI_NAMES.get(r, r.capitalize()) for r in order],
          ylim=(8.6, -.6), xlim=(-.027, .065), xticks=[-.02, 0, .02, .04, .06],
          xlabel="Brain–behaviour RSA ρ")
    a.axvline(0, lw=.65, color="#C9C9C9", zorder=0)
    means = data["model_behaviour"].set_index("endpoint")
    for y, endpoint in enumerate(["A", "S", "T"]):
        point(b, means.loc[endpoint], y, COLOURS[endpoint], MARKERS[endpoint])
    b.set(yticks=[0, 1, 2], yticklabels=["Article only", "Shuffled history", "True history"],
          ylim=(2.6, -.6), xlim=(0, .118), xticks=[0, .03, .06, .09], xlabel="RSA ρ")
    b.axvline(0, lw=.65, color="#C9C9C9", zorder=0)
    for contrast, start, x in [("T-S", 1, .084), ("T-A", 0, .101)]:
        row = data["behaviour_tests"].query("panel == 'c' and contrast == @contrast").iloc[0]
        b.plot([x-.0015, x, x, x-.0015], [start, start, 2, 2],
               color="#444444", lw=.65, clip_on=False)
        b.annotate(row.label, (x, (start+2)/2), xytext=(2, 0),
                   textcoords="offset points", fontsize=7, va="center")
    row = data["sa_tests"].query("panel == 'c'").iloc[0]
    b.plot([.0825, .084, .084, .0825], [0, 0, 1, 1], color="#444444", lw=.65)
    b.annotate(row.label, (.084, .5), xytext=(2, 0), textcoords="offset points",
               fontsize=7, va="center")
    layers = data["layers"]
    last = layers.query("layer == 28").set_index("endpoint")["mean"]
    for endpoint in ["A", "S", "T"]:
        q = layers[layers.endpoint == endpoint].sort_values("layer")
        assert q.layer.tolist() == list(range(29))
        line, = c.plot(q.layer, q["mean"], color=COLOURS[endpoint], lw=1.1,
                       marker=MARKERS[endpoint], ms=2.6, markevery=[7, 14, 21, 28])
        c.fill_between(q.layer, q.ci_low, q.ci_high, color=COLOURS[endpoint],
                       alpha=.13, linewidth=0)
        np.testing.assert_allclose(line.get_ydata(), q["mean"], equal_nan=True)
    c.axhline(0, color="#C9C9C9", lw=.65)
    c.set(xlim=(0, 28.4), xticks=[0, 7, 14, 21, 28], yticks=[0, .04, .08, .12],
          xlabel="Qwen layer", ylabel="RSA ρ")
    c.yaxis.labelpad = 2
    for contrast, ep1, ep2, x in [
        ("S-A", "S", "A", 1.035), ("T-S", "T", "S", 1.035),
        ("T-A", "T", "A", 1.15),
    ]:
        row = data["behaviour_tests"].query("panel == 'd' and contrast == @contrast").iloc[0]
        y1, y2 = last[ep1], last[ep2]
        c.plot([x-.015, x, x, x-.015], [y1, y1, y2, y2],
               transform=c.get_yaxis_transform(), color="#444444", lw=.65, clip_on=False)
        c.annotate(row.label, (x, (y1+y2)/2), xycoords=c.get_yaxis_transform(),
                   xytext=(2, 0), textcoords="offset points", va="center", fontsize=7)
    handles = [
        Line2D([], [], color=COLOURS[e], marker=MARKERS[e], lw=1.1, ms=3.8, label=t)
        for e, t in [("A", "Article only"), ("S", "Shuffled history"), ("T", "True history")]
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.63, .015),
               ncol=3, frameon=False, columnspacing=1.25, handlelength=1.5)
    alignment(fig, axes, list("abc"), "Figure_05")
    export_matplotlib(fig, output / "Figure_05")


def plot6(data, output):
    source = ROOT / "figures/main/Figure_06.pdf"
    source_doc = pymupdf.open(source)
    page = source_doc[0]
    width = page.rect.width
    crop_height, bottom_height = 312.0, 215.0
    # The source panel-a artwork and colour scale end at y=304 pt.
    # Remove former lower panels from the in-memory source to avoid hidden text.
    page.add_redact_annot(pymupdf.Rect(0, crop_height, width, page.rect.height),
                         fill=False, cross_out=False)
    page.apply_redactions(images=0, graphics=1, text=0)
    page.set_cropbox(pymupdf.Rect(0, 0, width, crop_height))
    assert "Brain–behavior" not in page.get_text().replace("\xa0", " ")
    fig = plt.figure(figsize=(width / 72, bottom_height / 72))
    axes = [fig.add_axes([x/width, 55/bottom_height, 150/width, 120/bottom_height])
            for x in (78, 325)]
    b, c = axes
    for ax, letter, title in zip(axes, "bc",
            ["Model–brain RSA", "Semantic adjustment: T − S"]):
        heading(ax, letter, title, 150)
        ax.set(yticks=range(3), yticklabels=[ROI_NAMES[r] for r in ROIS], ylim=(2.6, -.6))
        ax.tick_params(length=2.5, pad=3)
        ax.axvline(0, color="#CCCCCC", lw=.65, zorder=0)
    annotations = []
    for y, roi in enumerate(ROIS):
        for j, endpoint in enumerate(["A", "S", "T"]):
            row = data["brain_absolute"][
                (data["brain_absolute"].roi == roi)
                & (data["brain_absolute"].endpoint == ENDPOINT[endpoint])
            ]
            assert len(row) == 1
            point(b, row.iloc[0], y + (j-1)*.22, COLOURS[endpoint], MARKERS[endpoint])
        for contrast, x, y0, y1 in [
            ("true_minus_shuffled", .078, y, y+.22),
            ("true_minus_article", .102, y-.22, y+.22),
        ]:
            row = data["brain_tests"].query("roi == @roi and contrast == @contrast").iloc[0]
            label = stars(row.p_holm)
            b.plot([x-.0018, x, x, x-.0018], [y0, y0, y1, y1],
                   color="#333333", lw=.65, clip_on=False)
            b.annotate(label, (x, (y0+y1)/2), xytext=(3, 0),
                       textcoords="offset points", fontsize=7, va="center")
            annotations.append(dict(panel="6b", roi=roi, contrast=contrast,
                                    p_holm=float(row.p_holm), label=label))
        for j, variant in enumerate(["raw", "semantic_only"]):
            row = data["semantic"].query("roi == @roi and variant == @variant").iloc[0]
            point(c, row, y+(j-.5)*.24, ["#777777", FOCUS][j], ["o", "D"][j])
        row = data["semantic_tests"].query("roi == @roi").iloc[0]
        c.plot([.0118, .012, .012, .0118], [y-.12, y-.12, y+.12, y+.12],
               color="#333333", lw=.65)
        c.annotate(row.label, (.012, y), xytext=(3, 0),
                   textcoords="offset points", fontsize=7, va="center")
        annotations.append(dict(panel="6c", roi=roi, contrast=row.contrast,
                                p_holm=float(row.p_holm), label=row.label))
    b.set(xlim=(-.015, .12), xticks=[0, .03, .06, .09], xlabel="Model–brain RSA ρ")
    c.set(xlim=(-.001, .0145), xticks=[0, .005, .01], xlabel="ΔRSA ρ")
    b.legend(handles=[
        Line2D([], [], color=COLOURS[e], marker=MARKERS[e], ms=3.5, lw=1, label=t)
        for e, t in [("A", "Article only"), ("S", "Shuffled history"), ("T", "True history")]
    ], loc="upper center", bbox_to_anchor=(.5, -.24), ncol=3, frameon=False,
        handlelength=1.2, handletextpad=.45, columnspacing=.9, borderaxespad=0)
    c.legend(handles=[
        Line2D([], [], color=col, marker=marker, ms=3.5, lw=1, label=label)
        for col, marker, label in [("#777777", "o", "Raw"), (FOCUS, "D", "Semantic control")]
    ], loc="upper center", bbox_to_anchor=(.5, -.24), ncol=2, frameon=False,
        handlelength=1.2, handletextpad=.45, columnspacing=.9, borderaxespad=0)
    alignment(fig, axes, list("bc"), "Figure_06_bottom")
    bottom_path = RECORD / "qa/Figure_06_bottom.pdf"
    fig.savefig(bottom_path)
    plt.close(fig)
    bottom = pymupdf.open(bottom_path)
    result = pymupdf.open()
    dest = result.new_page(width=width, height=crop_height+bottom_height)
    dest.show_pdf_page(pymupdf.Rect(0, 0, width, crop_height), source_doc, 0)
    dest.show_pdf_page(pymupdf.Rect(0, crop_height, width, crop_height+bottom_height), bottom, 0)
    stem = output / "Figure_06"
    result.save(stem.with_suffix(".pdf"), garbage=4, deflate=True)
    dest.get_pixmap(dpi=600, alpha=False).save(stem.with_suffix(".png"))
    stem.with_suffix(".svg").write_text(dest.get_svg_image(text_as_path=False), encoding="utf-8")
    dest.get_pixmap(dpi=300, alpha=False).save(RECORD / "qa/Figure_06_preview.png")
    # Composition preserves panel-a geometry without rescaling.
    with pymupdf.open(source) as old_export, pymupdf.open(stem.with_suffix(".pdf")) as new_export:
        old_pix = old_export[0].get_pixmap(dpi=300, clip=pymupdf.Rect(0, 0, width, crop_height))
        new_pix = new_export[0].get_pixmap(dpi=300, clip=pymupdf.Rect(0, 0, width, crop_height))
        assert old_pix.samples == new_pix.samples, "Cortical artwork changed."
    assert dest.get_text().count("***") == 6
    assert dest.get_text().count("ns") >= 3
    assert "Brain–behaviour" not in dest.get_text()
    (RECORD / "qa/Figure_06_annotations.json").write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RECORD / "candidates")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    data = {}
    hashes = {}
    for key, name in READ_FILES.items():
        path = ROOT / "data" / name
        data[key] = pd.read_csv(path)
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    plot5(data, args.output)
    plot6(data, args.output)
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    (RECORD / "source_data_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Figure 5 and Figure 6 exported from frozen source tables; no statistics rerun.")


if __name__ == "__main__":
    main()
