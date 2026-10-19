#!/usr/bin/env python3
"""Read Figure 4a vector rectangles from the two LIBRA PDF versions.

Read-only PDF operation; writes numeric evidence, not a modified PDF.
Optional dependency: PyMuPDF. Page indices and rectangle filter are explicit
so that a different document fails rather than silently digitizing another plot.
"""
import argparse
import hashlib
import json
from pathlib import Path


def extract(path, page_index, url):
    import pymupdf
    document = pymupdf.open(path)
    bars = []
    for drawing in document[page_index].get_drawings():
        if drawing["color"] != (0.0, 0.0, 1.0):
            continue
        for item in drawing["items"]:
            if item[0] == "re":
                rect = item[1]
                if 5 < rect.width < 7 and rect.height > 15 and rect.y1 < 150:
                    bars.append(rect)
    bars.sort(key=lambda rect: rect.x0)
    if len(bars) != 5:
        raise ValueError(f"Expected exactly five blue cost bars, found {len(bars)} in {path}")
    policies = ("LIBRA", "FAAS", "MAX", "SPOCK", "AUTO")
    return {"url": url, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "pdf_page_index_zero_based": page_index,
            "method": "Relative heights of five blue vector rectangles in Figure 4a; graphical readout, not raw author metrics",
            "cost_bars": {policy: {"rectangle_pt": list(rect), "normalized_cost": rect.height / bars[0].height}
                          for policy, rect in zip(policies, bars)},
            "max_to_faas_cost_ratio": bars[2].height / bars[1].height}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arxiv", type=Path, default=root / "references/libra-arxiv.pdf")
    parser.add_argument("--published", type=Path, default=root / "references/libra-published.pdf")
    parser.add_argument("--output", type=Path, default=root / "references/paper_evidence.json")
    args = parser.parse_args()
    evidence = {
        "arxiv": extract(args.arxiv, 7, "https://arxiv.org/pdf/2104.05491v1"),
        "published": extract(args.published, 8, "https://open.bu.edu/bitstreams/86e15fd8-9c9a-436b-9d16-d9072bf4a39f/download")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
