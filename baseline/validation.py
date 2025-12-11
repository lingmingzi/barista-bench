from decimal import Decimal, getcontext
from typing import List

import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

getcontext().prec = 25
scale_factor = Decimal("1e15")


class ChristmasTree:
    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h
        initial_polygon = Polygon([
            (Decimal("0.0") * scale_factor, tip_y * scale_factor),
            (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated, xoff=float(self.center_x * scale_factor), yoff=float(self.center_y * scale_factor))


def load_configuration_from_df(n: int, df: pd.DataFrame) -> List[ChristmasTree]:
    group_data = df[df["id"].str.startswith(f"{n:03d}_")]
    trees = []
    for _, row in group_data.iterrows():
        x = str(row["x"])[1:]
        y = str(row["y"])[1:]
        deg = str(row["deg"])[1:]
        if x and y and deg:
            trees.append(ChristmasTree(x, y, deg))
    return trees


def get_score(trees: List[ChristmasTree], n: int) -> float:
    if not trees:
        return 0.0
    xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / float(scale_factor) for t in trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    side_length = max(max_x - min_x, max_y - min_y)
    return side_length**2 / n


def has_overlap(trees: List[ChristmasTree]) -> bool:
    if len(trees) <= 1:
        return False
    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)
    for i, poly in enumerate(polygons):
        indices = tree_index.query(poly)
        for idx in indices:
            if idx == i:
                continue
            if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                return True
    return False


def score_and_validate_submission(file_path: str, max_n: int = 200) -> dict:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {"status": "FAILED", "error": "File Not Found"}
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {"status": "FAILED", "error": f"CSV Read Error: {e}"}
    total_score = 0.0
    failed_overlap_n = []
    print(f"--- Scoring and Validation: {file_path} (N=1 to {max_n}) ---")
    for n in range(1, max_n + 1):
        trees = load_configuration_from_df(n, df)
        if trees:
            current_score = get_score(trees, n)
            total_score += current_score
            if has_overlap(trees):
                failed_overlap_n.append(n)
                print(f"  ❌ N={n:03d}: OVERLAP DETECTED! (Score contribution: {current_score:.6f})")
    print("\n--- Summary ---")
    if failed_overlap_n:
        print(f"❌ Validation FAILED: Overlaps found in N: {failed_overlap_n}")
        status = "FAILED (Overlaps)"
    else:
        print("✅ Validation SUCCESSFUL: No overlaps detected.")
        status = "SUCCESS"
    print(f"Total Submission Score (Σ S²/N): {total_score:.6f}")
    return {"status": status, "total_score": total_score, "failed_overlap_n": failed_overlap_n}
