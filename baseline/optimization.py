import numpy as np

from geometry import calc_side_auto as calc_side, check_overlap_single, find_corner_trees, get_global_bbox


# --- Deterministic tiling based on (auto-tuned) mirror pair ---
# Default pair parameters (previous best from offline search)
PAIR_DX = 0.435
PAIR_DY = -0.46
PAIR_W = 1.135  # approximate AABB width of the pair
PAIR_H = 1.46   # approximate AABB height of the pair

# Cache for small auto-tuning of pair offsets to reduce bbox
_PAIR_CACHE = {}
# Optional override from high-scoring submission
_PAIR_OVERRIDE = None


def set_pair_override(dx: float, dy: float):
    global _PAIR_OVERRIDE
    _PAIR_OVERRIDE = (dx, dy)


def tune_pair_from_submission(csv_path: str):
    """Extract a pair offset guess from a submission (use n=2 rows if present).

    Heuristic: use the vector between the two trees of the n=2 case; fall back to median
    nearest-neighbor vector for the smallest n present. This is lightweight and avoids
    heavy search; intended to seed the deterministic tiling.
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df['n'] = df['id'].str.slice(0, 3).astype(int)
        df['idx'] = df['id'].str.split('_').str[1].astype(int)
        # prefer n=2
        if (df['n'] == 2).any():
            sub = df[df['n'] == 2].sort_values('idx')
        else:
            n_min = df['n'].min()
            sub = df[df['n'] == n_min].sort_values('idx')
        xs = sub['x'].astype(str).str.lstrip('s').astype(float).to_numpy()
        ys = sub['y'].astype(str).str.lstrip('s').astype(float).to_numpy()
        if len(xs) >= 2:
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
            set_pair_override(dx, dy)
            return dx, dy
        # fallback: try nearest-neighbor median
        from scipy.spatial import KDTree  # only if available
        pts = np.column_stack([xs, ys])
        tree = KDTree(pts)
        dists, idxs = tree.query(pts, k=2)
        vectors = pts[idxs[:, 1]] - pts
        dx = np.median(vectors[:, 0])
        dy = np.median(vectors[:, 1])
        set_pair_override(dx, dy)
        return dx, dy
    except Exception:
        return None


def select_pair_params(n: int):
    """Lightweight coarse search over (dx, dy) to tighten the tiling seed.

    - Only runs a small grid search; cached per coarse bucket of n.
    - Uses calc_side_auto to keep GPU acceleration when available.
    - Falls back to defaults if anything fails.
    """
    try:
        # bucket by size to avoid repeated work
        if n <= 10:
            bucket = 10
        elif n <= 40:
            bucket = 40
        elif n <= 100:
            bucket = 100
        else:
            bucket = 200
        if bucket in _PAIR_CACHE:
            return _PAIR_CACHE[bucket]

        sample_n = min(60, n)
        base_dx, base_dy = _PAIR_OVERRIDE if _PAIR_OVERRIDE is not None else (PAIR_DX, PAIR_DY)
        dx_candidates = np.linspace(base_dx - 0.05, base_dx + 0.05, 5)
        dy_candidates = np.linspace(base_dy - 0.05, base_dy + 0.05, 5)
        best = (base_dx, base_dy)
        best_side = None
        for dx in dx_candidates:
            for dy in dy_candidates:
                xs, ys, angs = build_mirror_pair_positions(sample_n, dx, dy, PAIR_W, PAIR_H)
                side = calc_side(xs, ys, angs, sample_n)
                if (best_side is None) or (side < best_side):
                    best_side = side
                    best = (dx, dy)
        _PAIR_CACHE[bucket] = best
        return best
    except Exception:
        return (PAIR_DX, PAIR_DY)


def build_mirror_pair_positions(n: int, pair_dx=PAIR_DX, pair_dy=PAIR_DY, pair_w=PAIR_W, pair_h=PAIR_H):
    """Build deterministic layout using mirrored pairs in staggered rows.

    Strategy:
    - Use two-tree cell: tree A at (0, 0, 0), tree B mirrored and shifted by (PAIR_DX, PAIR_DY, 0).
    - Arrange cells in staggered rows to reduce width: odd rows offset by half cell width.
    - Choose rows/cols near sqrt(cells) to minimize square side.
    - Works for any n (last lone tree placed at end of grid).
    """
    cells = (n + 1) // 2  # number of two-tree cells needed
    if cells == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    best = None
    k = 4  # neighborhood around sqrt
    root = int(np.sqrt(cells))
    candidates = []
    for r in range(max(1, root - k), root + k + 1):
        c = (cells + r - 1) // r
        candidates.append((r, c))

    def layout_size(rows, cols, stagger):
        width = cols * PAIR_W
        if stagger and rows > 1:
            width += 0.5 * PAIR_W  # due to half-cell offset on staggered rows
        height = rows * PAIR_H
        return max(width, height), width, height

    for rows, cols in candidates:
        for stagger in (False, True):
            s, w, h = layout_size(rows, cols, stagger)
            if (best is None) or (s < best[0]):
                best = (s, rows, cols, stagger, w, h)

    _, rows, cols, stagger, _, _ = best

    xs = np.zeros(n)
    ys = np.zeros(n)
    angs = np.zeros(n)

    idx = 0
    for r in range(rows):
        row_offset = 0.5 * pair_w if (stagger and (r % 2 == 1)) else 0.0
        for c in range(cols):
            if idx >= n:
                break
            base_x = c * pair_w + row_offset
            base_y = r * pair_h
            # tree A
            xs[idx] = base_x
            ys[idx] = base_y
            angs[idx] = 0.0
            idx += 1
            if idx >= n:
                break
            # tree B mirrored and shifted
            xs[idx] = base_x + pair_dx
            ys[idx] = base_y + pair_dy
            angs[idx] = 0.0
            idx += 1
        if idx >= n:
            break

    # Center the layout around origin to reduce bbox size
    if n > 0:
        xs_mean = xs.mean()
        ys_mean = ys.mean()
        xs -= xs_mean
        ys -= ys_mean

    return xs, ys, angs


def sa_v3(xs, ys, angs, n, iterations, T0, Tmin, move_scale, rot_scale, seed):
    np.random.seed(seed)
    bxs, bys, bangs = xs.copy(), ys.copy(), angs.copy()
    cxs, cys, cangs = xs.copy(), ys.copy(), angs.copy()
    bs = calc_side(bxs, bys, bangs, n)
    cs = bs
    T = T0
    alpha = (Tmin / T0) ** (1.0 / iterations)
    no_imp = 0
    for _ in range(iterations):
        move_type = np.random.randint(0, 8)
        sc = T / T0
        i = -1
        ox, oy, oa = 0.0, 0.0, 0.0
        if move_type < 4:
            i = np.random.randint(0, n)
            ox, oy, oa = cxs[i], cys[i], cangs[i]
            cx = np.mean(cxs[:n])
            cy = np.mean(cys[:n])
            if move_type == 0:
                cxs[i] += (np.random.random() - 0.5) * 2 * move_scale * sc
                cys[i] += (np.random.random() - 0.5) * 2 * move_scale * sc
            elif move_type == 1:
                dx, dy = cx - cxs[i], cy - cys[i]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-6:
                    step = np.random.random() * move_scale * sc
                    cxs[i] += dx / d * step
                    cys[i] += dy / d * step
            elif move_type == 2:
                cangs[i] += (np.random.random() - 0.5) * 2 * rot_scale * sc
                cangs[i] = cangs[i] % 360
            else:
                cxs[i] += (np.random.random() - 0.5) * move_scale * sc
                cys[i] += (np.random.random() - 0.5) * move_scale * sc
                cangs[i] += (np.random.random() - 0.5) * rot_scale * sc
                cangs[i] = cangs[i] % 360
        elif move_type == 5:
            i = np.random.randint(0, n)
            ox, oy = cxs[i], cys[i]
            gx0, gy0, gx1, gy1 = get_global_bbox(cxs, cys, cangs, n)
            bcx, bcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
            dx, dy = bcx - cxs[i], bcy - cys[i]
            d = np.sqrt(dx * dx + dy * dy)
            if d > 1e-6:
                step = np.random.random() * move_scale * sc * 0.5
                cxs[i] += dx / d * step
                cys[i] += dy / d * step
        elif move_type == 6:
            corners = find_corner_trees(cxs, cys, cangs, n)
            if len(corners) > 0:
                i = corners[np.random.randint(0, len(corners))]
                ox, oy, oa = cxs[i], cys[i], cangs[i]
                gx0, gy0, gx1, gy1 = get_global_bbox(cxs, cys, cangs, n)
                bcx, bcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
                dx, dy = bcx - cxs[i], bcy - cys[i]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-6:
                    step = np.random.random() * move_scale * sc * 0.3
                    cxs[i] += dx / d * step
                    cys[i] += dy / d * step
                    cangs[i] += (np.random.random() - 0.5) * rot_scale * sc * 0.5
                    cangs[i] = cangs[i] % 360
            else:
                no_imp += 1
                T *= alpha
                if T < Tmin:
                    T = Tmin
                continue
        if move_type == 4 or move_type == 7:
            no_imp += 1
            T *= alpha
            if T < Tmin:
                T = Tmin
            continue
        if i != -1 and check_overlap_single(i, cxs, cys, cangs, n):
            if move_type < 4 or move_type == 5:
                cxs[i], cys[i] = ox, oy
                if move_type != 5:
                    cangs[i] = oa
            elif move_type == 6:
                cxs[i], cys[i], cangs[i] = ox, oy, oa
            no_imp += 1
            T *= alpha
            if T < Tmin:
                T = Tmin
            continue
        ns = calc_side(cxs, cys, cangs, n)
        delta = ns - cs
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            cs = ns
            if ns < bs:
                bs = ns
                bxs[:] = cxs
                bys[:] = cys
                bangs[:] = cangs
                no_imp = 0
            else:
                no_imp += 1
        else:
            cxs[:] = bxs
            cys[:] = bys
            cangs[:] = bangs
            cs = bs
            no_imp += 1
        if no_imp > 600:
            T = min(T * 3.0, T0 * 0.7)
            no_imp = 0
        T *= alpha
        if T < Tmin:
            T = Tmin
    return bxs, bys, bangs, bs


def local_search_v3(xs, ys, angs, n, max_iter):
    bxs, bys, bangs = xs.copy(), ys.copy(), angs.copy()
    bs = calc_side(bxs, bys, bangs, n)
    pos_steps = np.array([0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002])
    rot_steps = np.array([15.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25])
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)
    for _ in range(max_iter):
        improved = False
        corners = find_corner_trees(bxs, bys, bangs, n)
        for ci in range(len(corners)):
            i = corners[ci]
            for ps in pos_steps:
                for d in range(8):
                    ox, oy = bxs[i], bys[i]
                    bxs[i] += dirs[d, 0] * ps
                    bys[i] += dirs[d, 1] * ps
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bxs[i], bys[i] = ox, oy
                    else:
                        bxs[i], bys[i] = ox, oy
            for rs in rot_steps:
                for da in [rs, -rs]:
                    oa = bangs[i]
                    bangs[i] = (bangs[i] + da) % 360
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bangs[i] = oa
                    else:
                        bangs[i] = oa
        for i in range(n):
            in_corners = False
            for ci in range(len(corners)):
                if corners[ci] == i:
                    in_corners = True
                    break
            if in_corners:
                continue
            for ps in pos_steps:
                for d in range(8):
                    ox, oy = bxs[i], bys[i]
                    bxs[i] += dirs[d, 0] * ps
                    bys[i] += dirs[d, 1] * ps
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bxs[i], bys[i] = ox, oy
                    else:
                        bxs[i], bys[i] = ox, oy
            for rs in rot_steps:
                for da in [rs, -rs]:
                    oa = bangs[i]
                    bangs[i] = (bangs[i] + da) % 360
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bangs[i] = oa
                    else:
                        bangs[i] = oa
        if not improved:
            break
    return bxs, bys, bangs, bs


def perturb(xs, ys, angs, n, strength, seed):
    np.random.seed(seed)
    pxs, pys, pangs = xs.copy(), ys.copy(), angs.copy()
    num_perturb = max(1, int(n * 0.15))
    for _ in range(num_perturb):
        i = np.random.randint(0, n)
        pxs[i] += (np.random.random() - 0.5) * strength
        pys[i] += (np.random.random() - 0.5) * strength
        pangs[i] = (pangs[i] + (np.random.random() - 0.5) * 60) % 360
    for _ in range(100):
        fixed = True
        for i in range(n):
            if check_overlap_single(i, pxs, pys, pangs, n):
                fixed = False
                cx, cy = np.mean(pxs[:n]), np.mean(pys[:n])
                dx, dy = cx - pxs[i], cy - pys[i]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-6:
                    pxs[i] -= dx / d * 0.02
                    pys[i] -= dy / d * 0.02
                pangs[i] = (pangs[i] + np.random.random() * 20 - 10) % 360
        if fixed:
            break
    return pxs, pys, pangs


def optimize_config(n, xs, ys, angs, num_restarts, sa_iters, fast_only=False):
    """Hybrid: deterministic tiling seed + SA/local refinement.

    fast_only=True skips SA/local search and returns the deterministic tiling seed directly
    for lower CPU usage (useful on Kaggle when GPU is underutilized).
    """

    # deterministic seed using mirror tiling (auto-tuned pair params per n bucket)
    pair_dx, pair_dy = select_pair_params(n)
    seed_xs, seed_ys, seed_angs = build_mirror_pair_positions(n, pair_dx, pair_dy, PAIR_W, PAIR_H)
    seed_side = calc_side(seed_xs, seed_ys, seed_angs, n)
    init_side = calc_side(xs, ys, angs, n)
    if seed_side < init_side:
        xs0, ys0, angs0 = seed_xs, seed_ys, seed_angs
    else:
        xs0, ys0, angs0 = xs.copy(), ys.copy(), angs.copy()

    best_xs, best_ys, best_angs = xs0.copy(), ys0.copy(), angs0.copy()
    best_side = calc_side(best_xs, best_ys, best_angs, n)

    if fast_only:
        return best_xs, best_ys, best_angs, best_side

    population = [(xs0.copy(), ys0.copy(), angs0.copy(), best_side)]
    # also keep the original input as fallback candidate
    if init_side < best_side:
        population.append((xs.copy(), ys.copy(), angs.copy(), init_side))

    for r in range(num_restarts):
        if r < len(population):
            px, py, pa, _ = population[r % len(population)]
            start_xs, start_ys, start_angs = px.copy(), py.copy(), pa.copy()
        else:
            px, py, pa, _ = population[0]
            start_xs, start_ys, start_angs = perturb(px, py, pa, n, 0.1 + 0.05 * (r % 3), 42 + r * 1000 + n)
        seed = 42 + r * 1000 + n
        oxs, oys, oangs, os = sa_v3(start_xs, start_ys, start_angs, n, sa_iters,
                                     1.0, 0.000005, 0.25, 70.0, seed)
        oxs, oys, oangs, os = local_search_v3(oxs, oys, oangs, n, 300)
        population.append((oxs.copy(), oys.copy(), oangs.copy(), os))
        population.sort(key=lambda x: x[3])
        population = population[:3]
        if os < best_side:
            best_side = os
            best_xs, best_ys, best_angs = oxs.copy(), oys.copy(), oangs.copy()
    return best_xs, best_ys, best_angs, best_side
