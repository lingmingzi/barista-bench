import math

import numpy as np
from numba import cuda, float64, int32, njit, prange

# Tree polygon vertices (global for both CPU and GPU paths)
TREE_X = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075,
                   -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125], dtype=np.float64)
TREE_Y = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2,
                   -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5], dtype=np.float64)
NV = 15

# GPU launch configuration (balanced to avoid resource overuse)
PAIR_THREADS = (16, 16)  # 256 threads per block for pairwise overlap
BBOX_THREADS = 256       # 256 threads per block for bbox kernel
GPU_MIN_N = 32           # below this, prefer CPU to avoid tiny grids/overheads
FORCE_GPU = False        # if True, always attempt GPU path when CUDA is available


def configure_gpu_threads(pair_block_x: int = None, pair_block_y: int = None,
                          bbox_threads: int = None, gpu_min_n: int = None,
                          force_gpu: bool = None):
    global PAIR_THREADS, BBOX_THREADS, GPU_MIN_N, FORCE_GPU
    if pair_block_x and pair_block_y:
        PAIR_THREADS = (int(pair_block_x), int(pair_block_y))
    if bbox_threads:
        BBOX_THREADS = int(bbox_threads)
    if gpu_min_n is not None:
        GPU_MIN_N = int(gpu_min_n)
    if force_gpu is not None:
        FORCE_GPU = bool(force_gpu)
        if FORCE_GPU:
            GPU_MIN_N = 0


# --- CUDA device helpers ---
@cuda.jit(device=True)
def get_poly_device(cx, cy, deg, px, py):
    rad = deg * np.pi / 180.0
    c, s = np.cos(rad), np.sin(rad)
    for i in range(NV):
        tx = TREE_X[i]
        ty = TREE_Y[i]
        px[i] = tx * c - ty * s + cx
        py[i] = tx * s + ty * c + cy


@cuda.jit(device=True)
def get_bbox_device(px, py, bbox):
    x0, y0, x1, y1 = px[0], py[0], px[0], py[0]
    for i in range(1, NV):
        if px[i] < x0:
            x0 = px[i]
        if py[i] < y0:
            y0 = py[i]
        if px[i] > x1:
            x1 = px[i]
        if py[i] > y1:
            y1 = py[i]
    bbox[0] = x0
    bbox[1] = y0
    bbox[2] = x1
    bbox[3] = y1


@cuda.jit(device=True)
def pip_device(px_pt, py_pt, poly_x, poly_y):
    inside = False
    j = NV - 1
    for i in range(NV):
        if ((poly_y[i] > py_pt) != (poly_y[j] > py_pt) and
                px_pt < (poly_x[j] - poly_x[i]) * (py_pt - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]):
            inside = not inside
        j = i
    return inside


@cuda.jit(device=True)
def ccw_device(p1x, p1y, p2x, p2y, p3x, p3y):
    return (p3y - p1y) * (p2x - p1x) > (p2y - p1y) * (p3x - p1x)


@cuda.jit(device=True)
def seg_intersect_device(ax, ay, bx, by, cx, cy, dx, dy):
    return ccw_device(ax, ay, cx, cy, dx, dy) != ccw_device(bx, by, cx, cy, dx, dy) and \
           ccw_device(ax, ay, bx, by, cx, cy) != ccw_device(ax, ay, bx, by, dx, dy)


@cuda.jit(device=True)
def overlap_device(px1, py1, bb1, px2, py2, bb2):
    if bb1[2] < bb2[0] or bb2[2] < bb1[0] or bb1[3] < bb2[1] or bb2[3] < bb1[1]:
        return False
    for i in range(NV):
        if pip_device(px1[i], py1[i], px2, py2):
            return True
    for i in range(NV):
        if pip_device(px2[i], py2[i], px1, py1):
            return True
    for i in range(NV):
        ni = (i + 1) % NV
        for j in range(NV):
            nj = (j + 1) % NV
            if seg_intersect_device(px1[i], py1[i], px1[ni], py1[ni], px2[j], py2[j], px2[nj], py2[nj]):
                return True
    return False


@cuda.jit
def overlap_check_kernel(idx, n, xs, ys, angs, is_overlapping_out):
    j = cuda.grid(1)
    if j >= n or j == idx:
        return
    px1 = cuda.local.array(NV, float64)
    py1 = cuda.local.array(NV, float64)
    px2 = cuda.local.array(NV, float64)
    py2 = cuda.local.array(NV, float64)
    bb1 = cuda.local.array(4, float64)
    bb2 = cuda.local.array(4, float64)
    get_poly_device(xs[idx], ys[idx], angs[idx], px1, py1)
    get_bbox_device(px1, py1, bb1)
    get_poly_device(xs[j], ys[j], angs[j], px2, py2)
    get_bbox_device(px2, py2, bb2)
    if overlap_device(px1, py1, bb1, px2, py2, bb2):
        cuda.atomic.max(is_overlapping_out, 0, 1)


@cuda.jit
def overlap_any_kernel(xs, ys, angs, n, is_overlapping_out):
    # 2D grid over pairs (i, j); only evaluate i < j
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i >= n or j >= n or i >= j:
        return
    px1 = cuda.local.array(NV, float64)
    py1 = cuda.local.array(NV, float64)
    px2 = cuda.local.array(NV, float64)
    py2 = cuda.local.array(NV, float64)
    bb1 = cuda.local.array(4, float64)
    bb2 = cuda.local.array(4, float64)
    get_poly_device(xs[i], ys[i], angs[i], px1, py1)
    get_bbox_device(px1, py1, bb1)
    get_poly_device(xs[j], ys[j], angs[j], px2, py2)
    get_bbox_device(px2, py2, bb2)
    if overlap_device(px1, py1, bb1, px2, py2, bb2):
        cuda.atomic.max(is_overlapping_out, 0, 1)


@cuda.jit
def bbox_kernel(xs, ys, angs, bboxes, n):
    i = cuda.grid(1)
    if i >= n:
        return
    rad = angs[i] * math.pi / 180.0
    c = math.cos(rad)
    s = math.sin(rad)
    x0 = 1e9
    y0 = 1e9
    x1 = -1e9
    y1 = -1e9
    for k in range(NV):
        tx = TREE_X[k]
        ty = TREE_Y[k]
        px = tx * c - ty * s + xs[i]
        py = tx * s + ty * c + ys[i]
        if px < x0:
            x0 = px
        if py < y0:
            y0 = py
        if px > x1:
            x1 = px
        if py > y1:
            y1 = py
    bboxes[i, 0] = x0
    bboxes[i, 1] = y0
    bboxes[i, 2] = x1
    bboxes[i, 3] = y1


# --- CPU helpers ---
@njit(cache=True)
def get_poly(cx, cy, deg):
    rad = deg * np.pi / 180.0
    c, s = np.cos(rad), np.sin(rad)
    px = TREE_X * c - TREE_Y * s + cx
    py = TREE_X * s + TREE_Y * c + cy
    return px, py


@njit(cache=True)
def get_bbox(px, py):
    return px.min(), py.min(), px.max(), py.max()


@njit(cache=True, parallel=True)
def calc_side(xs, ys, angs, n):
    if n == 0:
        return 0.0
    x_min_arr = np.full(n, 1e9, dtype=np.float64)
    y_min_arr = np.full(n, 1e9, dtype=np.float64)
    x_max_arr = np.full(n, -1e9, dtype=np.float64)
    y_max_arr = np.full(n, -1e9, dtype=np.float64)
    for i in prange(n):
        px, py = get_poly(xs[i], ys[i], angs[i])
        x0, y0, x1, y1 = get_bbox(px, py)
        x_min_arr[i] = x0
        y_min_arr[i] = y0
        x_max_arr[i] = x1
        y_max_arr[i] = y1
    gx0 = x_min_arr.min()
    gy0 = y_min_arr.min()
    gx1 = x_max_arr.max()
    gy1 = y_max_arr.max()
    return max(gx1 - gx0, gy1 - gy0)


def calc_side_gpu(xs, ys, angs, n):
    if n == 0:
        return 0.0
    if n < GPU_MIN_N:
        return calc_side(xs, ys, angs, n)
    # allocate device buffers once per call; data copies dominate but move bbox math to GPU
    d_xs = cuda.to_device(xs[:n])
    d_ys = cuda.to_device(ys[:n])
    d_angs = cuda.to_device(angs[:n])
    d_bboxes = cuda.device_array((n, 4), dtype=np.float64)
    threads_per_block = BBOX_THREADS
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    bbox_kernel[blocks_per_grid, threads_per_block](d_xs, d_ys, d_angs, d_bboxes, n)
    cuda.synchronize()
    bboxes = d_bboxes.copy_to_host()
    gx0 = bboxes[:, 0].min()
    gy0 = bboxes[:, 1].min()
    gx1 = bboxes[:, 2].max()
    gy1 = bboxes[:, 3].max()
    return max(gx1 - gx0, gy1 - gy0)


def calc_side_auto(xs, ys, angs, n):
    try:
        if cuda.is_available():
            if FORCE_GPU or n >= GPU_MIN_N:
                return calc_side_gpu(xs, ys, angs, n)
    except Exception:
        pass
    return calc_side(xs, ys, angs, n)


@njit(cache=True)
def get_global_bbox(xs, ys, angs, n):
    gx0, gy0, gx1, gy1 = 1e9, 1e9, -1e9, -1e9
    for i in range(n):
        px, py = get_poly(xs[i], ys[i], angs[i])
        x0, y0, x1, y1 = get_bbox(px, py)
        gx0, gy0 = min(gx0, x0), min(gy0, y0)
        gx1, gy1 = max(gx1, x1), max(gy1, y1)
    return gx0, gy0, gx1, gy1


@njit(cache=True)
def find_corner_trees(xs, ys, angs, n):
    gx0, gy0, gx1, gy1 = get_global_bbox(xs, ys, angs, n)
    eps = 0.01
    corner_trees = np.zeros(n, dtype=np.int32)
    count = 0
    for i in range(n):
        px, py = get_poly(xs[i], ys[i], angs[i])
        x0, y0, x1, y1 = get_bbox(px, py)
        if abs(x0 - gx0) < eps or abs(x1 - gx1) < eps or \
           abs(y0 - gy0) < eps or abs(y1 - gy1) < eps:
            corner_trees[count] = i
            count += 1
    return corner_trees[:count]


# --- CPU overlap fallback ---
def pip_cpu(px_pt, py_pt, poly_x, poly_y):
    inside = False
    j = NV - 1
    for i in range(NV):
        if ((poly_y[i] > py_pt) != (poly_y[j] > py_pt) and
                px_pt < (poly_x[j] - poly_x[i]) * (py_pt - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]):
            inside = not inside
        j = i
    return inside


def seg_intersect_cpu(ax, ay, bx, by, cx, cy, dx, dy):
    def ccw(px1, py1, px2, py2, px3, py3):
        return (py3 - py1) * (px2 - px1) > (py2 - py1) * (px3 - px1)
    return ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and \
        ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)


def overlap_cpu(px1, py1, bb1, px2, py2, bb2):
    if bb1[2] < bb2[0] or bb2[2] < bb1[0] or bb1[3] < bb2[1] or bb2[3] < bb1[1]:
        return False
    for i in range(NV):
        if pip_cpu(px1[i], py1[i], px2, py2):
            return True
    for i in range(NV):
        if pip_cpu(px2[i], py2[i], px1, py1):
            return True
    for i in range(NV):
        ni = (i + 1) % NV
        for j in range(NV):
            nj = (j + 1) % NV
            if seg_intersect_cpu(px1[i], py1[i], px1[ni], py1[ni], px2[j], py2[j], px2[nj], py2[nj]):
                return True
    return False


def check_overlap_single(idx: int, xs: np.ndarray, ys: np.ndarray, angs: np.ndarray, n: int) -> bool:
    if n <= 1:
        return False
    # CPU fallback when CUDA is unavailable
    try:
        gpu_available = cuda.is_available()
    except Exception:
        gpu_available = False
    if (not gpu_available) or ((not FORCE_GPU) and n < GPU_MIN_N):
        px1, py1 = get_poly(xs[idx], ys[idx], angs[idx])
        bb1 = (px1.min(), py1.min(), px1.max(), py1.max())
        for j in range(n):
            if j == idx:
                continue
            px2, py2 = get_poly(xs[j], ys[j], angs[j])
            bb2 = (px2.min(), py2.min(), px2.max(), py2.max())
            if overlap_cpu(px1, py1, bb1, px2, py2, bb2):
                return True
        return False
    # GPU path: check all pairs in one kernel to increase GPU utilization and reduce Python/CPU work
    is_overlapping_out = np.zeros(1, dtype=np.int32)
    d_is_overlapping_out = cuda.to_device(is_overlapping_out)
    d_xs = cuda.to_device(xs)
    d_ys = cuda.to_device(ys)
    d_angs = cuda.to_device(angs)
    threads_per_block = PAIR_THREADS
    blocks_per_grid = ((n + threads_per_block[0] - 1) // threads_per_block[0],
                       (n + threads_per_block[1] - 1) // threads_per_block[1])
    overlap_any_kernel[blocks_per_grid, threads_per_block](d_xs, d_ys, d_angs, n, d_is_overlapping_out)
    cuda.synchronize()
    d_is_overlapping_out.copy_to_host(is_overlapping_out)
    return bool(is_overlapping_out[0])
