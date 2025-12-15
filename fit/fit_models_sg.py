# fit_models_sg.py
import numpy as np
from kneed import KneeLocator
from scipy.signal import savgol_filter

from config import DENSE_N
from sliding_stats import cv_uncertainty_rapidcode

# ========== Savitzky–Golay 平滑参数（你可以在这里改） ==========
# SG window length（必须是正奇数；下面代码会按需要自动调成奇数）
SG_WINDOW_DEFAULT = 21

# 多项式阶数 polyorder（必须 < window_length）
SG_POLY_DEFAULT = 2

# 至少需要的数据点数（避免点太少导致拟合/平滑不稳定）
MIN_POINTS = SG_POLY_DEFAULT + 2
# =============================================================


def _choose_sg_params(n_points: int):

    # 至少要 3 个点才能做 SG
    if n_points < 3:
        return None, None

    # 先选 window_length：不超过 n_points，且为奇数
    win = min(SG_WINDOW_DEFAULT, n_points)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        win = 3

    # 再选 polyorder：不超过默认值，且 < window_length
    poly = min(SG_POLY_DEFAULT, win - 1)
    if poly < 1:
        poly = 1

    return win, poly


def fit_models_and_knees(x_um: np.ndarray,
                         cv: np.ndarray,
                         n_windows: np.ndarray):
    """
    使用 Savitzky–Golay 对 CV(x) 做平滑，然后在平滑曲线上用 KneeLocator 找 knee 点。
    Interface is aligned with the original `fit_models.py` and your `fit_models_poly.py`:
        - Function name: fit_models_and_knees
        - Args: x_um, cv, n_windows
        - Returns: dict with keys
            x_dense_um, cv_exp_dense, cv_hyp_dense,
            knee_exp_um, knee_hyp_um,
            chi2_red_exp, chi2_red_hyper,
            exp_params, hyp_params

    In this SG version, we only use the "exp" branch to store results:
        - cv_exp_dense: SG-smoothed curve evaluated on a dense x grid
        - knee_exp_um:  knee of the smoothed curve
        - chi2_red_exp: reduced χ² between raw CV and smoothed CV
        - exp_params:   np.array([window_length, polyorder])

      The "hyper" branch (cv_hyp_dense, knee_hyp_um, chi2_red_hyper, hyp_params)
      is kept for interface compatibility but not used (set to None / NaN).
    """

    # ---- 初始化返回字典 / initialize result dict ----
    result = {
        "x_dense_um": None,
        "cv_exp_dense": None,
        "cv_hyp_dense": None,
        "knee_exp_um": None,
        "knee_hyp_um": None,
        "chi2_red_exp": np.nan,
        "chi2_red_hyper": np.nan,
        "exp_params": None,
        "hyp_params": None,
    }

    # ---- 过滤无效数据点 / filter invalid points ----
    mask = np.isfinite(x_um) & np.isfinite(cv) & (cv > 0) & (n_windows > 0)
    x = x_um[mask]
    y = cv[mask]
    n = n_windows[mask].astype(float)

    if x.size < MIN_POINTS:
        # 点太少，不进行平滑和 knee 检测
        # too few points to perform stable smoothing / knee detection
        return result

    # ---- 计算 δCV，用于 χ² 评估 / compute delta_cv for chi-square ----
    delta_cv = np.array(
        [cv_uncertainty_rapidcode(ci, int(ni)) for ci, ni in zip(y, n)],
        dtype=float
    )
    has_valid_err = np.isfinite(delta_cv) & (delta_cv > 0)

    # ---- 选择 SG 参数 / choose SG parameters ----
    win_len, poly_order = _choose_sg_params(x.size)
    if win_len is None or poly_order is None:
        # 出现极端情况（点数太少）则直接返回默认 result
        return result

    # ---- 对原始 CV 曲线做 SG 平滑 / apply Savitzky–Golay smoothing ----
    try:
        y_smooth = savgol_filter(y, window_length=win_len, polyorder=poly_order)
    except Exception:
        # SG 失败就直接返回默认 result
        return result

    # ---- 在稠密网格上插值平滑曲线 / interpolate smoothed curve on dense grid ----
    x_dense = np.linspace(x.min(), x.max(), DENSE_N)
    # x_um 是等间距的，这里简单用线性插值即可
    y_dense = np.interp(x_dense, x, y_smooth)

    result["x_dense_um"] = x_dense
    result["cv_exp_dense"] = y_dense
    # 把 (window_length, polyorder) 存到 exp_params 里，方便以后追踪参数
    result["exp_params"] = np.array([win_len, poly_order], dtype=float)

    # ---- 在平滑曲线上找 knee 点 / knee on smoothed curve ----
    # 假设 CV(x) 随 x 单调减且“convex”（和之前 exp/1/x 设定一致）
    try:
        kl = KneeLocator(x_dense, y_dense,
                         curve="convex",
                         direction="decreasing",
                         online=False)
        if kl.knee is not None:
            result["knee_exp_um"] = float(kl.knee)
    except Exception:
        pass

    # ---- 用 δCV 评估平滑质量：reduced χ² / evaluate smoothing via reduced chi-square ----
    if np.any(has_valid_err):
        xv = x[has_valid_err]
        yv = y[has_valid_err]
        ev = delta_cv[has_valid_err]

        # 在原始 x 上的平滑值
        yfit = y_smooth[has_valid_err]

        chi2 = np.sum(((yv - yfit) / ev) ** 2)
        # 自由度：N - 有效参数个数（这里粗略按 2 个自由度来减：窗口+阶数）
        dof = max(len(yv) - 2, 1)
        result["chi2_red_exp"] = chi2 / dof

    # ---- hyper 分支不使用，置空 / hyper branch unused ----
    result["cv_hyp_dense"] = None
    result["knee_hyp_um"] = None
    result["chi2_red_hyper"] = np.nan
    result["hyp_params"] = None

    return result
