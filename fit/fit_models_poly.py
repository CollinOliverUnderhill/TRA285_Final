# fit_models_poly.py
import numpy as np
from kneed import KneeLocator

from config import DENSE_N
from sliding_stats import cv_uncertainty_rapidcode

# ========== 多项式拟合参数（你可以在这里改） ==========
# 拟合使用的多项式阶数，例如 2 = 二次，3 = 三次
POLY_DEG = 2

# 是否使用带权重拟合（权重 = 1 / delta_cv）
USE_WEIGHTED_FIT = True

# 最少需要的数据点数（避免点太少导致拟合不稳定）
MIN_POINTS = POLY_DEG + 2
# ==================================================


def fit_models_and_knees(x_um: np.ndarray,
                         cv: np.ndarray,
                         n_windows: np.ndarray):
    """
    使用多项式拟合 CV(x)，然后在拟合曲线上用 KneeLocator 找 knee 点。
    接口和原来的 fit_models.py 完全一致：
        - 函数名：fit_models_and_knees
        - 参数：x_um, cv, n_windows
        - 返回：一个 dict，包含以下键：
            x_dense_um, cv_exp_dense, cv_hyp_dense,
            knee_exp_um, knee_hyp_um,
            chi2_red_exp, chi2_red_hyper,
            exp_params, hyp_params

    这里只使用“exp这一路”的槽位来存多项式结果：
        - cv_exp_dense: 多项式拟合曲线
        - knee_exp_um:  多项式的 knee
        - chi2_red_exp: 多项式的 reduced χ²
        - exp_params:   多项式系数（np.ndarray，长度 = POLY_DEG+1）

      “hyper 那一套”全部置为 None / NaN，以保持接口，但不使用。
    """

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

    # 过滤掉 NaN / 非正 CV / 无窗口数 的点
    mask = np.isfinite(x_um) & np.isfinite(cv) & (cv > 0) & (n_windows > 0)
    x = x_um[mask]
    y = cv[mask]
    n = n_windows[mask].astype(float)

    if x.size < MIN_POINTS:
        # 点太少，不拟合
        return result

    # === 计算 CV 不确定度，用于权重和 χ² ===
    delta_cv = np.array(
        [cv_uncertainty_rapidcode(ci, int(ni)) for ci, ni in zip(y, n)],
        dtype=float
    )
    has_valid_err = np.isfinite(delta_cv) & (delta_cv > 0)

    # 拟合用的权重（np.polyfit 中 w ≈ 1/σ）
    if USE_WEIGHTED_FIT and np.any(has_valid_err):
        weights = np.zeros_like(delta_cv)
        weights[has_valid_err] = 1.0 / delta_cv[has_valid_err]
    else:
        weights = None

    # === 多项式拟合 ===
    try:
        if weights is not None:
            coeffs = np.polyfit(x, y, POLY_DEG, w=weights)
        else:
            coeffs = np.polyfit(x, y, POLY_DEG)
    except Exception:
        # 拟合失败就返回默认 result
        return result

    # 在稠密网格上评估多项式
    x_dense = np.linspace(x.min(), x.max(), DENSE_N)
    y_dense = np.polyval(coeffs, x_dense)

    result["x_dense_um"] = x_dense
    result["cv_exp_dense"] = y_dense
    result["exp_params"] = coeffs  # 长度 POLY_DEG+1 的数组

    # === 在多项式拟合曲线上找 knee 点 ===
    # 默认假设 CV 随窗口宽度增大是“convex + decreasing”的，
    # 和你之前 exp/1/x 的设定一致，如有需要可以改成 concave / increasing。
    try:
        kl = KneeLocator(x_dense, y_dense,
                         curve="convex",
                         direction="decreasing",
                         online=False)
        if kl.knee is not None:
            result["knee_exp_um"] = float(kl.knee)
    except Exception:
        pass

    # === 计算 reduced χ²，用 delta_cv 作为误差 ===
    if np.any(has_valid_err):
        xv = x[has_valid_err]
        yv = y[has_valid_err]
        ev = delta_cv[has_valid_err]

        yfit = np.polyval(coeffs, xv)
        chi2 = np.sum(((yv - yfit) / ev) ** 2)
        dof = max(len(yv) - len(coeffs), 1)  # N - 参数个数
        result["chi2_red_exp"] = chi2 / dof

    # hyper 路径不使用，保持接口但置空
    result["cv_hyp_dense"] = None
    result["knee_hyp_um"] = None
    result["chi2_red_hyper"] = np.nan
    result["hyp_params"] = None

    return result
