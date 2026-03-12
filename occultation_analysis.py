"""两台望远镜联合掩星检测示例（纯 Python，无第三方依赖）。

实现目标：
1. 生成 2 小时观测数据，信噪比 SNR=7，包含一个掩星事件；
2. 两台性能相同望远镜同时观测，基线距离 30 米；
3. 并行处理两台数据，生成光变曲线（CSV）；
4. 匹配两台数据中的同一光变下降信号并判断是否为掩星。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import csv
import math
import random
import statistics


@dataclass
class SimulationConfig:
    total_seconds: int = 2 * 3600
    cadence_seconds: float = 1.0
    snr: float = 7.0
    baseline_flux: float = 1.0
    telescope_separation_m: float = 30.0

    # 掩星参数
    event_start_s: float = 4100.0
    event_duration_s: float = 8.0
    event_depth: float = 0.20

    # 目标相对速度（用于将30m基线转为理论时间差）
    relative_shadow_speed_mps: float = 20000.0


@dataclass
class DetectionResult:
    dip_found_1: bool
    dip_found_2: bool
    dip_time_1: float
    dip_time_2: float
    lag_seconds: float
    max_allowed_lag: float
    judged_occultation: bool
    confidence_sigma: float


def build_time_axis(cfg: SimulationConfig) -> list[float]:
    n = int(cfg.total_seconds / cfg.cadence_seconds)
    return [i * cfg.cadence_seconds for i in range(n)]


def inject_occultation(
    flux: list[float],
    time_s: list[float],
    event_start_s: float,
    event_duration_s: float,
    event_depth: float,
) -> list[float]:
    out: list[float] = []
    event_end = event_start_s + event_duration_s
    factor = 1.0 - event_depth
    for t, f in zip(time_s, flux):
        out.append(f * factor if event_start_s <= t < event_end else f)
    return out


def simulate_telescope(
    time_s: list[float],
    cfg: SimulationConfig,
    delay_s: float,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    clean = [cfg.baseline_flux for _ in time_s]
    clean = inject_occultation(
        clean,
        time_s,
        event_start_s=cfg.event_start_s + delay_s,
        event_duration_s=cfg.event_duration_s,
        event_depth=cfg.event_depth,
    )
    noise_sigma = cfg.baseline_flux / cfg.snr
    return [f + rng.gauss(0.0, noise_sigma) for f in clean]


def moving_average(x: list[float], window: int = 9) -> list[float]:
    if window <= 1:
        return x[:]
    half = window // 2
    out: list[float] = []
    n = len(x)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        seg = x[start:end]
        out.append(sum(seg) / len(seg))
    return out


def detect_primary_dip(
    time_s: list[float],
    flux: list[float],
    smooth_window: int = 9,
) -> tuple[bool, float, float]:
    """检测持续性下降，而非单点极值。"""
    smoothed = moving_average(flux, smooth_window)
    baseline = statistics.median(smoothed)
    sigma = statistics.pstdev(smoothed)

    # 阈值：平滑后低于 baseline-2.5sigma 的点被视为可疑下降
    thr = baseline - 2.5 * (sigma + 1e-12)
    below = [v < thr for v in smoothed]

    segments: list[tuple[int, int]] = []
    i = 0
    n = len(below)
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j < n and below[j]:
            j += 1
        segments.append((i, j))
        i = j

    # 至少持续3个采样点才认为是候选掩星下降
    min_len = 3
    candidates = [(s, e) for s, e in segments if (e - s) >= min_len]

    if not candidates:
        min_idx, min_val = min(enumerate(smoothed), key=lambda p: p[1])
        dip_depth = baseline - min_val
        significance = dip_depth / (sigma + 1e-12)
        return False, time_s[min_idx], significance

    # 在候选段中选择平均亮度最低的一段
    best_s, best_e = min(candidates, key=lambda se: sum(smoothed[se[0]:se[1]]) / (se[1] - se[0]))
    center_idx = (best_s + best_e - 1) // 2
    mean_seg = sum(smoothed[best_s:best_e]) / (best_e - best_s)
    dip_depth = baseline - mean_seg
    # 持续N点后，噪声约降低 sqrt(N)
    significance = dip_depth / ((sigma + 1e-12) / math.sqrt(best_e - best_s))
    return True, time_s[center_idx], significance


def dip_stat_series(flux: list[float], window: int) -> tuple[list[float], float, float]:
    smoothed = moving_average(flux, window)
    baseline = statistics.median(smoothed)
    sigma = statistics.pstdev(smoothed) + 1e-12
    # 正值越大表示下降越显著
    stat = [(baseline - v) / sigma for v in smoothed]
    return stat, baseline, sigma


def judge_occultation(
    time_s: list[float],
    flux1: list[float],
    flux2: list[float],
    cfg: SimulationConfig,
) -> DetectionResult:
    window = max(3, int(round(cfg.event_duration_s / cfg.cadence_seconds)))

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(dip_stat_series, flux1, window)
        f2 = ex.submit(dip_stat_series, flux2, window)
        stat1, _b1, s1 = f1.result()
        stat2, _b2, s2 = f2.result()

    # 允许的索引时间差（包含一个采样点余量）
    max_lag = cfg.telescope_separation_m / cfg.relative_shadow_speed_mps + cfg.cadence_seconds
    max_lag_idx = max(1, int(round(max_lag / cfg.cadence_seconds)))

    best_score = -1e9
    best_i = 0
    best_j = 0
    n = len(time_s)
    for i in range(n):
        j0 = max(0, i - max_lag_idx)
        j1 = min(n, i + max_lag_idx + 1)
        # 在允许lag内找stat2最强点
        local_j = max(range(j0, j1), key=lambda j: stat2[j])
        score = stat1[i] + stat2[local_j]
        if score > best_score:
            best_score = score
            best_i = i
            best_j = local_j

    dip_time_1 = time_s[best_i]
    dip_time_2 = time_s[best_j]
    lag = abs(dip_time_1 - dip_time_2)

    # 基于窗口平均后的近似显著性
    sig1 = stat1[best_i] * math.sqrt(window)
    sig2 = stat2[best_j] * math.sqrt(window)

    dip_found_1 = sig1 >= 3.0
    dip_found_2 = sig2 >= 3.0
    same_event = lag <= max_lag
    judged = dip_found_1 and dip_found_2 and same_event

    return DetectionResult(
        dip_found_1=dip_found_1,
        dip_found_2=dip_found_2,
        dip_time_1=dip_time_1,
        dip_time_2=dip_time_2,
        lag_seconds=lag,
        max_allowed_lag=max_lag,
        judged_occultation=judged,
        confidence_sigma=min(sig1, sig2),
    )


def save_light_curves_csv(
    time_s: list[float],
    flux1: list[float],
    flux2: list[float],
    path: str = "light_curves.csv",
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "telescope_A_flux", "telescope_B_flux"])
        for t, a, b in zip(time_s, flux1, flux2):
            writer.writerow([f"{t:.3f}", f"{a:.8f}", f"{b:.8f}"])


def save_quicklook_svg(
    time_s: list[float],
    flux1: list[float],
    flux2: list[float],
    path: str = "light_curves.svg",
    width: int = 1200,
    height: int = 420,
) -> None:
    """输出一个简单 SVG 光变图，便于直接查看。"""

    pad_l, pad_r, pad_t, pad_b = 70, 20, 30, 50
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    y_min = min(min(flux1), min(flux2))
    y_max = max(max(flux1), max(flux2))
    if math.isclose(y_min, y_max):
        y_min -= 1e-6
        y_max += 1e-6

    x_min = time_s[0]
    x_max = time_s[-1]

    def xpix(t: float) -> float:
        return pad_l + (t - x_min) / (x_max - x_min) * plot_w

    def ypix(y: float) -> float:
        return pad_t + (y_max - y) / (y_max - y_min) * plot_h

    # 为减小文件大小，按固定步长抽样绘图
    stride = max(1, len(time_s) // 1500)
    idxs = range(0, len(time_s), stride)

    pts1 = " ".join(f"{xpix(time_s[i]):.2f},{ypix(flux1[i]):.2f}" for i in idxs)
    pts2 = " ".join(f"{xpix(time_s[i]):.2f},{ypix(flux2[i]):.2f}" for i in idxs)

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n')
        f.write('<rect width="100%" height="100%" fill="white"/>\n')
        # 坐标轴
        f.write(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="black"/>\n')
        f.write(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="black"/>\n')

        # 曲线
        f.write(f'<polyline fill="none" stroke="#1f77b4" stroke-width="1" points="{pts1}"/>\n')
        f.write(f'<polyline fill="none" stroke="#d62728" stroke-width="1" points="{pts2}"/>\n')

        # 文本
        f.write(f'<text x="{width/2:.0f}" y="20" text-anchor="middle" font-size="16">2小时光变曲线（SNR=7）</text>\n')
        f.write(f'<text x="{width/2:.0f}" y="{height-10}" text-anchor="middle" font-size="12">Time (s)</text>\n')
        f.write(
            f'<text x="20" y="{height/2:.0f}" transform="rotate(-90,20,{height/2:.0f})" text-anchor="middle" font-size="12">Flux</text>\n'
        )

        f.write('<rect x="930" y="40" width="12" height="3" fill="#1f77b4"/>\n')
        f.write('<text x="950" y="45" font-size="12">Telescope A</text>\n')
        f.write('<rect x="930" y="60" width="12" height="3" fill="#d62728"/>\n')
        f.write('<text x="950" y="65" font-size="12">Telescope B</text>\n')

        f.write('</svg>\n')


def main() -> None:
    cfg = SimulationConfig()
    time_s = build_time_axis(cfg)

    delay_s = cfg.telescope_separation_m / cfg.relative_shadow_speed_mps

    flux1 = simulate_telescope(time_s, cfg, delay_s=0.0, seed=42)
    flux2 = simulate_telescope(time_s, cfg, delay_s=delay_s, seed=2024)

    result = judge_occultation(time_s, flux1, flux2, cfg)

    save_light_curves_csv(time_s, flux1, flux2)
    save_quicklook_svg(time_s, flux1, flux2)

    print("=== 掩星检测报告 ===")
    print(f"观测时长: {cfg.total_seconds/3600:.1f} 小时, SNR={cfg.snr}")
    print(f"望远镜间距: {cfg.telescope_separation_m} m")
    print(f"望远镜A检测到下降: {result.dip_found_1}, 时间={result.dip_time_1:.2f} s")
    print(f"望远镜B检测到下降: {result.dip_found_2}, 时间={result.dip_time_2:.2f} s")
    print(f"两者时间差: {result.lag_seconds:.3f} s (阈值 <= {result.max_allowed_lag:.3f} s)")
    print(f"联合显著性(较小值): {result.confidence_sigma:.2f} sigma")
    print(f"是否判定为同一掩星事件: {'是' if result.judged_occultation else '否'}")
    print("光变曲线已保存: light_curves.csv, light_curves.svg")


if __name__ == "__main__":
    main()
