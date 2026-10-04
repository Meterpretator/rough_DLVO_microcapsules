"""
old_ensemble_analysis.py

Генерация ансамбля из 10 реализаций гауссова поля.
Вычисление статистик: средняя дисперсия, средняя корреляционная длина,
усреднённая АКФ.
"""

import numpy as np
import matplotlib.pyplot as plt

# Импорт всех функций из вашего модуля
from generate_surface.WhiteNoise.white_noise_field_generator import (
    N, Lx, Ly, sigma_f, xi,
    build_grids,
    gaussian_psd,
    generate_field_from_psd,
    compute_correlation_function,
    test_correlation_length,
    test_correlation_shape
)

# ============================================================
# ПАРАМЕТРЫ АНСАМБЛЯ
# ============================================================

N_REALIZATIONS = 100
RANDOM_SEEDS = True


# ============================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: КОРРЕЛЯЦИОННАЯ ФУНКЦИЯ (с dx, dy)
# ============================================================

def compute_correlation_function(f, dx, dy, center=True):
    """
    Вычисляет АКФ через теорему Винера-Хинчина.
    """
    N = f.shape[0]
    F = np.fft.fft2(f)
    C = np.fft.ifft2(np.abs(F) ** 2).real
    C /= N * N

    if center:
        C = np.fft.fftshift(C)
        rx = (np.arange(N) - N // 2) * dx
        ry = (np.arange(N) - N // 2) * dy
    else:
        rx = np.arange(N) * dx
        ry = np.arange(N) * dy

    return C, rx, ry


# ============================================================
# ГЕНЕРАЦИЯ АНСАМБЛЯ
# ============================================================

def generate_ensemble(n_realizations, N, Lx, Ly, sigma_f, xi, random_seeds=True):
    """
    Генерирует ансамбль реализаций и собирает статистику.
    """

    # Построение сеток (один раз)
    X, Y, dx, dy, x, y, KX, KY, K, kx, ky = build_grids(N, Lx, Ly)
    S_gauss = gaussian_psd(K, sigma_f, xi)

    fields = []
    variances = []
    xi_measured_list = []
    xi_error_list = []

    # Для усреднения АКФ (ненормированные)
    acf_list = []

    # Для усреднения PSD
    psd_list = []

    used_seeds = []

    print(f"Генерация ансамбля из {n_realizations} реализаций...")
    print("=" * 60)

    for i in range(n_realizations):
        # Выбор seed
        if random_seeds:
            rng = np.random.default_rng()
            seed = rng.integers(0, 1e9)
        else:
            seed = 100 + i
        used_seeds.append(seed)

        print(f"\nРеализация {i + 1}/{n_realizations}, seed = {seed}")

        # Генерация поля
        f, F, S_corr, _ = generate_field_from_psd(
            S_gauss, dx, dy, sigma_target=sigma_f, seed=seed
        )

        fields.append(f)

        # Дисперсия (с ddof=1)
        var = np.var(f, ddof=1)
        variances.append(var)
        print(f"  Дисперсия: {var:.6f}")

        # Корреляционная длина
        xi_meas, xi_err, passed = test_correlation_length(f, dx, xi, tolerance=0.10)
        xi_measured_list.append(xi_meas)
        xi_error_list.append(xi_err)
        print(f"  xi: {xi_meas:.6f} (ошибка {xi_err:.2f}%)")

        # Ненормированная АКФ (сохраняем как есть)
        C, rx, _ = compute_correlation_function(f, dx, dy, center=True)
        acf_list.append(C)

        # PSD (физическая нормировка)
        F_fft = np.fft.fft2(f)
        PSD = np.abs(F_fft) ** 2 * (dx * dy) / (N * N)  # физическая PSD
        psd_list.append(PSD)

    # ============================================================
    # УСРЕДНЕНИЕ ПО АНСАМБЛЮ
    # ============================================================

    # 1. АКФ: усредняем ненормированные, затем нормируем
    acf_ensemble = np.mean(acf_list, axis=0)
    center_idx = len(rx) // 2
    acf_ensemble_norm = acf_ensemble / acf_ensemble[center_idx, center_idx]

    # 2. PSD: усредняем
    psd_ensemble = np.mean(psd_list, axis=0)

    # 3. Статистика дисперсии
    n = len(variances)
    var_mean = np.mean(variances)
    var_std = np.std(variances, ddof=1)
    var_sem = var_std / np.sqrt(n)
    var_ci_low = var_mean - 1.96 * var_sem
    var_ci_high = var_mean + 1.96 * var_sem

    # 4. Статистика корреляционной длины
    xi_mean = np.mean(xi_measured_list)
    xi_std = np.std(xi_measured_list, ddof=1)
    xi_sem = xi_std / np.sqrt(n)
    xi_ci_low = xi_mean - 1.96 * xi_sem
    xi_ci_high = xi_mean + 1.96 * xi_sem

    # 5. Процент PASS
    passed_count = sum(1 for err in xi_error_list if abs(err) <= 10.0)
    pass_percent = 100.0 * passed_count / n

    return {
        'fields': fields,
        'variances': variances,
        'xi_measured': xi_measured_list,
        'xi_errors': xi_error_list,
        'acf_ensemble_norm': acf_ensemble_norm,
        'psd_ensemble': psd_ensemble,
        'rx': rx,
        'dx': dx,
        'dy': dy,
        'used_seeds': used_seeds,
        'random_seeds': random_seeds,
        'var_mean': var_mean,
        'var_std': var_std,
        'var_sem': var_sem,
        'var_ci_low': var_ci_low,
        'var_ci_high': var_ci_high,
        'xi_mean': xi_mean,
        'xi_std': xi_std,
        'xi_sem': xi_sem,
        'xi_ci_low': xi_ci_low,
        'xi_ci_high': xi_ci_high,
        'pass_percent': pass_percent,
        'KX': KX, 'KY': KY,
        'kx': kx, 'ky': ky,
        'Lx': Lx, 'Ly': Ly,
        'xi': xi,
        'sigma_f': sigma_f
    }


# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_ensemble_results(results):
    """Визуализация с учётом всех исправлений."""

    fields = results['fields']
    variances = results['variances']
    xi_measured = results['xi_measured']
    acf_ensemble_norm = results['acf_ensemble_norm']
    psd_ensemble = results['psd_ensemble']
    rx = results['rx']
    Lx = results['Lx']
    Ly = results['Ly']
    xi = results['xi']
    sigma_f = results['sigma_f']
    KX = results['KX']
    KY = results['KY']
    kx = results['kx']
    ky = results['ky']
    dx = results['dx']
    dy = results['dy']

    n_real = len(fields)

    fig = plt.figure(figsize=(16, 14))

    # ------------------------------------------------------------
    # 1. Первая реализация
    # ------------------------------------------------------------
    ax1 = fig.add_subplot(3, 3, 1)
    im1 = ax1.imshow(fields[0], extent=[0, Lx, 0, Ly], origin='lower', aspect='equal')
    ax1.set_title(f"Реализация 1\nVar = {variances[0]:.4f}, xi = {xi_measured[0]:.3f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1)

    # ------------------------------------------------------------
    # 2. Последняя реализация
    # ------------------------------------------------------------
    ax2 = fig.add_subplot(3, 3, 2)
    im2 = ax2.imshow(fields[-1], extent=[0, Lx, 0, Ly], origin='lower', aspect='equal')
    ax2.set_title(f"Реализация {n_real}\nVar = {variances[-1]:.4f}, xi = {xi_measured[-1]:.3f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.colorbar(im2, ax=ax2)

    # ------------------------------------------------------------
    # 3. Гистограмма дисперсий
    # ------------------------------------------------------------
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.hist(variances, bins=15, edgecolor='black', alpha=0.7)
    ax3.axvline(sigma_f ** 2, color='red', linestyle='--', label=f'цель = {sigma_f ** 2:.3f}')
    ax3.axvline(results['var_mean'], color='blue', linestyle='-', label=f'среднее = {results["var_mean"]:.3f}')
    ax3.set_xlabel("Дисперсия")
    ax3.set_ylabel("Количество реализаций")
    ax3.set_title("Распределение дисперсий")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # 4. Гистограмма корреляционных длин
    # ------------------------------------------------------------
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.hist(xi_measured, bins=15, edgecolor='black', alpha=0.7)
    ax4.axvline(xi, color='red', linestyle='--', label=f'цель = {xi:.3f}')
    ax4.axvline(results['xi_mean'], color='blue', linestyle='-', label=f'среднее = {results["xi_mean"]:.3f}')
    ax4.set_xlabel("Корреляционная длина xi")
    ax4.set_ylabel("Количество реализаций")
    ax4.set_title("Распределение корреляционных длин")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # 5. Усреднённая АКФ (нормированная после усреднения)
    # ------------------------------------------------------------
    ax5 = fig.add_subplot(3, 3, 5)
    C_theory = np.exp(-rx ** 2 / (2.0 * xi ** 2))
    center_idx = len(rx) // 2
    acf_center = acf_ensemble_norm[center_idx, :]

    ax5.plot(rx, acf_center, 'b-', label='усреднённая (100 реализаций)')
    ax5.plot(rx, C_theory, 'r--', label='теоретическая')
    ax5.set_xlim(-4.0 * xi, 4.0 * xi)
    ax5.set_ylim(-0.1, 1.1)
    ax5.set_xlabel("r")
    ax5.set_ylabel("C(r) / C(0)")
    ax5.set_title("Усреднённая АКФ")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # 6. Усреднённая PSD (лог. масштаб)
    # ------------------------------------------------------------
    ax6 = fig.add_subplot(3, 3, 6)
    psd_shifted = np.fft.fftshift(psd_ensemble)
    im6 = ax6.imshow(
        np.log10(np.maximum(psd_shifted, 1e-30)),
        extent=[-np.max(np.abs(kx)), np.max(np.abs(kx)),
                -np.max(np.abs(ky)), np.max(np.abs(ky))],
        origin='lower',
        aspect='equal'
    )
    ax6.set_title("Усреднённая log10 PSD")
    ax6.set_xlabel("kx")
    ax6.set_ylabel("ky")
    plt.colorbar(im6, ax=ax6)

    # ------------------------------------------------------------
    # 7. Сравнение PSD: усреднённая vs теоретическая (сечение)
    # ------------------------------------------------------------
    ax7 = fig.add_subplot(3, 3, 7)
    # Сечение PSD вдоль kx при ky=0
    mid_y = len(ky) // 2
    psd_slice = psd_ensemble[mid_y, :]
    # Центрируем для сравнения
    psd_slice_shifted = np.fft.fftshift(psd_slice)
    kx_shifted = np.fft.fftshift(kx)

    # Теоретическая PSD (сечение)
    S_theory = gaussian_psd(np.sqrt(KX ** 2 + KY ** 2), sigma_f, xi)
    S_theory_slice = S_theory[mid_y, :]
    S_theory_slice_shifted = np.fft.fftshift(S_theory_slice)

    ax7.plot(kx_shifted, psd_slice_shifted, label='усреднённая (ансамбль)')
    ax7.plot(kx_shifted, S_theory_slice_shifted, 'r--', label='теоретическая')
    ax7.set_xlabel("kx")
    ax7.set_ylabel("PSD")
    ax7.set_title("Сечение PSD (ky=0)")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(-5, 5)

    # ------------------------------------------------------------
    # 8. Статистические сводки (текст)
    # ------------------------------------------------------------
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.axis('off')
    stats_text = (
        f"СТАТИСТИКА ПО {len(variances)} РЕАЛИЗАЦИЯМ\n"
        f"{'=' * 40}\n"
        f"Дисперсия:\n"
        f"  среднее = {results['var_mean']:.4f} (цель {sigma_f ** 2:.4f})\n"
        f"  std = {results['var_std']:.4f}\n"
        f"  SEM = {results['var_sem']:.4f}\n"
        f"  95% CI = [{results['var_ci_low']:.4f}, {results['var_ci_high']:.4f}]\n"
        f"\nКорреляционная длина xi:\n"
        f"  среднее = {results['xi_mean']:.4f} (цель {xi:.4f})\n"
        f"  std = {results['xi_std']:.4f}\n"
        f"  SEM = {results['xi_sem']:.4f}\n"
        f"  95% CI = [{results['xi_ci_low']:.4f}, {results['xi_ci_high']:.4f}]\n"
        f"\nПроцент PASS (ошибка xi < 10%):\n"
        f"  {results['pass_percent']:.1f}%"
    )
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace')

    # ------------------------------------------------------------
    # 9. Ошибка корреляционной длины по реализациям
    # ------------------------------------------------------------
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.plot(range(1, len(xi_measured) + 1), results['xi_errors'], 'o-')
    ax9.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax9.axhline(10, color='red', linestyle='--', label='+10%')
    ax9.axhline(-10, color='red', linestyle='--', label='-10%')
    ax9.set_xlabel("Номер реализации")
    ax9.set_ylabel("Ошибка xi, %")
    ax9.set_title("Индивидуальные ошибки xi")
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# ОСНОВНОЙ СЦЕНАРИЙ
# ============================================================

if __name__ == "__main__":
    results = generate_ensemble(
        n_realizations=N_REALIZATIONS,
        N=N,
        Lx=Lx,
        Ly=Ly,
        sigma_f=sigma_f,
        xi=xi,
        random_seeds=True
    )

    plot_ensemble_results(results)

    # Краткая выдача в консоль
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(
        f"Дисперсия: среднее = {results['var_mean']:.4f}, SEM = {results['var_sem']:.4f}, 95% CI = [{results['var_ci_low']:.4f}, {results['var_ci_high']:.4f}]")
    print(
        f"xi: среднее = {results['xi_mean']:.4f}, SEM = {results['xi_sem']:.4f}, 95% CI = [{results['xi_ci_low']:.4f}, {results['xi_ci_high']:.4f}]")
    print(f"Процент PASS: {results['pass_percent']:.1f}%")