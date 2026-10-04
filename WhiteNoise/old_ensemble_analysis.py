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
SEED_BASE = 100  # начальное значение seed, для каждой реализации увеличиваем на 1


# ============================================================
# ГЕНЕРАЦИЯ АНСАМБЛЯ
# ============================================================

def generate_ensemble(n_realizations, N, Lx, Ly, sigma_f, xi, seed_base=100):
    """
    Генерирует ансамбль реализаций гауссова поля.

    Возвращает:
        fields : list — список полей
        variances : list — дисперсии каждой реализации
        xi_measured : list — измеренные корреляционные длины
        xi_errors : list — относительные ошибки корреляционной длины
        acf_ensemble : 2D array — усреднённая АКФ (нормированная)
    """

    # Построение сеток (один раз для всех реализаций)
    X, Y, dx, dy, x, y, KX, KY, K, kx, ky = build_grids(N, Lx, Ly)

    # PSD (один раз для всех реализаций)
    S_gauss = gaussian_psd(K, sigma_f, xi)

    fields = []
    variances = []
    xi_measured_list = []
    xi_error_list = []
    acf_list = []

    print(f"Генерация ансамбля из {n_realizations} реализаций...")
    print("=" * 60)

    for i in range(n_realizations):
        seed = seed_base + i
        print(f"\nРеализация {i + 1}/{n_realizations}, seed = {seed}")

        # Генерация поля
        f, F, S_corr, _ = generate_field_from_psd(
            S_gauss, dx, dy, sigma_target=sigma_f, seed=seed
        )

        fields.append(f)

        # Дисперсия
        var = np.var(f)
        variances.append(var)
        print(f"  Дисперсия: {var:.6f}")

        # Корреляционная длина
        xi_meas, xi_err, passed = test_correlation_length(f, dx, xi, tolerance=0.10)
        xi_measured_list.append(xi_meas)
        xi_error_list.append(xi_err)
        print(f"  xi: {xi_meas:.6f} (ошибка {xi_err:.2f}%)")

        # АКФ (центрированная, нормированная)
        C, rx, _ = compute_correlation_function(f, dx, center=True)
        center_idx = len(rx) // 2
        C_norm = C / C[center_idx, center_idx]
        acf_list.append(C_norm)

    # Усреднение АКФ по ансамблю
    acf_ensemble = np.mean(acf_list, axis=0)

    return {
        'fields': fields,
        'variances': variances,
        'xi_measured': xi_measured_list,
        'xi_errors': xi_error_list,
        'acf_ensemble': acf_ensemble,
        'rx': rx,
        'dx': dx,
        'KX': KX,
        'KY': KY,
        'kx': kx,
        'ky': ky,
        'Lx': Lx,
        'Ly': Ly,
        'xi': xi,
        'sigma_f': sigma_f
    }


# ============================================================
# СТАТИСТИЧЕСКАЯ ОБРАБОТКА
# ============================================================

def print_statistics(results):
    """Выводит статистику по ансамблю."""

    variances = results['variances']
    xi_measured = results['xi_measured']
    xi_errors = results['xi_errors']

    print("\n" + "=" * 60)
    print("СТАТИСТИКА ПО АНСАМБЛЮ")
    print("=" * 60)

    # Дисперсия
    var_mean = np.mean(variances)
    var_std = np.std(variances)
    print(f"\nДисперсия поля:")
    print(f"  Среднее:  {var_mean:.6f} (цель: {results['sigma_f'] ** 2:.6f})")
    print(f"  Стандартное отклонение: {var_std:.6f}")
    print(f"  Относительная ошибка: {100.0 * abs(var_mean - results['sigma_f'] ** 2) / results['sigma_f'] ** 2:.2f}%")

    # Корреляционная длина
    xi_mean = np.mean(xi_measured)
    xi_std = np.std(xi_measured)
    print(f"\nКорреляционная длина xi:")
    print(f"  Среднее:  {xi_mean:.6f} (цель: {results['xi']:.6f})")
    print(f"  Стандартное отклонение: {xi_std:.6f}")
    print(f"  Относительная ошибка: {100.0 * abs(xi_mean - results['xi']) / results['xi']:.2f}%")

    # Процент реализаций, прошедших тест
    passed_count = sum(1 for err in xi_errors if abs(err) <= 10.0)
    print(f"\nПроцент реализаций с ошибкой xi < 10%: {100.0 * passed_count / len(xi_errors):.0f}%")


# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_ensemble_results(results):
    """Визуализация результатов ансамбля."""

    fields = results['fields']
    variances = results['variances']
    xi_measured = results['xi_measured']
    acf_ensemble = results['acf_ensemble']
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

    n_real = len(fields)

    fig = plt.figure(figsize=(16, 12))

    # ------------------------------------------------------------
    # 1. Первая и последняя реализации (для сравнения)
    # ------------------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(fields[0], extent=[0, Lx, 0, Ly], origin='lower', aspect='equal')
    ax1.set_title(f"Реализация 1\nVar = {variances[0]:.4f}, xi = {xi_measured[0]:.3f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(fields[-1], extent=[0, Lx, 0, Ly], origin='lower', aspect='equal')
    ax2.set_title(f"Реализация {n_real}\nVar = {variances[-1]:.4f}, xi = {xi_measured[-1]:.3f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.colorbar(im2, ax=ax2)

    # ------------------------------------------------------------
    # 2. Гистограмма дисперсий
    # ------------------------------------------------------------
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(variances, bins=10, edgecolor='black', alpha=0.7)
    ax3.axvline(sigma_f ** 2, color='red', linestyle='--', label=f'цель = {sigma_f ** 2:.3f}')
    ax3.axvline(np.mean(variances), color='blue', linestyle='-', label=f'среднее = {np.mean(variances):.3f}')
    ax3.set_xlabel("Дисперсия")
    ax3.set_ylabel("Частота")
    ax3.set_title("Распределение дисперсий")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # 4. Гистограмма корреляционных длин
    # ------------------------------------------------------------
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(xi_measured, bins=10, edgecolor='black', alpha=0.7)
    ax4.axvline(xi, color='red', linestyle='--', label=f'цель = {xi:.3f}')
    ax4.axvline(np.mean(xi_measured), color='blue', linestyle='-', label=f'среднее = {np.mean(xi_measured):.3f}')
    ax4.set_xlabel("Корреляционная длина xi")
    ax4.set_ylabel("Частота")
    ax4.set_title("Распределение корреляционных длин")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # 5. Усреднённая АКФ с доверительным интервалом
    # ------------------------------------------------------------
    ax5 = fig.add_subplot(2, 3, 5)

    # Теоретическая АКФ
    C_theory = np.exp(-rx ** 2 / (2.0 * xi ** 2))

    # Усреднённая АКФ
    center_idx = len(rx) // 2
    acf_center = acf_ensemble[center_idx, :]
    acf_center /= acf_center[center_idx]

    ax5.plot(rx, acf_center, 'b-', label=f'усреднённая ({N_REALIZATIONS} реализаций)')
    ax5.plot(rx, C_theory, 'r--', label='теоретическая')
    ax5.set_xlim(-4.0 * xi, 4.0 * xi)
    ax5.set_ylim(-0.1, 1.1)
    ax5.set_xlabel("r")
    ax5.set_ylabel("C(r) / C(0)")
    ax5.set_title("Усреднённая АКФ")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # 6. Средняя PSD
    # ------------------------------------------------------------
    ax6 = fig.add_subplot(2, 3, 6)

    # Вычисляем PSD из последней реализации (для примера)
    F_last = np.fft.fft2(fields[-1])
    PSD_last = np.abs(F_last) ** 2 / (N * N)
    PSD_shifted = np.fft.fftshift(PSD_last)

    im6 = ax6.imshow(
        np.log10(PSD_shifted + 1e-30),
        extent=[-np.max(np.abs(kx)), np.max(np.abs(kx)),
                -np.max(np.abs(ky)), np.max(np.abs(ky))],
        origin='lower',
        aspect='equal'
    )
    ax6.set_title("log10 PSD (последняя реализация)")
    ax6.set_xlabel("kx")
    ax6.set_ylabel("ky")
    plt.colorbar(im6, ax=ax6)

    plt.tight_layout()
    plt.show()


# ============================================================
# ОСНОВНОЙ СЦЕНАРИЙ
# ============================================================

if __name__ == "__main__":
    # Генерация ансамбля
    results = generate_ensemble(
        n_realizations=N_REALIZATIONS,
        N=N,
        Lx=Lx,
        Ly=Ly,
        sigma_f=sigma_f,
        xi=xi,
        seed_base=SEED_BASE
    )

    # Статистика
    print_statistics(results)

    # Визуализация
    plot_ensemble_results(results)