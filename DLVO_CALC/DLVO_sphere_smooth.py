import numpy as np
import matplotlib.pyplot as plt

'''
СКРИПТ РАСЧИТЫВАЕТ ЭНЕРГИЮ И СИЛУ СФЕРЫ
ВЗАИМОДЕЙСТВУЮЩЕЙ С ШЕРОХОВАТОЙ ПЛОСКОСТЬЮ.
'''

# ============================================================
# ИМПОРТЫ РАНЕЕ НАПИСАННЫХ ФУНКЦИЙ
# ============================================================

# Генерация поля
from generate_surface.WhiteNoise.white_noise_field_generator import (
    build_grids,
    gaussian_psd,
    generate_field_from_psd,
)

from generate_surface.DLVO_CALC.dlvo_params import (
    k_B, e, epsilon_0, N_A,
    T, epsilon_r,
    A_H, zeta1, zeta2, ionic_strength, d_min,
    N, Lx, Ly, sigma_f, xi, seed,
    R,
    H0_min, H0_max, H0_n_points,
    SUPTITLE_Y, SUPTITLE_SIZE, RECT_TOP, FIG_WIDTH, FIG_HEIGHT,
    get_derived_params, print_params,
)
# DLVO функции и физические константы
from generate_surface.DLVO_CALC.dlvo_calc_v2 import (
    compute_kappa,
    compute_gamma,
    compute_ion_density,
    dlvo_energy_per_area,
    dlvo_pressure,
)


# ============================================================
# 1. DLVO ДЛЯ ГЛАДКОЙ СФЕРЫ И ШЕРОХОВАТОЙ ПЛОСКОСТИ
# ============================================================

def compute_dlvo_sphere_plane(
    field, dx, dy, D0,
    A_H, kappa, gamma1, gamma2,
    n0, kT, R,
    d_min=0.3e-9,
    return_maps=False
):
    """
    DLVO-взаимодействие сферической частицы
    с шероховатой плоскостью.

    Параметры
    ----------
    field : ndarray
        Карта шероховатости плоскости [м].
    dx, dy : float
        Шаг сетки [м].
    D0 : float
        Минимальный геометрический зазор между сферой
        и шероховатой плоскостью [м].
    R : float
        Радиус сферы [м].

    Возвращает
    ----------
    U_total, U_vdw, U_edl : энергия [Дж]
    F_total, F_vdw, F_edl : сила [Н]

    При return_maps=True дополнительно возвращаются:
        d_map
        W_map
        P_map
        и карты отдельных вкладов.
    """

    Nx, Ny = field.shape

    # ============================================================
    # 1. Сетка
    # ============================================================

    x = (np.arange(Nx) - (Nx - 1) / 2) * dx
    y = (np.arange(Ny) - (Ny - 1) / 2) * dy

    X, Y = np.meshgrid(x, y, indexing='ij')

    rho2 = X**2 + Y**2

    # Только проекция сферы
    inside = rho2 <= R**2

    # ============================================================
    # 2. Геометрия сферы
    # ============================================================

    sphere_profile = np.zeros_like(rho2)

    sphere_profile[inside] = (
        R - np.sqrt(R**2 - rho2[inside])
    )

    # ============================================================
    # 3. Общая локальная геометрия
    # ============================================================

    geometry = field + sphere_profile

    # ------------------------------------------------------------
    # Сдвигаем геометрию так, чтобы реальный минимальный
    # зазор внутри проекции сферы был равен D0
    # ------------------------------------------------------------

    geometry_min = np.min(geometry[inside])

    d = D0 + geometry - geometry_min

    # ============================================================
    # 4. Проверка геометрического зазора
    # ============================================================

    d_inside = d[inside]

    min_gap = np.min(d_inside)

    if min_gap <= 0:
        raise RuntimeError(
            f"Ошибка геометрии: min gap = "
            f"{min_gap * 1e9:.6f} nm"
        )

    # Доля области, где расстояние меньше d_min
    below_dmin_fraction = np.mean(d_inside < d_min)

    # ============================================================
    # 5. Минимальное расстояние для DLVO
    # ============================================================

    d_eval = np.maximum(d, d_min)

    # ============================================================
    # 6. Локальная энергия DLVO
    # ============================================================

    W_total, W_vdw, W_edl = dlvo_energy_per_area(
        d_eval,
        A_H,
        kappa,
        gamma1,
        gamma2,
        n0,
        kT,
        d_min
    )

    # ============================================================
    # 7. Локальное давление DLVO
    # ============================================================

    P_total, P_vdw, P_edl = dlvo_pressure(
        d_eval,
        A_H,
        kappa,
        gamma1,
        gamma2,
        n0,
        kT,
        d_min
    )

    # ============================================================
    # 8. Вне проекции сферы взаимодействия нет
    # ============================================================

    W_total = np.where(inside, W_total, 0.0)
    W_vdw   = np.where(inside, W_vdw,   0.0)
    W_edl   = np.where(inside, W_edl,   0.0)

    P_total = np.where(inside, P_total, 0.0)
    P_vdw   = np.where(inside, P_vdw,   0.0)
    P_edl   = np.where(inside, P_edl,   0.0)

    # ============================================================
    # 9. Интегрирование энергии
    # ============================================================

    dA = dx * dy

    U_total = np.sum(W_total) * dA
    U_vdw   = np.sum(W_vdw)   * dA
    U_edl   = np.sum(W_edl)   * dA

    # ============================================================
    # 10. Интегрирование силы
    # ============================================================

    F_total = np.sum(P_total) * dA
    F_vdw   = np.sum(P_vdw)   * dA
    F_edl   = np.sum(P_edl)   * dA

    # ============================================================
    # 11. Результат
    # ============================================================

    result = {
        # Энергия
        "U_total": U_total,
        "U_vdw": U_vdw,
        "U_edl": U_edl,

        # Сила
        "F_total": F_total,
        "F_vdw": F_vdw,
        "F_edl": F_edl,

        # Геометрия
        "min_gap": min_gap,
        "geometry_min": geometry_min,

        # Минимум исходной геометрии до нормировки
        "d_raw_min": np.min(
            (D0 + geometry)[inside]
        ),

        # Доля точек ниже d_min
        "below_dmin_fraction": below_dmin_fraction,
    }

    # ============================================================
    # 12. Карты
    # ============================================================

    if return_maps:

        result.update({

            # ----------------------------------------------------
            # Зазор
            # ----------------------------------------------------

            "d": d,
            "d_map": d,
            "d_eval": d_eval,

            # ----------------------------------------------------
            # Энергия на единицу площади
            # ----------------------------------------------------

            "W_total": W_total,
            "W_map": W_total,

            "W_vdw": W_vdw,
            "W_vdw_map": W_vdw,

            "W_edl": W_edl,
            "W_edl_map": W_edl,

            # ----------------------------------------------------
            # Давление
            # ----------------------------------------------------

            "P_total": P_total,
            "P_map": P_total,

            "P_vdw": P_vdw,
            "P_vdw_map": P_vdw,

            "P_edl": P_edl,
            "P_edl_map": P_edl,

            # ----------------------------------------------------
            # Геометрия
            # ----------------------------------------------------

            "inside": inside,
            "sphere_profile": sphere_profile,
            "geometry": geometry,
        })

    return result

def analytic_sphere_plane_smooth(D0, A_H, kappa, gamma1, gamma2, n0, kT, R):
    """
    Аналитическое приближение Дерягина для гладкой сферы и гладкой плоскости.

        F = 2πR · W_plane(D0)
        U = 2πR · ∫W_plane dD = 2πR · [ -A/(12π D0) + B/κ² · e^{-κ D0} ]
    """
    # Энергия на единицу площади (plane-plane)
    W_vdw = -A_H / (12 * np.pi * D0 ** 2)
    W_edl = (64 * n0 * kT * gamma1 * gamma2 / kappa) * np.exp(-kappa * D0)

    F_vdw = 2 * np.pi * R * W_vdw
    F_edl = 2 * np.pi * R * W_edl

    # Интеграл от W по d (от D0 до ∞)
    U_vdw = 2 * np.pi * R * (-A_H / (12 * np.pi * D0))
    U_edl = 2 * np.pi * R * (64 * n0 * kT * gamma1 * gamma2 / kappa ** 2) * np.exp(-kappa * D0)

    return {
        'F_total': F_vdw + F_edl, 'F_vdw': F_vdw, 'F_edl': F_edl,
        'U_total': U_vdw + U_edl, 'U_vdw': U_vdw, 'U_edl': U_edl,
    }


# ============================================================
# 2. ОСНОВНОЙ СЦЕНАРИЙ
# ============================================================

def main():
    # # ------------------------------------------------------------
    # # 2.1 ПАРАМЕТРЫ СЕТКИ
    # # ------------------------------------------------------------
    # N = 1024
    # Lx, Ly = 1e-6, 1e-6  # 1 мкм (сфера R = 320 нм свободно помещается)
    # sigma_f = 2e-9  # 2 нм (амплитуда шероховатости)
    # xi = 1e-8  # 10 нм (корреляционная длина)
    # seed = np.random.seed() #np.random.seed()  # генерация рандомной
    #
    # # ------------------------------------------------------------
    # # 2.2 ПАРАМЕТРЫ DLVO
    # # ------------------------------------------------------------
    # A_H = 1e-20  # Дж (константа Гамакера)
    # zeta1 = -25e-3  # -25 мВ (сфера)
    # zeta2 = -25e-3  # -25 мВ (плоскость)
    # ionic_strength = 1  # моль/м³ (1 мМ)
    # T = 298.0  # К
    # d_min = 0.3e-9  # 0.3 нм (отсечка)
    #
    # # ------------------------------------------------------------
    # # 2.3 ПАРАМЕТРЫ СФЕРЫ
    # # ------------------------------------------------------------
    # R = 300e-9  # 300 нм (радиус сферы)

    # ------------------------------------------------------------
    # 2.4 ПРОИЗВОДНЫЕ ПАРАМЕТРЫ
    # ------------------------------------------------------------
    kappa, lambda_D = compute_kappa(ionic_strength, T)
    gamma1 = compute_gamma(zeta1, T)
    gamma2 = compute_gamma(zeta2, T)
    n0 = compute_ion_density(ionic_strength)
    kT = k_B * T

    print("=" * 70)
    print("DLVO: ГЛАДКАЯ СФЕРА + ШЕРОХОВАТАЯ ПЛОСКОСТЬ")
    print("=" * 70)
    print(f"  Радиус сферы R = {R * 1e9:.1f} нм")
    print(f"  ζ₁ = {zeta1 * 1000:.0f} мВ, ζ₂ = {zeta2 * 1000:.0f} мВ")
    print(f"  γ₁ = {gamma1:.4f}, γ₂ = {gamma2:.4f}")
    print(f"  I = {ionic_strength:.2f} моль/м³")
    print(f"  κ = {kappa:.3e} 1/м, λ_D = {lambda_D * 1e9:.2f} нм")
    print(f"  A_H = {A_H:.2e} Дж, d_min = {d_min * 1e9:.2f} нм")
    print(f"  σ_f = {sigma_f * 1e9:.1f} нм, ξ = {xi * 1e9:.1f} нм")

    # ------------------------------------------------------------
    # 2.5 ГЕНЕРАЦИЯ ШЕРОХОВАТОЙ ПОВЕРХНОСТИ
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ГЕНЕРАЦИЯ ШЕРОХОВАТОЙ ПОВЕРХНОСТИ")
    print("=" * 70)

    X, Y, dx, dy, x, y, KX, KY, K, kx, ky = build_grids(N, Lx, Ly)

    S_gauss = gaussian_psd(K, sigma_f, xi)
    f_gauss, _, _, _ = generate_field_from_psd(
        S_gauss, dx, dy, sigma_target=sigma_f, seed=seed
    )

    print(f"  Размер сетки: {N}×{N}")
    print(f"  Шаг: dx = {dx * 1e9:.1f} нм, dy = {dy * 1e9:.1f} нм")
    print(f"  Размер ячейки: {Lx * 1e9:.1f}×{Ly * 1e9:.1f} нм²")
    print(f"  Дисперсия поля: {np.var(f_gauss):.3e} м²")

    # ------------------------------------------------------------
    # 2.6 ПРОВЕРКА: ГЛАДКАЯ СФЕРА vs АНАЛИТИКА ДЕРЯГИНА
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ГЛАДКОЙ СФЕРЫ (численный vs аналитика Дерягина)")
    print("=" * 70)

    field_smooth = np.zeros_like(f_gauss)
    D0_test = [10e-9, 15e-9, 20e-9, 30e-9]

    print(f"{'D0 [нм]':>10} {'F_num [пН]':>14} {'F_anal [пН]':>14} "
          f"{'rel_err_F':>12} {'U_num [аДж]':>14} {'U_anal [аДж]':>14} {'rel_err_U':>12}")

    for D0 in D0_test:
        r_num = compute_dlvo_sphere_plane(
            field_smooth, dx, dy, D0, A_H, kappa, gamma1, gamma2,
            n0, kT, R, d_min
        )
        r_anal = analytic_sphere_plane_smooth(
            D0, A_H, kappa, gamma1, gamma2, n0, kT, R
        )

        rel_err_F = abs(r_num['F_total'] - r_anal['F_total']) / (abs(r_anal['F_total']) + 1e-30)
        rel_err_U = abs(r_num['U_total'] - r_anal['U_total']) / (abs(r_anal['U_total']) + 1e-30)

        print(f"{D0 * 1e9:>10.1f} "
              f"{r_num['F_total'] * 1e12:>14.4f} "
              f"{r_anal['F_total'] * 1e12:>14.4f} "
              f"{rel_err_F:>12.2e} "
              f"{r_num['U_total'] * 1e18:>14.4f} "
              f"{r_anal['U_total'] * 1e18:>14.4f} "
              f"{rel_err_U:>12.2e}")

    # ------------------------------------------------------------
    # 2.7 РАСЧЁТ ЗАВИСИМОСТЕЙ ОТ D0
    # ------------------------------------------------------------
    D0_range = np.linspace(H0_min, H0_max, H0_n_points)

    U_rough = []
    F_rough = []
    U_smooth = []
    F_smooth = []
    U_anal = []
    F_anal = []

    print("\n" + "=" * 70)
    print("РАСЧЁТ ЗАВИСИМОСТЕЙ U(D0) И F(D0)")
    print("=" * 70)
    print("  Прогресс: ", end="", flush=True)

    for i, D0 in enumerate(D0_range):
        r_rough = compute_dlvo_sphere_plane(
            f_gauss, dx, dy, D0, A_H, kappa, gamma1, gamma2, n0, kT, R, d_min
        )
        r_smooth = compute_dlvo_sphere_plane(
            field_smooth, dx, dy, D0, A_H, kappa, gamma1, gamma2, n0, kT, R, d_min
        )
        r_anal = analytic_sphere_plane_smooth(
            D0, A_H, kappa, gamma1, gamma2, n0, kT, R
        )

        U_rough.append(r_rough['U_total'])
        F_rough.append(r_rough['F_total'])
        U_smooth.append(r_smooth['U_total'])
        F_smooth.append(r_smooth['F_total'])
        U_anal.append(r_anal['U_total'])
        F_anal.append(r_anal['F_total'])

        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)

    print(" готово!")

    U_rough = np.array(U_rough)
    F_rough = np.array(F_rough)
    U_smooth = np.array(U_smooth)
    F_smooth = np.array(F_smooth)
    U_anal = np.array(U_anal)
    F_anal = np.array(F_anal)

    # ------------------------------------------------------------
    # 2.8 ПОИСК ПОЛОЖЕНИЯ РАВНОВЕСИЯ
    # ------------------------------------------------------------
    def find_equilibrium(D0_arr, F_arr):
        idx = np.where(np.diff(np.sign(F_arr)))[0]
        if len(idx) > 0:
            i = idx[0]
            D_eq = D0_arr[i] + (D0_arr[i + 1] - D0_arr[i]) * (0 - F_arr[i]) / (F_arr[i + 1] - F_arr[i])
            return D_eq
        return np.nan

    D_eq_rough = find_equilibrium(D0_range, F_rough)
    D_eq_smooth = find_equilibrium(D0_range, F_smooth)
    D_eq_anal = find_equilibrium(D0_range, F_anal)

    print(f"\nПоложение равновесия:")
    print(f"  Шероховатая плоскость: {D_eq_rough * 1e9:.1f} нм" if not np.isnan(
        D_eq_rough) else "  Шероховатая плоскость: не найдено")
    print(f"  Гладкая (числ.):       {D_eq_smooth * 1e9:.1f} нм" if not np.isnan(
        D_eq_smooth) else "  Гладкая (числ.): не найдено")
    print(f"  Гладкая (аналит.):     {D_eq_anal * 1e9:.1f} нм" if not np.isnan(
        D_eq_anal) else "  Гладкая (аналит.): не найдено")

    # ------------------------------------------------------------
    # 2.9 ВИЗУАЛИЗАЦИЯ
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Сила
    ax = axes[0, 0]
    ax.plot(D0_range * 1e9, F_smooth * 1e12, 'b--', lw=2, label='Гладкая (числ.)')
    ax.plot(D0_range * 1e9, F_anal * 1e12, 'g:', lw=2, label='Гладкая (Дерягин)')
    ax.plot(D0_range * 1e9, F_rough * 1e12, 'r-', lw=2, label='Шероховатая (числ.)')
    ax.axhline(y=0, color='gray', linestyle=':', lw=1)
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel('F [пН]')
    ax.set_title(f'Сила DLVO: сфера (R={R * 1e9:.0f} нм) – плоскость')
    ax.grid(True)
    ax.legend()

    # (b) Энергия
    ax = axes[0, 1]
    ax.plot(D0_range * 1e9, U_smooth * 1e18, 'b--', lw=2, label='Гладкая (числ.)')
    ax.plot(D0_range * 1e9, U_anal * 1e18, 'g:', lw=2, label='Гладкая (Дерягин)')
    ax.plot(D0_range * 1e9, U_rough * 1e18, 'r-', lw=2, label='Шероховатая (числ.)')
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel('U [аДж]')
    ax.set_title('Энергия DLVO: сфера – плоскость')
    ax.grid(True)
    ax.legend()

    # (c) Относительная ошибка (числ. vs аналит.) для гладкой
    ax = axes[1, 0]
    rel_err_F = np.abs(F_smooth - F_anal) / (np.abs(F_anal) + 1e-30)
    rel_err_U = np.abs(U_smooth - U_anal) / (np.abs(U_anal) + 1e-30)
    ax.semilogy(D0_range * 1e9, rel_err_F, 'b-', lw=2, label='Сила')
    ax.semilogy(D0_range * 1e9, rel_err_U, 'r-', lw=2, label='Энергия')
    ax.axhline(y=0.05, color='gray', linestyle='--', lw=1, label='5% порог')
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel('Относительная ошибка')
    ax.set_title('Числ. vs аналит. (гладкая сфера)')
    ax.grid(True)
    ax.legend()

    # (d) Влияние шероховатости на силу
    ax = axes[1, 1]
    delta_F = F_rough - F_smooth
    ax.plot(D0_range * 1e9, delta_F * 1e12, 'm-', lw=2, label='ΔF = F_шерох − F_гладк')
    ax.axhline(y=0, color='gray', linestyle=':', lw=1)
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel('ΔF [пН]')
    ax.set_title('Вклад шероховатости в силу')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 2.10 КАРТЫ ДЛЯ D0 = 10 нм
    # ------------------------------------------------------------
    D0_show = 10e-9
    r_map_rough = compute_dlvo_sphere_plane(
        f_gauss, dx, dy, D0_show, A_H, kappa, gamma1, gamma2,
        n0, kT, R, d_min, return_maps=True
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # extent = [0, Lx * 1e9, 0, Ly * 1e9]
    extent = [
        x[0] * 1e9,
        x[-1] * 1e9,
        y[0] * 1e9,
        y[-1] * 1e9
    ]
    ax = axes[0]
    im = ax.imshow(r_map_rough['d_map'] * 1e9, origin='lower', extent=extent,
                   vmin=0, vmax=30, cmap='viridis')
    ax.set_title(f'Зазор d(x,y) [нм]\nD₀ = {D0_show * 1e9:.0f} нм')
    ax.set_xlabel('x [нм]')
    ax.set_ylabel('y [нм]')
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(r_map_rough['W_map'] * 1e18, origin='lower', extent=extent, cmap='RdBu_r')
    ax.set_title('Локальная энергия [аДж/нм²]')
    ax.set_xlabel('x [нм]')
    ax.set_ylabel('y [нм]')
    plt.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow(r_map_rough['P_map'] / 1000, origin='lower', extent=extent, cmap='RdBu_r')
    ax.set_title('Локальное давление [кПа]')
    ax.set_xlabel('x [нм]')
    ax.set_ylabel('y [нм]')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 2.11 ИТОГОВАЯ ДИАГНОСТИКА
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ ДИАГНОСТИКА")
    print("=" * 70)
    print(f"  R = {R * 1e9:.1f} нм")
    print(f"  σ_f = {sigma_f * 1e9:.1f} нм, ξ = {xi * 1e9:.1f} нм")
    print(f"  A_H = {A_H:.2e} Дж")
    print(f"  ζ₁ = {zeta1 * 1000:.0f} мВ, ζ₂ = {zeta2 * 1000:.0f} мВ")
    print(f"  I = {ionic_strength:.2f} моль/м³, λ_D = {lambda_D * 1e9:.2f} нм")
    print(f"  d_min = {d_min * 1e9:.2f} нм")
    print(f"  Положение равновесия (шерох.): {D_eq_rough * 1e9:.1f} нм" if not np.isnan(
        D_eq_rough) else "  Равновесие не найдено")
    print(f"  Положение равновесия (гладк.): {D_eq_smooth * 1e9:.1f} нм" if not np.isnan(
        D_eq_smooth) else "  Равновесие не найдено")


if __name__ == "__main__":
    main()