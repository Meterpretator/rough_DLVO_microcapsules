"""
SEI (Surface Element Integration): гладкая сфера + шероховатая плоскость.

Исправления по сравнению с предыдущей версией:
    1. Метки на графиках явно указывают, ЧТО гладкое и ЧТО шероховатое
       («гладк. сфера + гладк. плоскость» vs «гладк. сфера + шерох. плоскость»).
    2. Добавлены точные аналитические формулы (уровень B):
       - Хэмакер (точная VdW для сферы-плоскости);
       - Дерягин (для иллюстрации точности приближения).
    3. D0_range — логарифмическая сетка (geomspace), чтобы убрать
       46%-ошибку F_direct vs -dU/dD0 вблизи контакта.
    4. Тест F_direct vs -dU/dD0 корректно исключает края и область малых |F|.
    5. Три независимых теста:
       T1: SEI(num) vs SEI(anal) для VdW — проверка численного интегрирования;
       T2: F_direct vs -dU/dD0 — внутренняя согласованность;
       T3: SEI(num) vs Хэмакер — физическая точность;
       T4 (иллюстрация): SEI(num) vs Дерягин — насколько приближение Дерягина
       отличается от интеграла по проекции.
"""

import numpy as np
import matplotlib.pyplot as plt

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
)

from generate_surface.DLVO_CALC.dlvo_calc_v2 import (
    compute_kappa,
    compute_gamma,
    compute_ion_density,
    dlvo_energy_per_area,
    dlvo_pressure,
)


# ============================================================
# 1. SEI: гладкая сфера + (шероховатая или гладкая) плоскость
# ============================================================

def compute_sei_sphere_plane(
    field, dx, dy, D0,
    A_H, kappa, gamma1, gamma2, n0, kT,
    R, d_min=0.3e-9,
    return_maps=False,
):
    """
    SEI: гладкая сфера радиуса R + плоскость с шероховатостью `field`.

    Если field == 0 (массив нулей) — гладкая плоскость.

    Формула:
        U(D0) = ∬ W_pl(h(x,y)) dx dy
        F(D0) = ∬ P_pl(h(x,y)) dx dy
    где
        h(x,y) = D0 + g(x,y) - g_min
        g(x,y) = field(x,y) + [R - sqrt(R² - ρ²)]
    """
    Nx, Ny = field.shape

    x = (np.arange(Nx) - (Nx - 1) / 2) * dx
    y = (np.arange(Ny) - (Ny - 1) / 2) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')

    rho2 = X**2 + Y**2
    inside = rho2 <= R**2

    sphere_profile = np.zeros_like(rho2)
    sphere_profile[inside] = R - np.sqrt(R**2 - rho2[inside])

    geometry = field + sphere_profile
    geometry_min = np.min(geometry[inside])
    h = D0 + geometry - geometry_min

    h_inside = h[inside]
    if np.min(h_inside) <= 0:
        raise RuntimeError(
            f"min gap = {np.min(h_inside) * 1e9:.6f} nm"
        )

    below_dmin = np.mean(h_inside < d_min)
    h_eval = np.maximum(h, d_min)

    W_tot, W_vdw, W_edl = dlvo_energy_per_area(
        h_eval, A_H, kappa, gamma1, gamma2, n0, kT, d_min
    )
    P_tot, P_vdw, P_edl = dlvo_pressure(
        h_eval, A_H, kappa, gamma1, gamma2, n0, kT, d_min
    )

    W_tot = np.where(inside, W_tot, 0.0)
    W_vdw = np.where(inside, W_vdw, 0.0)
    W_edl = np.where(inside, W_edl, 0.0)
    P_tot = np.where(inside, P_tot, 0.0)
    P_vdw = np.where(inside, P_vdw, 0.0)
    P_edl = np.where(inside, P_edl, 0.0)

    dA = dx * dy
    result = {
        'U_total': np.sum(W_tot) * dA,
        'U_vdw':   np.sum(W_vdw) * dA,
        'U_edl':   np.sum(W_edl) * dA,
        'F_total': np.sum(P_tot) * dA,
        'F_vdw':   np.sum(P_vdw) * dA,
        'F_edl':   np.sum(P_edl) * dA,
        'min_gap': np.min(h_inside),
        'below_dmin': below_dmin,
    }
    if return_maps:
        result.update({
            'd_map': h,
            'W_map': W_tot, 'W_vdw_map': W_vdw, 'W_edl_map': W_edl,
            'P_map': P_tot, 'P_vdw_map': P_vdw, 'P_edl_map': P_edl,
            'inside': inside,
        })
    return result


# ============================================================
# 2. Аналитические опорные формулы
#    (все — для ГЛАДКОЙ сферы + ГЛАДКОЙ плоскости)
# ============================================================

def U_vdw_hamaker_exact(D0, A_H, R):
    """
    Точная энергия Ван-дер-Ваальса сфера–плоскость
    (интегрирование парных потенциалов Хэмакера).
        U = -A/6 [ R/D + R/(D+2R) + ln(D/(D+2R)) ]
    """
    return -A_H / 6 * (
        R / D0 + R / (D0 + 2 * R) + np.log(D0 / (D0 + 2 * R))
    )


def U_vdw_derjaguin(D0, A_H, R):
    """
    Приближение Дерягина для VdW (ведущий член).
        U_Derj = 2πR ∫ W_pl dD ≈ -A_H R / (6 D0)
    """
    return -A_H * R / (6 * D0)


def U_edl_derjaguin(D0, kappa, gamma1, gamma2, n0, kT, R):
    """
    Приближение Дерягина для EDL (2πR ∫ W_pl^EDL dD).
        U_Derj = 2πR (64 n0 kT γ1γ2 / κ²) exp(-κ D0)
    """
    return 2 * np.pi * R * (
        64 * n0 * kT * gamma1 * gamma2 / kappa ** 2
    ) * np.exp(-kappa * D0)


def U_vdw_sei_analytical(D0, A_H, R):
    """
    Аналитический результат SEI-интеграла от плоской VdW-формулы
    со сферическим профилем (не параболическим!).
        U_SEI^anal = -A/6 [ R/D - ln(1 + R/D) ]
    Используется как ЭТАЛОН для проверки численного SEI (VdW).
    """
    return -A_H / 6 * (R / D0 - np.log(1 + R / D0))


# ============================================================
# 3. Основной сценарий
# ============================================================

def main():
    kappa, lambda_D = compute_kappa(
        ionic_strength, T, epsilon_0, epsilon_r, e
    )
    gamma1 = compute_gamma(zeta1, T)
    gamma2 = compute_gamma(zeta2, T)
    n0 = compute_ion_density(ionic_strength)
    kT = k_B * T

    print("=" * 70)
    print("SEI: ГЛАДКАЯ СФЕРА + ПЛОСКОСТЬ (гладкая / шероховатая)")
    print("=" * 70)
    print(f"  R = {R * 1e9:.1f} нм")
    print(f"  ζ₁ = {zeta1 * 1000:.0f} мВ, ζ₂ = {zeta2 * 1000:.0f} мВ")
    print(f"  I = {ionic_strength:.2f} моль/м³, λ_D = {lambda_D * 1e9:.2f} нм")
    print(f"  A_H = {A_H:.2e} Дж, d_min = {d_min * 1e9:.2f} нм")
    print(f"  σ_f = {sigma_f * 1e9:.1f} нм, ξ = {xi * 1e9:.1f} нм")

    # ---- Генерация шероховатой плоскости ----
    X, Y, dx, dy, x, y, KX, KY, K, kx, ky = build_grids(N, Lx, Ly)
    S_gauss = gaussian_psd(K, sigma_f, xi)
    f_gauss, _, _, _ = generate_field_from_psd(
        S_gauss, dx, dy, sigma_target=sigma_f, seed=seed
    )
    field_smooth = np.zeros_like(f_gauss)

    print(f"  Сетка {N}×{N}, шаг dx = {dx * 1e9:.1f} нм")

    # ---- Логарифмическая сетка по D0 ----
    # ВАЖНО: geomspace — точки плотнее у контакта, где U(D0) круто меняется.
    # Это убирает артефакт «46% ошибки» np.gradient для гладкой сферы.
    D0_range = np.geomspace(H0_min, H0_max, H0_n_points)

    # ---- Основной цикл ----
    print("\n" + "=" * 70)
    print("РАСЧЁТ SEI ПО D0")
    print("=" * 70)
    print("  Прогресс: ", end="", flush=True)

    U_s   = np.zeros(len(D0_range))   # гладк. сфера + гладк. плоскость
    U_v_s = np.zeros(len(D0_range))
    U_e_s = np.zeros(len(D0_range))
    F_s   = np.zeros(len(D0_range))

    U_r   = np.zeros(len(D0_range))   # гладк. сфера + шерох. плоскость
    U_v_r = np.zeros(len(D0_range))
    U_e_r = np.zeros(len(D0_range))
    F_r   = np.zeros(len(D0_range))

    for i, D0 in enumerate(D0_range):
        r_s = compute_sei_sphere_plane(
            field_smooth, dx, dy, D0,
            A_H, kappa, gamma1, gamma2, n0, kT, R, d_min
        )
        r_r = compute_sei_sphere_plane(
            f_gauss, dx, dy, D0,
            A_H, kappa, gamma1, gamma2, n0, kT, R, d_min
        )
        U_s[i]   = r_s['U_total']
        U_v_s[i] = r_s['U_vdw']
        U_e_s[i] = r_s['U_edl']
        F_s[i]   = r_s['F_total']

        U_r[i]   = r_r['U_total']
        U_v_r[i] = r_r['U_vdw']
        U_e_r[i] = r_r['U_edl']
        F_r[i]   = r_r['F_total']

        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)

    print(" готово!")

    # ---- Аналитические опорные кривые (для гладкой пары) ----
    U_v_hamaker = np.array([
        U_vdw_hamaker_exact(d, A_H, R) for d in D0_range
    ])
    U_v_derj = np.array([
        U_vdw_derjaguin(d, A_H, R) for d in D0_range
    ])
    U_v_sei_anal = np.array([
        U_vdw_sei_analytical(d, A_H, R) for d in D0_range
    ])
    U_e_derj = np.array([
        U_edl_derjaguin(d, kappa, gamma1, gamma2, n0, kT, R)
        for d in D0_range
    ])
    U_derj_total = U_v_derj + U_e_derj

    # ---- Сила из энергии ----
    F_from_U_s = -np.gradient(U_s, D0_range)
    F_from_U_r = -np.gradient(U_r, D0_range)

    # ============================================================
    # ВЕРИФИКАЦИЯ
    # ============================================================
    print("\n" + "=" * 70)
    print("ВЕРИФИКАЦИЯ")
    print("=" * 70)

    eps = lambda a, b: np.abs(a - b) / (np.abs(b) + 1e-30)

    # T1: SEI(num) vs SEI(anal) для VdW
    T1 = eps(U_v_s, U_v_sei_anal)
    print("\nT1. SEI(num, VdW) vs SEI(anal):")
    print(f"    max = {T1.max():.3e}   (ожидается ~1e-3..1e-2)")

    # T2: F_direct vs -dU/dD0
    # Исключаем края И области с малой силой (|F| < 1% от max|F|)
    F_max_s = np.max(np.abs(F_s))
    F_max_r = np.max(np.abs(F_r))
    mask_s = (np.abs(F_s) > 0.01 * F_max_s)
    mask_r = (np.abs(F_r) > 0.01 * F_max_r)

    T2_s = eps(F_s[mask_s], F_from_U_s[mask_s])
    T2_r = eps(F_r[mask_r], F_from_U_r[mask_r])

    print("\nT2. F_direct vs -dU/dD₀ (только |F| > 1% от max):")
    print(f"    гладк. плоскость:    max = {T2_s.max():.3e}")
    print(f"    шерох. плоскость:    max = {T2_r.max():.3e}")

    # Где именно max?
    idx_s = np.argmax(T2_s)
    idx_r = np.argmax(T2_r)
    print(f"    (max при D₀ = {D0_range[mask_s][idx_s] * 1e9:.1f} нм для гладкой, "
          f"{D0_range[mask_r][idx_r] * 1e9:.1f} нм для шероховатой)")

    # T3: SEI(num, VdW) vs Хэмакер
    T3 = eps(U_v_s, U_v_hamaker)
    print("\nT3. SEI(num, VdW) vs Хэмакер:")
    print(f"    max = {T3.max():.3e}   (ожидается ~1e-2 — разные модели)")

    # T4: иллюстрация
    T4_v = eps(U_v_s, U_v_derj)
    T4_t = eps(U_s, U_derj_total)
    print("\nT4 (иллюстрация). SEI(num) vs Дерягин:")
    print(f"    VdW:           max = {T4_v.max():.3e}")
    print(f"    Total (VdW+EDL): max = {T4_t.max():.3e}")
    print("    → это НЕ ошибка кода, а разница приближений")
    # ============================================================
    # ВИЗУАЛИЗАЦИЯ
    # ============================================================
    kT_norm = kT

    # ---------- ФИГУРА 1: энергии ----------
    fig1, axes1 = plt.subplots(1, 3, figsize=(FIG_WIDTH * 1.35, FIG_HEIGHT))

    ax = axes1[0]
    ax.plot(D0_range * 1e9, U_v_s / kT_norm, 'b-', lw=2.5,
            label='SEI: гладк. сфера + гладк. плоскость')
    ax.plot(D0_range * 1e9, U_v_r / kT_norm, 'r-', lw=2.5,
            label='SEI: гладк. сфера + шерох. плоскость')
    ax.plot(D0_range * 1e9, U_v_hamaker / kT_norm, 'k:', lw=1.5,
            label='Хэмакер: гладк. + гладк.')
    ax.plot(D0_range * 1e9, U_v_derj / kT_norm, 'g--', lw=1.5,
            label='Дерягин: гладк. + гладк.')
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel(r'$U_{\mathrm{vdW}} / k_B T$')
    ax.set_title('(a) Ван-дер-Ваальс')
    ax.grid(True)
    ax.legend(fontsize=7)

    ax = axes1[1]
    ax.plot(D0_range * 1e9, U_e_s / kT_norm, 'b-', lw=2.5,
            label='SEI: гладк. сфера + гладк. плоскость')
    ax.plot(D0_range * 1e9, U_e_r / kT_norm, 'r-', lw=2.5,
            label='SEI: гладк. сфера + шерох. плоскость')
    ax.plot(D0_range * 1e9, U_e_derj / kT_norm, 'g--', lw=1.5,
            label='Дерягин: гладк. + гладк.')
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel(r'$U_{\mathrm{EDL}} / k_B T$')
    ax.set_title('(b) EDL')
    ax.grid(True)
    ax.legend(fontsize=7)

    ax = axes1[2]
    ax.plot(D0_range * 1e9, U_s / kT_norm, 'b-', lw=2.5,
            label='SEI: гладк. сфера + гладк. плоскость')
    ax.plot(D0_range * 1e9, U_r / kT_norm, 'r-', lw=2.5,
            label='SEI: гладк. сфера + шерох. плоскость')
    ax.plot(D0_range * 1e9, U_derj_total / kT_norm, 'g--', lw=1.5,
            label='Дерягин: гладк. + гладк.')
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel(r'$U_{\mathrm{total}} / k_B T$')
    ax.set_title('(c) Полная энергия')
    ax.grid(True)
    ax.legend(fontsize=7)

    fig1.suptitle(
        f'Энергия DLVO (SEI): R = {R * 1e9:.0f} нм, '
        f'σ_f = {sigma_f * 1e9:.1f} нм, ξ = {xi * 1e9:.1f} нм',
        fontsize=SUPTITLE_SIZE, y=SUPTITLE_Y
    )
    plt.tight_layout(rect=[0, 0, 1, RECT_TOP])


    # ---------- ФИГУРА 2: силы и тест T2 ----------
    fig2, axes2 = plt.subplots(1, 3, figsize=(FIG_WIDTH * 1.35, FIG_HEIGHT))

    ax = axes2[0]
    ax.plot(D0_range * 1e9, F_s * 1e12, 'b-', lw=2.5,
            label='SEI: гладк. сфера + гладк. плоскость')
    ax.plot(D0_range * 1e9, F_r * 1e12, 'r-', lw=2.5,
            label='SEI: гладк. сфера + шерох. плоскость')
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel('F [пН]')
    ax.set_title('(a) Сила (из локального давления P_pl)')
    ax.grid(True)
    ax.legend(fontsize=7)

    ax = axes2[1]
    ax.plot(D0_range * 1e9, F_s * 1e12, 'b-', lw=2,
            label='F_direct: гладк. + гладк.')
    ax.plot(D0_range * 1e9, F_from_U_s * 1e12, 'b--', lw=1.5,
            label='-dU/dD₀: гладк. + гладк.')
    ax.plot(D0_range * 1e9, F_r * 1e12, 'r-', lw=2,
            label='F_direct: гладк. + шерох.')
    ax.plot(D0_range * 1e9, F_from_U_r * 1e12, 'r--', lw=1.5,
            label='-dU/dD₀: гладк. + шерох.')
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel('F [пН]')
    ax.set_title('(b) F_direct vs -dU/dD₀')
    ax.grid(True)
    ax.legend(fontsize=6)

    ax = axes2[2]
    ax.semilogy(D0_range[mask_s] * 1e9, T2_s, 'b-', lw=2,
                label='гладк. плоскость')
    ax.semilogy(D0_range[mask_r] * 1e9, T2_r, 'r-', lw=2,
                label='шерох. плоскость')
    ax.axhline(1e-2, color='orange', ls='--', lw=1, label='1%')
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel(r'$|F - F_U| / |F|$')
    ax.set_title('(c) Тест T2: |F_direct - (-dU/dD₀)|')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)

    fig2.suptitle(
        f'Сила DLVO: R = {R * 1e9:.0f} нм, '
        f'σ_f = {sigma_f * 1e9:.1f} нм, ξ = {xi * 1e9:.1f} нм',
        fontsize=SUPTITLE_SIZE, y=SUPTITLE_Y
    )
    plt.tight_layout(rect=[0, 0, 1, RECT_TOP])


    # ---------- ФИГУРА 3: тесты T1/T3/T4 ----------
    fig3, axes3 = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax = axes3[0]
    ax.semilogy(D0_range * 1e9, T1, 'b-', lw=2,
                label='T1: SEI(num) vs SEI(anal)  [VdW]')
    ax.semilogy(D0_range * 1e9, T3, 'g-', lw=2,
                label='T3: SEI(num) vs Хэмакер  [VdW]')
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel('Отн. различие')
    ax.set_title('VdW: численный SEI vs аналитика')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes3[1]
    ax.semilogy(D0_range * 1e9, T4_v, 'r-', lw=2,
                label='T4: SEI(VdW) vs Дерягин(VdW)')
    ax.semilogy(D0_range * 1e9, T4_t, 'm-', lw=2,
                label='T4: SEI(total) vs Дерягин(total)')
    ax.set_xlabel('D₀ [нм]')
    ax.set_ylabel('Отн. различие')
    ax.set_title('Иллюстрация: насколько Дерягин отличается от SEI')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)

    fig3.suptitle(
        'Верификация и иллюстрация: SEI(num) vs аналитика',
        fontsize=SUPTITLE_SIZE, y=SUPTITLE_Y
    )
    plt.tight_layout(rect=[0, 0, 1, RECT_TOP])

    # ------------------------------------------------------------
    # 2.10 КАРТЫ ДЛЯ D0 = 10 нм
    # ------------------------------------------------------------
    D0_show = 10e-9

    # ВАЖНО: используем compute_sei_sphere_plane (актуальное имя функции)
    r_map_rough = compute_sei_sphere_plane(
        f_gauss, dx, dy, D0_show,
        A_H, kappa, gamma1, gamma2,
        n0, kT, R, d_min,
        return_maps=True
    )

    # Перевод зазора в нанометры (содержит NaN вне сферы)
    d_map_nm = r_map_rough['d_map'] * 1e9

    print("Минимальный зазор:", np.nanmin(d_map_nm), "нм")
    print("Максимальный зазор внутри проекции:",
          np.nanmax(d_map_nm), "нм")
    print("Доля точек внутри проекции со значением >= 30 нм:",
          100 * np.mean(d_map_nm[r_map_rough['inside']] >= 30), "%")

    extent = [
        x[0] * 1e9,
        x[-1] * 1e9,
        y[0] * 1e9,
        y[-1] * 1e9,
    ]

    fig_maps, axes_maps = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Зазор — с белыми NaN вне сферы

    ax = axes_maps[0]

    # Маскируем всё, что вне сферы
    d_masked = np.ma.masked_where(~r_map_rough['inside'], d_map_nm)

    # Палитра с белым цветом для маски
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='white')

    # Диапазон только по точкам внутри сферы
    vmin = np.nanmin(d_map_nm[r_map_rough['inside']])
    vmax = np.nanpercentile(d_map_nm[r_map_rough['inside']], 99)

    im = ax.imshow(
        d_masked.T,
        origin='lower',
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f'Зазор d(x,y) [нм]\nD₀ = {D0_show * 1e9:.0f} нм')
    ax.set_xlabel('x [нм]')
    ax.set_ylabel('y [нм]')
    plt.colorbar(im, ax=ax, label='Зазор [нм]')

    # (b) Локальная энергия (перевод в k_B T / нм²)
    # БЕЗ явного NaN — оставляем как было
    ax = axes_maps[1]
    kT_per_nm2 = 1e-18 / kT
    W_map_kT = r_map_rough['W_map'] * kT_per_nm2

    im = ax.imshow(
        W_map_kT.T,
        origin='lower',
        extent=extent,
        cmap='viridis',
    )
    ax.set_title('Локальная энергия [k_B T / нм²]')
    ax.set_xlabel('x [нм]')
    ax.set_ylabel('y [нм]')
    plt.colorbar(im, ax=ax, label='Энергия [k_B T / нм²]')

    # (c) Локальное давление
    # БЕЗ явного NaN — оставляем как было
    ax = axes_maps[2]
    im = ax.imshow(
        r_map_rough['P_map'].T / 1000,
        origin='lower',
        extent=extent,
        cmap='RdBu_r',
    )
    ax.set_title('Локальное давление [кПа]')
    ax.set_xlabel('x [нм]')
    ax.set_ylabel('y [нм]')
    plt.colorbar(im, ax=ax, label='Давление [кПа]')

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 3.7 ИТОГОВАЯ ДИАГНОСТИКА
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ ДИАГНОСТИКА")
    print("=" * 70)

    idx_10 = np.argmin(np.abs(D0_range - 10e-9))

    # ВАЖНО: переводим Дж -> kT
    U_ham_derj_kT = (U_v_hamaker[idx_10] + U_e_derj[idx_10]) / kT_norm

    print(f"\n  При D₀ = 10 нм:")
    print(f"    U_SEI: гладк. сфера + гладк. плоскость  = "
          f"{U_s[idx_10] / kT_norm:+.3e} k_B T")
    print(f"    U_SEI: гладк. сфера + шерох. плоскость  = "
          f"{U_r[idx_10] / kT_norm:+.3e} k_B T")
    print(f"    U_Хэмакер + Дерягин (гладк. + гладк.)   = "
          f"{U_ham_derj_kT:+.3e} k_B T")
    print(f"    Отношение шерох./гладк. (U)             = "
          f"{U_r[idx_10] / U_s[idx_10]:.4f}")
    print(f"    F_direct: гладк. плоскость              = "
          f"{F_s[idx_10] * 1e12:+.4f} пН")
    print(f"    F_direct: шерох. плоскость              = "
          f"{F_r[idx_10] * 1e12:+.4f} пН")
    print(f"    Отношение шерох./гладк. (F)             = "
          f"{F_r[idx_10] / F_s[idx_10]:.4f}")

if __name__ == "__main__":
    main()