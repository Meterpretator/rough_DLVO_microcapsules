import numpy as np
import matplotlib.pyplot as plt

# Импорт функций генерации поля из вашего модуля
from generate_surface.WhiteNoise.white_noise_field_generator import (
    build_grids,
    gaussian_psd,
    generate_field_from_psd,
    compute_correlation_function
)
'''
скрипт расчета взаимодействия шерх. поверхн. с плоской.

'''
# # ============================================================
# # КОНСТАНТЫ
# # ============================================================
# k_B = 1.380649e-23  # Дж/К
# e = 1.602176634e-19  # Кл
# T_default = 298.0  # К
# epsilon_0 = 8.854187817e-12  # Ф/м
# epsilon_r = 78.5  # относительная диэлектрическая проницаемость воды
# N_A = 6.02214076e23  # 1/моль

from generate_surface.DLVO_CALC.dlvo_params import (
    k_B, e, epsilon_0, N_A,
    T, epsilon_r,
    A_H, zeta1, zeta2, ionic_strength, d_min,
    N, Lx, Ly, sigma_f, xi, H0, seed,
    H0_min, H0_max, H0_n_points,
    SUPTITLE_Y, SUPTITLE_SIZE, RECT_TOP, FIG_WIDTH, FIG_HEIGHT,
    get_derived_params, print_params,
)

# ============================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ DLVO
# ============================================================

def compute_kappa(ionic_strength, T):
    """
    Вычисляет обратную дебаевскую длину κ [1/м].

    κ = sqrt(2 e² N_A I / (ε₀ ε_r k_B T))
    """
    kappa = np.sqrt(
        (2 * e ** 2 * N_A * ionic_strength) / (epsilon_0 * epsilon_r * k_B * T)
    )
    return kappa, 1.0 / kappa


def compute_gamma(zeta_potential, T):
    """Вычисляет параметр gamma = tanh(e ζ / (4 k_B T))."""
    return np.tanh((e * zeta_potential) / (4 * k_B * T))


def compute_ion_density(ionic_strength):
    """Концентрация ионов [1/м³] для симметричного 1:1 электролита."""
    return  2 * N_A * ionic_strength # нужна ли тут 2??


# ============================================================
# 2. ФУНКЦИИ DLVO
# ============================================================

def dlvo_energy_per_area(d, A_H, kappa, gamma1, gamma2, n0, kT, d_min=0.3e-9):
    """
    Энергия DLVO на единицу площади [Дж/м²].

    W(d) = -A/(12πd²) + (64 n0 kT γ1 γ2 / κ) exp(-κd)
    """
    d = np.maximum(d, d_min)

    W_vdw = -A_H / (12 * np.pi * d ** 2)
    W_edl = (64 * n0 * kT * gamma1 * gamma2 / kappa) * np.exp(-kappa * d)
    W_total = W_vdw + W_edl

    return W_total, W_vdw, W_edl


def dlvo_pressure(d, A_H, kappa, gamma1, gamma2, n0, kT, d_min=0.3e-9):
    """
    Давление DLVO [Па].

    P(d) = -dW/dd = -A/(6πd³) + 64 n0 kT γ1γ2 e^{-κd}
    """
    d = np.maximum(d, d_min)

    P_vdw = -A_H / (6 * np.pi * d ** 3)
    P_edl = (64 * n0 * kT * gamma1 * gamma2) * np.exp(-kappa * d)
    P_total = P_vdw + P_edl

    return P_total, P_vdw, P_edl


def compute_dlvo_energy(field, dx, dy, H0, A_H, kappa, gamma1, gamma2, n0, kT,
                        d_min=0.3e-9, return_local=False):
    """
    Вычисляет полную энергию DLVO для шероховатой поверхности.

    Возвращает энергию в [Дж] и [kT/нм²].
    """
    d = np.maximum(H0 + field, d_min)

    W_local, W_vdw_local, W_edl_local = dlvo_energy_per_area(
        d, A_H, kappa, gamma1, gamma2, n0, kT, d_min
    )

    area = dx * dy * field.size

    E_vdw = np.sum(W_vdw_local) * dx * dy
    E_edl = np.sum(W_edl_local) * dx * dy
    E_total = np.sum(W_local) * dx * dy

    # Перевод в kT/нм²: 1 Дж/м² = 1e-18 / kT [kT/нм²]
    kT_per_nm2 = 1e-18 / kT

    result = {
        'E_total': E_total,
        'E_vdw': E_vdw,
        'E_edl': E_edl,
        'W_avg': E_total / area,
        'W_vdw_avg': E_vdw / area,
        'W_edl_avg': E_edl / area,
        'W_avg_kT_per_nm2': E_total / area * kT_per_nm2,
        'W_vdw_avg_kT_per_nm2': E_vdw / area * kT_per_nm2,
        'W_edl_avg_kT_per_nm2': E_edl / area * kT_per_nm2,
        'area': area,
        'd_min_used': np.min(d)
    }

    if return_local:
        result['W_local'] = W_local
        result['W_local_kT_per_nm2'] = W_local * kT_per_nm2
        result['W_vdw_local'] = W_vdw_local
        result['W_edl_local'] = W_edl_local

    return result


def compute_dlvo_force(field, dx, dy, H0, A_H, kappa, gamma1, gamma2, n0, kT,
                       d_min=0.3e-9, return_local=False):
    """
    Вычисляет силу DLVO для шероховатой поверхности через аналитическую производную.

    F = ∫[ -A/(6πd³) + 64 n0 kT γ1γ2 e^{-κd} ] dA

    Возвращает силу в [Н] и [нН].
    """
    d = np.maximum(H0 + field, d_min)

    # Аналитическая производная -dW/dd
    P_vdw = -A_H / (6 * np.pi * d ** 3)
    P_edl = (64 * n0 * kT * gamma1 * gamma2) * np.exp(-kappa * d)
    P_local = P_vdw + P_edl

    area = dx * dy * field.size

    F_vdw = np.sum(P_vdw) * dx * dy
    F_edl = np.sum(P_edl) * dx * dy
    F_total = np.sum(P_local) * dx * dy

    result = {
        'F_total': F_total,
        'F_vdw': F_vdw,
        'F_edl': F_edl,
        'F_total_nN': F_total * 1e9,
        'F_vdw_nN': F_vdw * 1e9,
        'F_edl_nN': F_edl * 1e9,
        'P_avg': np.mean(P_local),
        'P_vdw_avg': np.mean(P_vdw),
        'P_edl_avg': np.mean(P_edl),
        'area': area,
        'd_min_used': np.min(d)
    }

    if return_local:
        result['P_local'] = P_local
        result['P_local_kPa'] = P_local / 1000
        result['P_vdw_local'] = P_vdw
        result['P_edl_local'] = P_edl

    return result


def check_smooth_surface(field_shape, dx, dy, H0, A_H, kappa, gamma1, gamma2, n0, kT,
                         d_min=0.3e-9, tol=1e-10):
    """
    Проверка согласованности с гладкой поверхностью (field = 0).
    """
    field = np.zeros(field_shape)

    E_num = compute_dlvo_energy(field, dx, dy, H0, A_H, kappa, gamma1, gamma2, n0, kT, d_min)
    F_num = compute_dlvo_force(field, dx, dy, H0, A_H, kappa, gamma1, gamma2, n0, kT, d_min)

    W_anal, W_vdw_anal, W_edl_anal = dlvo_energy_per_area(
        H0, A_H, kappa, gamma1, gamma2, n0, kT, d_min
    )
    P_anal, P_vdw_anal, P_edl_anal = dlvo_pressure(
        H0, A_H, kappa, gamma1, gamma2, n0, kT, d_min
    )

    rel_error_W = np.abs(E_num['W_avg'] - W_anal) / (np.abs(W_anal) + 1e-30)
    rel_error_P = np.abs(F_num['P_avg'] - P_anal) / (np.abs(P_anal) + 1e-30)

    passed = (rel_error_W < tol) and (rel_error_P < tol)

    return {
        'passed': passed,
        'rel_error_W': rel_error_W,
        'rel_error_P': rel_error_P,
        'W_num': E_num['W_avg'],
        'W_anal': W_anal,
        'P_num': F_num['P_avg'],
        'P_anal': P_anal,
        'W_vdw_anal': W_vdw_anal,
        'W_edl_anal': W_edl_anal,
        'P_vdw_anal': P_vdw_anal,
        'P_edl_anal': P_edl_anal
    }


# ============================================================
# 3. ОСНОВНОЙ СЦЕНАРИЙ
# ============================================================

def main():
    # # ------------------------------------------------------------
    # # 3.1 ПАРАМЕТРЫ ГЕНЕРАЦИИ ПОЛЯ
    # # ------------------------------------------------------------
    # N = 1024
    # Lx, Ly = 1e-6, 1e-6  # 1 мкм × 1 мкм
    # sigma_f = 2e-9  # 2 нм (амплитуда шероховатости)
    # xi = 1e-8  # 10 нм (корреляционная длина)
    # H0 = 10e-9  # 10 нм (среднее расстояние)
    # seed = np.random.seed()  # генерация рандомной
    # # ------------------------------------------------------------
    # # 3.2 ФИЗИЧЕСКИЕ ПАРАМЕТРЫ DLVO
    # # ------------------------------------------------------------
    # A_H = 1e-20  # Дж (константа Гамакера)
    #
    # # Потенциалы поверхностей [В] (отрицательные для отрицательно заряженных)
    # zeta1 = -25e-3  # -25 мВ (частица/шероховатая поверхность)
    # zeta2 = -25e-3  # -25 мВ (плоскость)
    #
    # ionic_strength = 0.1  # моль/м³ (1 мМ)
    # T = 298.0  # К
    #
    # d_min = 0.3e-9  # 0.3 нм (минимальное расстояние)

    # ------------------------------------------------------------
    # 3.3 ВЫЧИСЛЕНИЕ ПРОИЗВОДНЫХ ПАРАМЕТРОВ
    # ------------------------------------------------------------
    kappa, lambda_D = compute_kappa(ionic_strength, T)
    gamma1 = compute_gamma(zeta1, T)
    gamma2 = compute_gamma(zeta2, T)
    n0 = compute_ion_density(ionic_strength)
    kT = k_B * T

    print("=" * 70)
    print("ПАРАМЕТРЫ DLVO")
    print("=" * 70)
    print(f"  zeta1 = {zeta1 * 1000:.1f} мВ, zeta2 = {zeta2 * 1000:.1f} мВ")
    print(f"  gamma1 = {gamma1:.4f}, gamma2 = {gamma2:.4f}")
    print(f"  ionic_strength = {ionic_strength:.6f} моль/м³ ({ionic_strength / 1000:.6f} М)")
    print(f"  κ = {kappa:.3e} 1/м, λ_D = {lambda_D * 1e9:.2f} нм")
    print(f"  n0 = {n0:.3e} 1/м³, kT = {kT:.3e} Дж")
    print(f"  A_H = {A_H:.2e} Дж, d_min = {d_min * 1e9:.2f} нм")

    # ------------------------------------------------------------
    # 3.4 ГЕНЕРАЦИЯ ПОЛЯ
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ГЕНЕРАЦИЯ ШЕРОХОВАТОЙ ПОВЕРХНОСТИ")
    print("=" * 70)

    X, Y, dx, dy, x, y, KX, KY, K, kx, ky = build_grids(N, Lx, Ly)

    S_gauss = gaussian_psd(K, sigma_f, xi)
    f_gauss, F_gauss, S_corr, var_psd = generate_field_from_psd(
        S_gauss, dx, dy, sigma_target=sigma_f, seed=seed
    )

    print(f"  Размер сетки: {N}×{N}")
    print(f"  Шаг сетки: dx = {dx * 1e9:.1f} нм, dy = {dy * 1e9:.1f} нм")
    print(f"  Размер ячейки: {Lx * 1e9:.1f}×{Ly * 1e9:.1f} нм²")
    print(f"  Дисперсия поля: {np.var(f_gauss):.6e} м²")
    print(f"  Среднее поле: {np.mean(f_gauss):.6e} м")

    # ------------------------------------------------------------
    # 3.5 ПРОВЕРКА ГЛАДКОЙ ПОВЕРХНОСТИ
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ГЛАДКОЙ ПОВЕРХНОСТИ (field = 0)")
    print("=" * 70)

    check = check_smooth_surface(
        f_gauss.shape, dx, dy, H0, A_H, kappa, gamma1, gamma2, n0, kT, d_min
    )

    print(f"  Относительная ошибка энергии: {check['rel_error_W']:.2e}")
    print(f"  Относительная ошибка давления: {check['rel_error_P']:.2e}")
    print(f"  Результат: {'✅ ПРОЙДЕНА' if check['passed'] else '❌ НЕ ПРОЙДЕНА'}")

    # ------------------------------------------------------------
    # 3.6 РАСЧЁТ ЭНЕРГИИ ПРИ ФИКСИРОВАННОМ H0
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"РАСЧЁТ ЭНЕРГИИ DLVO ПРИ H0 = {H0 * 1e9:.1f} нм")
    print("=" * 70)

    energy = compute_dlvo_energy(
        f_gauss, dx, dy, H0, A_H, kappa, gamma1, gamma2, n0, kT,
        d_min=d_min, return_local=True
    )

    print(f"  E_vdw   = {energy['E_vdw']:.3e} Дж")
    print(f"  E_edl   = {energy['E_edl']:.3e} Дж")
    print(f"  E_total = {energy['E_total']:.3e} Дж")
    print(f"  W_vdw_avg = {energy['W_vdw_avg_kT_per_nm2']:.3f} kT/нм²")
    print(f"  W_edl_avg = {energy['W_edl_avg_kT_per_nm2']:.3f} kT/нм²")
    print(f"  W_avg   = {energy['W_avg_kT_per_nm2']:.3f} kT/нм²")
    print(f"  d_min   = {energy['d_min_used'] * 1e9:.2f} нм")

    # ------------------------------------------------------------
    # 3.7 РАСЧЁТ СИЛЫ ПРИ ФИКСИРОВАННОМ H0
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"РАСЧЁТ СИЛЫ DLVO ПРИ H0 = {H0 * 1e9:.1f} нм")
    print("=" * 70)

    force = compute_dlvo_force(
        f_gauss, dx, dy, H0, A_H, kappa, gamma1, gamma2, n0, kT,
        d_min=d_min, return_local=True
    )

    print(f"  F_vdw   = {force['F_vdw_nN']:.3f} нН")
    print(f"  F_edl   = {force['F_edl_nN']:.3f} нН")
    print(f"  F_total = {force['F_total_nN']:.3f} нН")
    print(f"  P_vdw_avg = {force['P_vdw_avg']:.3e} Па")
    print(f"  P_edl_avg = {force['P_edl_avg']:.3e} Па")
    print(f"  P_avg   = {force['P_avg']:.3e} Па")
    print(f"  d_min   = {force['d_min_used'] * 1e9:.2f} нм")

    # ------------------------------------------------------------
    # 3.8 РАСЧЁТ ЗАВИСИМОСТЕЙ ОТ H0
    # ------------------------------------------------------------
    H0_range = np.linspace(H0_min, H0_max, H0_n_points)

    E_vdw_list = []
    E_edl_list = []
    E_total_list = []
    F_vdw_list = []
    F_edl_list = []
    F_total_list = []

    print("\n" + "=" * 70)
    print("РАСЧЁТ ЗАВИСИМОСТЕЙ E(H0) И F(H0)")
    print("=" * 70)
    print("  Прогресс: ", end="", flush=True)
    for i, H in enumerate(H0_range):
        energy_H = compute_dlvo_energy(f_gauss, dx, dy, H, A_H, kappa, gamma1, gamma2, n0, kT, d_min)
        force_H = compute_dlvo_force(f_gauss, dx, dy, H, A_H, kappa, gamma1, gamma2, n0, kT, d_min)

        E_vdw_list.append(energy_H['E_vdw'])
        E_edl_list.append(energy_H['E_edl'])
        E_total_list.append(energy_H['E_total'])

        F_vdw_list.append(force_H['F_vdw'])
        F_edl_list.append(force_H['F_edl'])
        F_total_list.append(force_H['F_total'])

        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)

    print(" готово!")

    E_vdw_list = np.array(E_vdw_list)
    E_edl_list = np.array(E_edl_list)
    E_total_list = np.array(E_total_list)
    F_vdw_list = np.array(F_vdw_list)
    F_edl_list = np.array(F_edl_list)
    F_total_list = np.array(F_total_list)

    # Перевод в удобные единицы
    E_vdw_kT = E_vdw_list / (Lx * Ly) * (1e-18 / kT)
    E_edl_kT = E_edl_list / (Lx * Ly) * (1e-18 / kT)
    E_total_kT = E_total_list / (Lx * Ly) * (1e-18 / kT)

    F_vdw_nN = F_vdw_list * 1e9
    F_edl_nN = F_edl_list * 1e9
    F_total_nN = F_total_list * 1e9

    # ------------------------------------------------------------
    # 3.9 ПОИСК ПОЛОЖЕНИЯ РАВНОВЕСИЯ
    # ============================================================
    idx_zero = np.where(np.diff(np.sign(F_total_list)))[0]
    if len(idx_zero) > 0:
        i = idx_zero[0]
        H_eq = H0_range[i] + (H0_range[i + 1] - H0_range[i]) * (0 - F_total_list[i]) / (
                    F_total_list[i + 1] - F_total_list[i])
        print(f"\nПоложение равновесия (F=0): H_eq ≈ {H_eq * 1e9:.1f} нм")
    else:
        H_eq = np.nan
        print("\nПоложение равновесия не найдено (сила не меняет знак)")

    # ------------------------------------------------------------
    # 3.10 ВИЗУАЛИЗАЦИЯ
    # ------------------------------------------------------------
    Lx_nm = Lx * 1e9
    Ly_nm = Ly * 1e9

    kT_per_nm2 = 1e-18 / kT   # перевод Дж/м² -> kT/нм²
    area_cell = Lx * Ly        # площадь ячейки [м²]

    # Единые параметры для всех suptitle
    SUPTITLE_Y = 0.98
    SUPTITLE_SIZE = 11
    RECT_TOP = 0.93

    # ============================================================
    # ФИГУРА 1: ВЕРИФИКАЦИЯ check_smooth_surface()
    # ============================================================
    #
    # Цель: показать, что численный расчёт для field = 0
    # совпадает с аналитическими формулами.
    #
    # ============================================================

    fig_ver, axes_ver = plt.subplots(1, 3, figsize=(16, 5.5))

    # ------------------------------------------------------------
    # (a) Полная энергия: аналитика vs численно
    # ------------------------------------------------------------
    ax = axes_ver[0]
    labels_energy = ['W_vdW', 'W_edl', 'W_total']

    w_anal_vals = [check['W_vdw_anal'], check['W_edl_anal'], check['W_anal']]
    w_num_vals  = [check['W_vdw_anal'], check['W_edl_anal'], check['W_num']]

    x_pos = np.arange(len(labels_energy))
    width = 0.35

    ax.bar(x_pos - width/2, w_anal_vals, width,
           label='Аналитика', color='steelblue')
    ax.bar(x_pos + width/2, w_num_vals, width,
           label='Численно (field=0)', color='orange', alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_energy)
    ax.set_ylabel('W [Дж/м²]')
    ax.set_title(f'Энергия при H0 = {H0*1e9:.1f} нм')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    # ------------------------------------------------------------
    # (b) Полное давление: аналитика vs численно
    # ------------------------------------------------------------
    ax = axes_ver[1]
    labels_pressure = ['P_vdW', 'P_edl', 'P_total']

    p_anal_vals = [check['P_vdw_anal'], check['P_edl_anal'], check['P_anal']]
    p_num_vals  = [check['P_vdw_anal'], check['P_edl_anal'], check['P_num']]

    ax.bar(x_pos - width/2, p_anal_vals, width,
           label='Аналитика', color='steelblue')
    ax.bar(x_pos + width/2, p_num_vals, width,
           label='Численно (field=0)', color='orange', alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_pressure)
    ax.set_ylabel('P [Па]')
    ax.set_title(f'Давление при H0 = {H0*1e9:.1f} нм')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    # ------------------------------------------------------------
    # (c) Относительные ошибки
    # ------------------------------------------------------------
    ax = axes_ver[2]
    err_labels = ['W_total', 'P_total']
    err_values = [check['rel_error_W'], check['rel_error_P']]

    ax.bar(err_labels, err_values, color='crimson', alpha=0.7)
    ax.axhline(y=1e-10, color='gray', linestyle='--', lw=1,
               label='Порог 1e-10')
    ax.set_yscale('log')
    ax.set_ylabel('|числ − аналит| / |аналит|')
    ax.set_title('Относительная ошибка')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    fig_ver.suptitle(
        f'Верификация: check_smooth_surface() при H0 = {H0 * 1e9:.1f} нм\n'
        f'ζ₁ = {zeta1 * 1000:.0f} мВ, ζ₂ = {zeta2 * 1000:.0f} мВ, '
        f'I = {ionic_strength:.2f} мМ',
        fontsize=SUPTITLE_SIZE, y=SUPTITLE_Y
    )
    plt.tight_layout(rect=[0, 0, 1, RECT_TOP])


    # ============================================================
    # ФИГУРА 2: ЭНЕРГИЯ И СИЛА ДЛЯ ГЛАДКОЙ ПОВЕРХНОСТИ
    # ============================================================
    #
    # Для гладкой поверхности используем аналитические формулы
    # (см. check_smooth_surface: численное совпадает с аналитикой
    # с точностью ~1e-15).
    #
    # ============================================================

    E_smooth_vdw_kT   = np.zeros_like(H0_range)
    E_smooth_edl_kT   = np.zeros_like(H0_range)
    E_smooth_total_kT = np.zeros_like(H0_range)
    F_smooth_vdw_nN   = np.zeros_like(H0_range)
    F_smooth_edl_nN   = np.zeros_like(H0_range)
    F_smooth_total_nN = np.zeros_like(H0_range)

    for i, H in enumerate(H0_range):
        W_total, W_vdw, W_edl = dlvo_energy_per_area(
            H, A_H, kappa, gamma1, gamma2, n0, kT, d_min
        )
        P_total, P_vdw, P_edl = dlvo_pressure(
            H, A_H, kappa, gamma1, gamma2, n0, kT, d_min
        )

        E_smooth_vdw_kT[i]   = W_vdw   * kT_per_nm2
        E_smooth_edl_kT[i]   = W_edl   * kT_per_nm2
        E_smooth_total_kT[i] = W_total * kT_per_nm2

        F_smooth_vdw_nN[i]   = P_vdw   * area_cell * 1e9
        F_smooth_edl_nN[i]   = P_edl   * area_cell * 1e9
        F_smooth_total_nN[i] = P_total * area_cell * 1e9

    fig_smooth, axes_sm = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Энергия гладкой поверхности
    ax = axes_sm[0]
    ax.plot(H0_range * 1e9, E_smooth_vdw_kT,   '--', color='green', lw=2, label='VdW')
    ax.plot(H0_range * 1e9, E_smooth_edl_kT,   ':',  color='blue',  lw=2, label='EDL')
    ax.plot(H0_range * 1e9, E_smooth_total_kT, '-',  color='black', lw=2, label='Total')
    ax.axhline(y=0, color='gray', linestyle=':', lw=1)
    ax.set_xlabel('H0 [нм]')
    ax.set_ylabel('E [kT/нм²]')
    ax.set_title('Энергия DLVO: гладкая поверхность')
    ax.grid(True)
    ax.legend()

    # (b) Сила гладкой поверхности
    ax = axes_sm[1]
    ax.plot(H0_range * 1e9, F_smooth_vdw_nN,   '--', color='green', lw=2, label='VdW')
    ax.plot(H0_range * 1e9, F_smooth_edl_nN,   ':',  color='blue',  lw=2, label='EDL')
    ax.plot(H0_range * 1e9, F_smooth_total_nN, '-',  color='black', lw=2, label='Total')
    ax.axhline(y=0, color='gray', linestyle=':', lw=1)
    ax.set_xlabel('H0 [нм]')
    ax.set_ylabel('F [нН]')
    ax.set_title('Сила DLVO: гладкая поверхность')
    ax.grid(True)
    ax.legend()

    fig_smooth.suptitle(
        f'Гладкая поверхность (field = 0)\n'
        f'ζ₁ = {zeta1*1000:.1f} мВ, ζ₂ = {zeta2*1000:.1f} мВ, '
        f'I = {ionic_strength:.2f} моль/м³',
        fontsize=SUPTITLE_SIZE, y=SUPTITLE_Y
    )
    plt.tight_layout(rect=[0, 0, 1, RECT_TOP])


    # ============================================================
    # ФИГУРА 3: СРАВНЕНИЕ ГЛАДКОЙ И ШЕРОХОВАТОЙ ПОВЕРХНОСТЕЙ
    # ============================================================
    #
    # Данные для шероховатой поверхности — из раздела 3.8:
    #   E_total_kT, F_total_nN
    # Данные для гладкой — из Фигуры 2 (аналитика).
    #
    # ============================================================

    fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Полная энергия: гладкая vs шероховатая
    ax = axes_cmp[0]
    ax.plot(H0_range * 1e9, E_smooth_total_kT, '--', color='red',   lw=2, label='Гладкая (аналит.)')
    ax.plot(H0_range * 1e9, E_total_kT,        '-',  color='black', lw=2, label='Шероховатая (числ.)')
    ax.axhline(y=0, color='gray', linestyle=':', lw=1)
    ax.set_xlabel('H0 [нм]')
    ax.set_ylabel('E [kT/нм²]')
    ax.set_title('Полная энергия: гладкая vs шероховатая')
    ax.grid(True)
    ax.legend()

    # (b) Полная сила: гладкая vs шероховатая
    ax = axes_cmp[1]
    ax.plot(H0_range * 1e9, F_smooth_total_nN, '--', color='red',   lw=2, label='Гладкая (аналит.)')
    ax.plot(H0_range * 1e9, F_total_nN,        '-',  color='black', lw=2, label='Шероховатая (числ.)')
    ax.axhline(y=0, color='gray', linestyle=':', lw=1)
    ax.set_xlabel('H0 [нм]')
    ax.set_ylabel('F [нН]')
    ax.set_title('Полная сила: гладкая vs шероховатая')
    ax.grid(True)
    ax.legend()

    fig_cmp.suptitle(
        f'Сравнение гладкой и шероховатой поверхностей\n'
        f'σ_f = {sigma_f*1e9:.1f} нм, ξ = {xi*1e9:.1f} нм',
        fontsize=SUPTITLE_SIZE, y=SUPTITLE_Y
    )
    plt.tight_layout(rect=[0, 0, 1, RECT_TOP])


    # ============================================================
    # ФИГУРА 4: КАРТЫ (поверхность, локальная энергия, давление)
    # ============================================================

    fig_maps = plt.figure(figsize=(18, 6))

    # (a) Шероховатая поверхность
    ax1 = fig_maps.add_subplot(1, 3, 1)
    im1 = ax1.imshow(f_gauss * 1e9, extent=[0, Lx_nm, 0, Ly_nm], origin='lower')
    ax1.set_title('Шероховатая поверхность (высота, нм)')
    ax1.set_xlabel('x [нм]')
    ax1.set_ylabel('y [нм]')
    plt.colorbar(im1, ax=ax1, label='Высота [нм]')

    # (b) Локальная энергия
    ax2 = fig_maps.add_subplot(1, 3, 2)
    im2 = ax2.imshow(energy['W_local_kT_per_nm2'],
                     extent=[0, Lx_nm, 0, Ly_nm], origin='lower')
    ax2.set_title(f'Локальная энергия DLVO\nH0 = {H0*1e9:.1f} нм')
    ax2.set_xlabel('x [нм]')
    ax2.set_ylabel('y [нм]')
    plt.colorbar(im2, ax=ax2, label='Энергия [kT/нм²]')

    # (c) Локальное давление
    ax3 = fig_maps.add_subplot(1, 3, 3)
    im3 = ax3.imshow(force['P_local_kPa'],
                     extent=[0, Lx_nm, 0, Ly_nm], origin='lower')
    ax3.set_title(f'Локальное давление DLVO\nH0 = {H0*1e9:.1f} нм')
    ax3.set_xlabel('x [нм]')
    ax3.set_ylabel('y [нм]')
    plt.colorbar(im3, ax=ax3, label='Давление [кПа]')

    plt.tight_layout()
    plt.show()
    # ------------------------------------------------------------
    # 3.11 ИТОГОВАЯ ДИАГНОСТИКА
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ ДИАГНОСТИКА")
    print("=" * 70)
    print(f"  Размер сетки: {N}×{N}")
    print(f"  Размер ячейки: {Lx * 1e9:.1f}×{Ly * 1e9:.1f} нм²")
    print(f"  sigma_f = {sigma_f * 1e9:.1f} нм, xi = {xi * 1e9:.1f} нм")
    print(f"  A_H = {A_H:.2e} Дж, κ = {kappa:.2e} 1/м")
    print(f"  gamma1 = {gamma1:.4f}, gamma2 = {gamma2:.4f}")
    print(f"  zeta1 = {zeta1 * 1000:.1f} мВ, zeta2 = {zeta2 * 1000:.1f} мВ")
    print(f"  d_min = {d_min * 1e9:.2f} нм")
    print(f"  Минимальный зазор в расчёте: {energy['d_min_used'] * 1e9:.2f} нм")
    print(f"  Положение равновесия: {H_eq * 1e9:.1f} нм")
    print(f"  Проверка гладкой поверхности: {'✅ ПРОЙДЕНА' if check['passed'] else '❌ НЕ ПРОЙДЕНА'}")


if __name__ == "__main__":
    main()