import numpy as np
import matplotlib.pyplot as plt

# Импорт функций генерации поля из вашего модуля
from generate_surface.WhiteNoise.white_noise_field_generator import (
    build_grids,
    gaussian_psd,
    generate_field_from_psd,
    compute_correlation_function
)

# ============================================================
# 1. ФУНКЦИИ ДЛЯ РАСЧЁТА DLVO
# ============================================================

def dlvo_energy_per_area(d, A_H, kappa, gamma, n0, kT=4.11e-21, d_min=1e-10):
    """
    Энергия DLVO на единицу площади [Дж/м^2] для двух плоских поверхностей.

    Формула:
        W(d) = -A/(12π d²) + (64 n0 kT γ² / κ) exp(-κ d)

    Параметры:
        d      : array_like – локальные расстояния между поверхностями [м]
        A_H    : float – константа Гамакера [Дж]
        kappa  : float – обратная дебаевская длина [1/м]
        gamma  : float – tanh(zeψ₀/(4kT)) безразмерный
        n0     : float – концентрация электролита [1/м³]
        kT     : float – тепловая энергия [Дж] (по умолчанию 4.11e-21 при 300 K)
        d_min  : float – минимальное расстояние (отсечка) [м]

    Возвращает:
        W      : ndarray – энергия на единицу площади [Дж/м²]
    """
    d = np.maximum(d, d_min)
    vdw = -A_H / (12 * np.pi * d**2)
    el = (64 * n0 * kT * gamma**2 / kappa) * np.exp(-kappa * d)
    return vdw + el


def compute_dlvo_energy(field, dx, dy, H0, A_H, kappa, gamma, n0,
                        kT=4.11e-21, d_min=1e-10, return_local=False):
    """
    Вычисляет полную энергию DLVO взаимодействия между шероховатой
    поверхностью (field) и идеально плоской поверхностью.

    Локальное расстояние:
        d(x,y) = H0 + field(x,y)

    Полная энергия на ячейку (период):
        E_tot = ∫∫ W(d(x,y)) dx dy

    Параметры:
        field       : 2D ndarray – высоты шероховатости [м]
        dx, dy      : float – шаги сетки [м]
        H0          : float – среднее расстояние между плоскостями [м]
        A_H         : float – константа Гамакера [Дж]
        kappa       : float – обратная дебаевская длина [1/м]
        gamma       : float – безразмерный параметр
        n0          : float – концентрация электролита [1/м³]
        kT          : float – тепловая энергия [Дж]
        d_min       : float – минимальное расстояние (отсечка) [м]
        return_local: bool – если True, возвращает также карту локальной энергии

    Возвращает:
        E_total     : float – полная энергия на ячейку [Дж]
        P_avg       : float – среднее давление (сила на единицу площади) [Па]
        W_local     : ndarray (опционально) – карта локальной энергии [Дж/м²]
    """
    d = H0 + field
    W_local = dlvo_energy_per_area(d, A_H, kappa, gamma, n0, kT, d_min)

    E_total = np.sum(W_local) * dx * dy
    P_avg = np.mean(W_local)

    if return_local:
        return E_total, P_avg, W_local
    else:
        return E_total, P_avg


def compute_dlvo_force(field, dx, dy, H0, A_H, kappa, gamma, n0,
                       kT=4.11e-21, d_min=1e-10, delta=1e-10):
    """
    Вычисляет силу взаимодействия как производную энергии по H0:
        F(H0) = - dE/dH0

    Параметры:
        delta : float – шаг для численного дифференцирования [м]

    Возвращает:
        F      : float – сила [Н] (положительная – отталкивание)
        E0     : float – энергия при H0 [Дж]
    """
    E_plus, _ = compute_dlvo_energy(field, dx, dy, H0 + delta,
                                    A_H, kappa, gamma, n0, kT, d_min)
    E_minus, _ = compute_dlvo_energy(field, dx, dy, H0 - delta,
                                     A_H, kappa, gamma, n0, kT, d_min)
    E0, _ = compute_dlvo_energy(field, dx, dy, H0,
                                A_H, kappa, gamma, n0, kT, d_min)

    F = -(E_plus - E_minus) / (2 * delta)
    return F, E0


# ============================================================
# 2. ОСНОВНОЙ СЦЕНАРИЙ
# ============================================================

def main():
    # ------------------------------------------------------------
    # 2.1 ПАРАМЕТРЫ ГЕНЕРАЦИИ ПОЛЯ
    # ------------------------------------------------------------
    N = 1024
    Lx, Ly = 1e-6, 1e-6  # 1 мкм × 1 мкм
    sigma_f = 3e-9  # 1 нм (амплитуда)
    xi = 1e-8  # 10 нм (корреляционная длина)
    H0 = 10e-9  # 10 нм

    # ------------------------------------------------------------
    # 2.2 ГЕНЕРАЦИЯ ПОЛЯ
    # ------------------------------------------------------------
    print("Генерация шероховатой поверхности...")
    X, Y, dx, dy, x, y, KX, KY, K, kx, ky = build_grids(N, Lx, Ly)

    S_gauss = gaussian_psd(K, sigma_f, xi)
    f_gauss, F_gauss, S_corr, var_psd = generate_field_from_psd(
        S_gauss, dx, dy, sigma_target=sigma_f, seed=42
    )

    print(f"  Размер сетки: {N}×{N}")
    # print(f"  Шаг сетки: dx = {dx*1e9:.3f} нм, dy = {dy*1e9:.3f} нм")
    print(f"  Шаг сетки: dx = {dx * 1e9:.1f} нм, dy = {dy * 1e9:.1f} нм")
    print(f"  Дисперсия поля: {np.var(f_gauss):.6e} м²")
    print(f"  Среднее поле: {np.mean(f_gauss):.6e} м")

    # ------------------------------------------------------------
    # 2.3 ПАРАМЕТРЫ DLVO (типичные для воды, 1:1 электролит)
    # ------------------------------------------------------------
    A_H = 1e-20          # Дж (константа Гамакера)
    kappa = 1e8          # 1/м (дебаевская длина 10 нм)
    gamma = 0.5          # безразмерный
    n0 = 1e24            # 1/м³ (концентрация ~1 мМ)
    kT = 4.11e-21        # Дж (при 300 К)
    d_min = 0.3e-9       # 0.3 нм (минимальное расстояние)

    # ------------------------------------------------------------
    # 2.4 РАСЧЁТ ЭНЕРГИИ ПРИ ФИКСИРОВАННОМ H0
    # ------------------------------------------------------------

    print("\nРасчёт энергии DLVO при H0 = {:.1f} нм...".format(H0*1e9))
    E_total, P_avg, W_local = compute_dlvo_energy(
        f_gauss, dx, dy, H0, A_H, kappa, gamma, n0,
        kT=kT, d_min=d_min, return_local=True
    )

    E_per_area = E_total / (Lx * Ly)

    print(f"  Энергия на ячейку ({Lx:.1f}×{Ly:.1f} м²): {E_total:.3e} Дж")
    print(f"  Энергия на единицу площади: {E_per_area:.3e} Дж/м²")
    print(f"  Среднее давление: {P_avg:.3e} Па")

    # ------------------------------------------------------------
    # 2.5 РАСЧЁТ ЗАВИСИМОСТИ ЭНЕРГИИ И СИЛЫ ОТ H0
    # ------------------------------------------------------------
    H0_range = np.linspace(2e-9, 30e-9, 100)  # 2–30 нм

    E_list = []
    F_list = []

    print("\nРасчёт зависимости E(H0) и F(H0)...")
    for H in H0_range:
        E, _ = compute_dlvo_energy(f_gauss, dx, dy, H,
                                   A_H, kappa, gamma, n0, kT, d_min)
        F, _ = compute_dlvo_force(f_gauss, dx, dy, H,
                                  A_H, kappa, gamma, n0, kT, d_min)
        E_list.append(E)
        F_list.append(F)

    E_list = np.array(E_list)
    F_list = np.array(F_list)

    # Нормировка на площадь для давления
    P_list = E_list / (Lx * Ly)  # давление (Па)

    # ------------------------------------------------------------
    # 2.6 ВИЗУАЛИЗАЦИЯ
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))

    # (a) Поверхность
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(f_gauss * 1e9, extent=[0, Lx, 0, Ly], origin='lower')
    ax1.set_title('Шероховатая поверхность (высота, нм)')
    ax1.set_xlabel('x [нм]')
    ax1.set_ylabel('y [нм]')
    plt.colorbar(im1, ax=ax1, label='Высота [нм]')

    # (b) Локальная энергия DLVO
    ax2 = fig.add_subplot(2, 2, 2)
    # Переводим энергию в мДж/м² для удобства (1 Дж/м² = 1000 мДж/м²)
    W_local_mJ = W_local * 1000
    im2 = ax2.imshow(W_local_mJ, extent=[0, Lx, 0, Ly], origin='lower')
    ax2.set_title(f'Локальная энергия DLVO при H0 = {H0 * 1e9:.1f} нм')
    ax2.set_xlabel('x [нм]')
    ax2.set_ylabel('y [нм]')
    plt.colorbar(im2, ax=ax2, label='Энергия [мДж/м²]')

    #           то же самое для kT/нм²
    # W_local_kT = W_local / kbT * 1e-18  # перевод в kT/нм²
    # im2 = ax2.imshow(W_local_kT, extent=[0, Lx_nm, 0, Ly_nm], origin='lower')
    # ax2.set_title(f'Локальная энергия DLVO при H0 = {H0 * 1e9:.1f} нм')
    # ax2.set_xlabel('x [нм]')
    # ax2.set_ylabel('y [нм]')
    # plt.colorbar(im2, ax=ax2, label='Энергия [kT/нм²]')
    # ============================================================
    # (c) Энергия: раздельные вклады
    # ============================================================
    ax3 = fig.add_subplot(2, 2, 3)

    # Пересчитываем энергию для гладкой поверхности при тех же параметрах
    f_smooth = np.zeros_like(f_gauss)

    # Для каждого расстояния считаем вклады
    E_vdw_smooth = []
    E_edl_smooth = []
    E_total_smooth = []
    E_vdw_rough = []
    E_edl_rough = []
    E_total_rough = []

    for D in H0_range:
        # Гладкая поверхность
        d_smooth = D + f_smooth
        W_vdw_smooth = -A_H / (12 * np.pi * np.maximum(d_smooth, d_min) ** 2)
        W_edl_smooth = (64 * n0 * kT * gamma * gamma / kappa) * np.exp(-kappa * d_smooth)
        W_total_smooth = W_vdw_smooth + W_edl_smooth
        E_vdw_smooth.append(np.sum(W_vdw_smooth) * dx * dy)
        E_edl_smooth.append(np.sum(W_edl_smooth) * dx * dy)
        E_total_smooth.append(np.sum(W_total_smooth) * dx * dy)

        # Шероховатая поверхность
        d_rough = D + f_gauss
        W_vdw_rough = -A_H / (12 * np.pi * np.maximum(d_rough, d_min) ** 2)
        W_edl_rough = (64 * n0 * kT * gamma * gamma / kappa) * np.exp(-kappa * d_rough)
        W_total_rough = W_vdw_rough + W_edl_rough
        E_vdw_rough.append(np.sum(W_vdw_rough) * dx * dy)
        E_edl_rough.append(np.sum(W_edl_rough) * dx * dy)
        E_total_rough.append(np.sum(W_total_rough) * dx * dy)

    E_vdw_smooth = np.array(E_vdw_smooth)
    E_edl_smooth = np.array(E_edl_smooth)
    E_total_smooth = np.array(E_total_smooth)
    E_vdw_rough = np.array(E_vdw_rough)
    E_edl_rough = np.array(E_edl_rough)
    E_total_rough = np.array(E_total_rough)

    # График энергии: шероховатая поверхность (вклады)
    ax3.plot(H0_range * 1e9, E_vdw_rough * 1e18, '--', color='green', linewidth=2, label='VdW (шерох.)')
    ax3.plot(H0_range * 1e9, E_edl_rough * 1e18, ':', color='blue', linewidth=2, label='EDL (шерох.)')
    ax3.plot(H0_range * 1e9, E_total_rough * 1e18, '-', color='black', linewidth=2, label='Total (шерох.)')
    # Для сравнения — гладкая поверхность (суммарная)
    ax3.plot(H0_range * 1e9, E_total_smooth * 1e18, '--', color='red', linewidth=1.5, label='Total (гладкая)',
             alpha=0.6)
    ax3.set_xlabel('H0 [нм]')
    ax3.set_ylabel('E [10⁻¹⁸ Дж]')
    ax3.set_title('Энергия взаимодействия (раздельные вклады)')
    ax3.grid(True)
    ax3.legend(fontsize=9)

    # ============================================================
    # (d) Сила: раздельные вклады
    # ============================================================
    ax4 = fig.add_subplot(2, 2, 4)

    # Численное дифференцирование для вкладов
    F_vdw_rough = -np.gradient(E_vdw_rough, H0_range)
    F_edl_rough = -np.gradient(E_edl_rough, H0_range)
    F_total_rough = -np.gradient(E_total_rough, H0_range)

    # Для гладкой поверхности
    F_total_smooth = -np.gradient(E_total_smooth, H0_range)

    ax4.plot(H0_range * 1e9, F_vdw_rough * 1e12, '--', color='green', linewidth=2, label='VdW (шерох.)')
    ax4.plot(H0_range * 1e9, F_edl_rough * 1e12, ':', color='blue', linewidth=2, label='EDL (шерох.)')
    ax4.plot(H0_range * 1e9, F_total_rough * 1e12, '-', color='black', linewidth=2, label='Total (шерох.)')
    ax4.plot(H0_range * 1e9, F_total_smooth * 1e12, '--', color='red', linewidth=1.5, label='Total (гладкая)',
             alpha=0.6)
    ax4.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax4.set_xlabel('H0 [нм]')
    ax4.set_ylabel('F [10⁻¹² Н]')
    ax4.set_title('Сила взаимодействия (раздельные вклады)')
    ax4.grid(True)
    ax4.legend(fontsize=9)

    plt.tight_layout()

    # # (c) Зависимость энергии от расстояния
    # ax3 = fig.add_subplot(2, 2, 3)
    # ax3.plot(H0_range * 1e9, E_list * 1e18, 'b-', linewidth=2, label='Шероховатая')
    # # Для сравнения — гладкая поверхность (field = 0)
    # E_smooth = [compute_dlvo_energy(np.zeros_like(f_gauss), dx, dy, H,
    #                                 A_H, kappa, gamma, n0, kT, d_min)[0] for H in H0_range]
    # ax3.plot(H0_range * 1e9, np.array(E_smooth) * 1e18, 'r--', linewidth=2, label='Гладкая')
    # ax3.set_xlabel('H0 [нм]')
    # ax3.set_ylabel('E [10⁻¹⁸ Дж]')
    # ax3.set_title('Энергия взаимодействия')
    # ax3.grid(True)
    # ax3.legend()
    #
    # # (d) Зависимость силы от расстояния
    # ax4 = fig.add_subplot(2, 2, 4)
    # ax4.plot(H0_range * 1e9, F_list * 1e12, 'b-', linewidth=2, label='Шероховатая')
    # F_smooth = [compute_dlvo_force(np.zeros_like(f_gauss), dx, dy, H,
    #                                A_H, kappa, gamma, n0, kT, d_min)[0] for H in H0_range]
    # ax4.plot(H0_range * 1e9, np.array(F_smooth) * 1e12, 'r--', linewidth=2, label='Гладкая')
    # ax4.axhline(y=0, color='k', linestyle=':', linewidth=1)
    # ax4.set_xlabel('H0 [нм]')
    # ax4.set_ylabel('F [10⁻¹² Н]')
    # ax4.set_title('Сила взаимодействия')
    # ax4.grid(True)
    # ax4.legend()
    # plt.tight_layout()


    # ------------------------------------------------------------
    # 2.7 ПОИСК ПОЛОЖЕНИЯ РАВНОВЕСИЯ (F = 0)
    # ------------------------------------------------------------
    # Находим индекс, где сила меняет знак
    idx_zero = np.where(np.diff(np.sign(F_list)))[0]
    if len(idx_zero) > 0:
        i = idx_zero[0]
        H_eq = H0_range[i] + (H0_range[i+1] - H0_range[i]) * (0 - F_list[i]) / (F_list[i+1] - F_list[i])
        print(f"\nПоложение равновесия (F=0): H_eq ≈ {H_eq*1e9:.1f} нм")
    else:
        print("\nПоложение равновесия не найдено (сила не меняет знак)")
    print("mean gap =", np.mean(D))
    print("min gap  =", np.min(D))
    print("max gap  =", np.max(D))
    plt.show()
if __name__ == "__main__":
    main()