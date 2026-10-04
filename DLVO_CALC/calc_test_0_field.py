import numpy as np
import matplotlib.pyplot as plt

# Импорт ваших функций для генерации поля и расчёта DLVO
from generate_surface.WhiteNoise.white_noise_field_generator import (
    build_grids,
    gaussian_psd,
    generate_field_from_psd,
    compute_correlation_function
)

# Импорт аналитических функций DLVO из norm_force_dlvo.py
from DLVO.norm_force_dlvo import (
    debye_length,
    gamma,
    vdw_plane_plane,
    edl_force,
    surface_energy,
    k, T, e, epsilon_0, epsilon_r, N_A, kbT
)

# ============================================================
# 1. ОБЩИЕ ПАРАМЕТРЫ (должны совпадать в обоих подходах)
# ============================================================

# Параметры DLVO (из norm_force_dlvo.py)
ionic_strength = 1.0          # моль/м³ (1 мМ)
zeta_potential_particle = -15e-3   # В
zeta_potential_surface = -25e-3    # В
A_H = 4.1e-21                 # Дж (константа Гамакера)
R = 320e-9                    # м (радиус сферы, для справки)

# Вычисляем производные параметры DLVO
kappa_D = debye_length(ionic_strength, T)
gamma1 = gamma(zeta_potential_particle, T)
gamma2 = gamma(zeta_potential_surface, T)
n = 2 * ionic_strength * N_A   # 1/м³

# Параметры сетки (для численного расчёта)
N = 256                        # меньше для скорости теста
Lx, Ly = 64e-9, 64e-9          # 64 нм (маленькая область для теста)
sigma_f = 1e-9                 # 1 нм (амплитуда, но для field=0 не важно)
xi = 10e-9                     # 10 нм (корреляционная длина, не важно)

# Минимальное расстояние (отсечка)
d_min = 0.3e-9                 # 0.3 нм

# Диапазон расстояний для сравнения
D_range = np.linspace(1e-9, 30e-9, 100)  # 1–30 нм

# ============================================================
# 2. АНАЛИТИЧЕСКИЙ РАСЧЁТ (plane-plane из norm_force_dlvo.py)
# ============================================================

print("=" * 60)
print("1. АНАЛИТИЧЕСКИЙ РАСЧЁТ (plane-plane)")
print("=" * 60)

# Силы и энергии из аналитики
f_vdw_analytical = vdw_plane_plane(D_range, A_H)
f_edl_analytical = edl_force(D_range, n, kappa_D, gamma1, gamma2)
f_total_analytical = f_vdw_analytical + f_edl_analytical

# Энергия на единицу площади из аналитики
W_vdw_analytical = surface_energy(f_vdw_analytical)
W_edl_analytical = surface_energy(f_edl_analytical)
W_total_analytical = W_vdw_analytical + W_edl_analytical

print(f"  Debye length: {1/kappa_D*1e9:.2f} нм")
print(f"  gamma1 = {gamma1:.4f}, gamma2 = {gamma2:.4f}")
print(f"  n = {n:.2e} 1/м³")

# ============================================================
# 3. ЧИСЛЕННЫЙ РАСЧЁТ (field = 0)
# ============================================================

print("\n" + "=" * 60)
print("2. ЧИСЛЕННЫЙ РАСЧЁТ (field = 0, гладкая плоскость)")
print("=" * 60)

# Строим сетки
X, Y, dx, dy, x, y, KX, KY, K, kx, ky = build_grids(N, Lx, Ly)
print(f"  Размер сетки: {N}×{N}")
print(f"  Шаг сетки: dx = {dx*1e9:.2f} нм, dy = {dy*1e9:.2f} нм")

# Поле = 0 (гладкая поверхность)
f_smooth = np.zeros((N, N))

# Переопределяем функцию compute_dlvo_energy из вашего кода
# (если она уже есть в white_noise_field_generator.py, можно импортировать)
# Здесь я продублирую её для ясности

def dlvo_energy_per_area(d, A_H, kappa, gamma1, gamma2, n0, kT=kbT, d_min=1e-10):
    """
    Энергия DLVO на единицу площади [Дж/м^2].
    """
    d = np.maximum(d, d_min)
    vdw = -A_H / (12 * np.pi * d**2)
    el = (64 * n0 * kT * gamma1 * gamma2 / kappa) * np.exp(-kappa * d)
    return vdw + el

def compute_dlvo_energy_numerical(field, dx, dy, H0, A_H, kappa, gamma1, gamma2, n0, kT=kbT, d_min=1e-10):
    """
    Численный расчёт энергии DLVO для шероховатой поверхности.
    """
    d = H0 + field
    W_local = dlvo_energy_per_area(d, A_H, kappa, gamma1, gamma2, n0, kT, d_min)
    E_total = np.sum(W_local) * dx * dy
    return E_total

# Численный расчёт для каждого расстояния
E_numerical = []
for D in D_range:
    E = compute_dlvo_energy_numerical(f_smooth, dx, dy, D, A_H, kappa_D, gamma1, gamma2, n, kT=kbT, d_min=d_min)
    E_numerical.append(E)

E_numerical = np.array(E_numerical)

# Энергия на единицу площади (численная)
W_total_numerical = E_numerical / (Lx * Ly)

# Сила (численная производная)
F_numerical = -np.gradient(E_numerical, D_range)
F_per_area_numerical = F_numerical / (Lx * Ly)  # Н/м²

print(f"  Количество точек по D: {len(D_range)}")
print(f"  Энергия при H0=10 нм: {E_numerical[np.argmin(np.abs(D_range-10e-9))]:.3e} Дж")

# ============================================================
# 4. СРАВНЕНИЕ
# ============================================================

print("\n" + "=" * 60)
print("3. СРАВНЕНИЕ АНАЛИТИКИ И ЧИСЛЕННОГО РАСЧЁТА")
print("=" * 60)

# Относительная ошибка (при H0 = 10 нм)
idx_10nm = np.argmin(np.abs(D_range - 10e-9))
E_analytical_at_10nm = W_total_analytical[idx_10nm] * (Lx * Ly)
E_numerical_at_10nm = E_numerical[idx_10nm]

rel_error_energy = np.abs(E_analytical_at_10nm - E_numerical_at_10nm) / np.abs(E_analytical_at_10nm + 1e-30)
rel_error_pressure = np.abs(W_total_analytical[idx_10nm] - W_total_numerical[idx_10nm]) / np.abs(W_total_analytical[idx_10nm] + 1e-30)

print(f"\nПри H0 = 10 нм:")
print(f"  Аналитическая энергия на ячейку: {E_analytical_at_10nm:.3e} Дж")
print(f"  Численная энергия на ячейку:    {E_numerical_at_10nm:.3e} Дж")
print(f"  Относительная ошибка:           {rel_error_energy:.2e}")

print(f"\n  Аналитическая энергия/площадь: {W_total_analytical[idx_10nm]:.3e} Дж/м²")
print(f"  Численная энергия/площадь:    {W_total_numerical[idx_10nm]:.3e} Дж/м²")
print(f"  Относительная ошибка:         {rel_error_pressure:.2e}")

# Максимальная ошибка по всему диапазону
max_rel_error = np.max(np.abs(W_total_analytical - W_total_numerical) / (np.abs(W_total_analytical) + 1e-30))
print(f"\nМаксимальная относительная ошибка по всему диапазону: {max_rel_error:.2e}")

if max_rel_error < 1e-10:
    print("\n✅ ТЕСТ ПРОЙДЕН: численный и аналитический расчёты совпадают с машинной точностью.")
else:
    print(f"\n⚠️ ВНИМАНИЕ: ошибка {max_rel_error:.2e} > 1e-10. Проверьте параметры или реализацию.")

# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Энергия на единицу площади
ax1 = axes[0, 0]
ax1.plot(D_range * 1e9, W_total_analytical, 'b-', linewidth=2, label='Аналитическая')
ax1.plot(D_range * 1e9, W_total_numerical, 'ro', markersize=3, label='Численная (field=0)')
ax1.set_xlabel('H0 [нм]')
ax1.set_ylabel('Энергия/площадь [Дж/м²]')
ax1.set_title('Сравнение энергии (plane-plane)')
ax1.grid(True)
ax1.legend()

# (b) Относительная ошибка
ax2 = axes[0, 1]
rel_error_all = np.abs(W_total_analytical - W_total_numerical) / (np.abs(W_total_analytical) + 1e-30)
ax2.semilogy(D_range * 1e9, rel_error_all, 'g-', linewidth=2)
ax2.set_xlabel('H0 [нм]')
ax2.set_ylabel('Относительная ошибка')
ax2.set_title('Относительная ошибка численного расчёта')
ax2.grid(True)

# (c) Сила на единицу площади
ax3 = axes[1, 0]
ax3.plot(D_range * 1e9, f_total_analytical, 'b-', linewidth=2, label='Аналитическая')
ax3.plot(D_range * 1e9, F_per_area_numerical, 'ro', markersize=3, label='Численная (field=0)')
ax3.set_xlabel('H0 [нм]')
ax3.set_ylabel('Сила/площадь [Н/м²]')
ax3.set_title('Сравнение силы (plane-plane)')
ax3.grid(True)
ax3.legend()

# (d) Сила: вклад VdW и EDL
ax4 = axes[1, 1]
ax4.plot(D_range * 1e9, f_vdw_analytical, '--', color='green', label='VdW')
ax4.plot(D_range * 1e9, f_edl_analytical, ':', color='blue', label='EDL')
ax4.plot(D_range * 1e9, f_total_analytical, 'k-', linewidth=2, label='Total')
ax4.axhline(y=0, color='gray', linestyle=':', linewidth=1)
ax4.set_xlabel('H0 [нм]')
ax4.set_ylabel('Сила/площадь [Н/м²]')
ax4.set_title('Аналитический вклад сил (plane-plane)')
ax4.grid(True)
ax4.legend()

plt.tight_layout()


# ============================================================
# 6. ВЫВОД ПАРАМЕТРОВ
# ============================================================

print("\n" + "=" * 60)
print("4. ПАРАМЕТРЫ РАСЧЁТА")
print("=" * 60)
print(f"  Константа Гамакера A_H = {A_H:.2e} Дж")
print(f"  Обратная дебаевская длина κ = {kappa_D:.2e} 1/м")
print(f"  Дебаевская длина = {1/kappa_D*1e9:.2f} нм")
print(f"  gamma1 (частица) = {gamma1:.4f}")
print(f"  gamma2 (поверхность) = {gamma2:.4f}")
print(f"  Концентрация ионов n = {n:.2e} 1/м³")
print(f"  Минимальное расстояние d_min = {d_min*1e9:.2f} нм")
print(f"  Размер ячейки: {Lx*1e9:.0f}×{Ly*1e9:.0f} нм")
print(f"  Число точек N = {N}")

plt.show()