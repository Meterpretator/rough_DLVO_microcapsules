import numpy as np
import matplotlib.pyplot as plt

# Импорт функций построения сеток
from generate_surface.WhiteNoise.white_noise_field_generator import build_grids

# Импорт аналитических функций DLVO
from DLVO.norm_force_dlvo import (
    debye_length,
    gamma,
    vdw_plane_plane,
    edl_force,
    surface_energy,
    k, T, N_A, kbT
)

# ============================================================
# 1. ОБЩИЕ ПАРАМЕТРЫ
# ============================================================

# Параметры DLVO (из norm_force_dlvo.py)
ionic_strength = 1.0
zeta_potential_particle = -15e-3
zeta_potential_surface = -25e-3
A_H = 4.1e-21

# Вычисляем производные параметры
kappa_D = debye_length(ionic_strength, T)
gamma1 = gamma(zeta_potential_particle, T)
gamma2 = gamma(zeta_potential_surface, T)
n = 2 * ionic_strength * N_A

# Параметры сетки
N = 256
Lx, Ly = 64e-9, 64e-9
d_min = 0.3e-9

# Диапазон расстояний
D_range = np.linspace(1e-9, 30e-9, 100)
delta_H = D_range[1] - D_range[0]  # шаг для численного дифференцирования

# ============================================================
# 2. ОПРЕДЕЛЕНИЕ DLVO ФУНКЦИЙ (локально)
# ============================================================

def dlvo_energy_per_area(d, A_H, kappa, gamma1, gamma2, n0, kT=kbT, d_min=1e-10):
    """Энергия DLVO на единицу площади [Дж/м²]."""
    d = np.maximum(d, d_min)
    vdw = -A_H / (12 * np.pi * d**2)
    el = (64 * n0 * kT * gamma1 * gamma2 / kappa) * np.exp(-kappa * d)
    return vdw + el

def dlvo_pressure_analytical(d, A_H, kappa, gamma1, gamma2, n0, kT=kbT, d_min=1e-10):
    """
    Аналитическая производная -dW/dd [Па].
    Используется для проверки численной производной.
    """
    d = np.maximum(d, d_min)
    # dW/dd = A/(6π d^3) - (64 n0 kT γ1 γ2) * e^{-κd}
    dp_vdw = A_H / (6 * np.pi * d**3)
    dp_el = (64 * n0 * kT * gamma1 * gamma2) * np.exp(-kappa * d)
    return -dp_vdw + dp_el

def compute_dlvo_energy_numerical(field, dx, dy, H0, A_H, kappa, gamma1, gamma2, n0, kT=kbT, d_min=1e-10):
    """Численный расчёт энергии DLVO для заданной поверхности."""
    d = H0 + field
    W_local = dlvo_energy_per_area(d, A_H, kappa, gamma1, gamma2, n0, kT, d_min)
    E_total = np.sum(W_local) * dx * dy
    return E_total

# ============================================================
# 3. АНАЛИТИЧЕСКИЙ РАСЧЁТ (plane-plane)
# ============================================================

print("=" * 70)
print("АНАЛИТИЧЕСКИЙ РАСЧЁТ (plane-plane)")
print("=" * 70)

# Силы и энергии из аналитики
f_vdw_analytical = vdw_plane_plane(D_range, A_H)
f_edl_analytical = edl_force(D_range, n, kappa_D, gamma1, gamma2)
f_total_analytical = f_vdw_analytical + f_edl_analytical

# Энергия на единицу площади
W_vdw_analytical = surface_energy(f_vdw_analytical)
W_edl_analytical = surface_energy(f_edl_analytical)
W_total_analytical = W_vdw_analytical + W_edl_analytical

# Аналитическое давление (производная энергии)
P_analytical = dlvo_pressure_analytical(D_range, A_H, kappa_D, gamma1, gamma2, n)

print(f"  Debye length: {1/kappa_D*1e9:.2f} нм")
print(f"  gamma1 = {gamma1:.4f}, gamma2 = {gamma2:.4f}")
print(f"  n = {n:.2e} 1/м³")
print(f"  delta_H = {delta_H*1e9:.2f} нм")

# ============================================================
# 4. ПОСТРОЕНИЕ СЕТКИ И ГЛАДКОЙ ПОВЕРХНОСТИ
# ============================================================

print("\n" + "=" * 70)
print("ПОСТРОЕНИЕ СЕТКИ")
print("=" * 70)

X, Y, dx, dy, x, y, KX, KY, K, kx, ky = build_grids(N, Lx, Ly)
print(f"  Размер сетки: {N}×{N}")
print(f"  Шаг: dx = {dx*1e9:.2f} нм, dy = {dy*1e9:.2f} нм")
print(f"  Площадь ячейки: {Lx*1e9:.1f}×{Ly*1e9:.1f} нм²")

# Гладкая поверхность
f_smooth = np.zeros((N, N))

# ============================================================
# 5. ЧИСЛЕННЫЙ РАСЧЁТ (field = 0)
# ============================================================

print("\n" + "=" * 70)
print("ЧИСЛЕННЫЙ РАСЧЁТ (field = 0)")
print("=" * 70)

# Энергия на ячейку
E_numerical = []
for D in D_range:
    E = compute_dlvo_energy_numerical(f_smooth, dx, dy, D, A_H, kappa_D, gamma1, gamma2, n, kT=kbT, d_min=d_min)
    E_numerical.append(E)

E_numerical = np.array(E_numerical)

# Энергия на единицу площади
W_total_numerical = E_numerical / (Lx * Ly)

# Численная производная (сила на ячейку)
F_numerical = -np.gradient(E_numerical, D_range)
F_per_area_numerical = F_numerical / (Lx * Ly)

print(f"  Количество точек по D: {len(D_range)}")
print(f"  Энергия при H0=10 нм: {E_numerical[np.argmin(np.abs(D_range-10e-9))]:.3e} Дж")

# ============================================================
# 6. ТЕСТ 1: ЭНЕРГИЯ
# ============================================================

print("\n" + "=" * 70)
print("ТЕСТ 1: Сравнение энергии")
print("=" * 70)

idx_10nm = np.argmin(np.abs(D_range - 10e-9))

# Относительная ошибка для энергии на единицу площади
rel_error_energy = np.abs(W_total_analytical - W_total_numerical) / (np.abs(W_total_analytical) + 1e-30)

# Максимальная ошибка
max_rel_error_energy = np.max(rel_error_energy)

print(f"  При H0 = 10 нм:")
print(f"    W_analytical = {W_total_analytical[idx_10nm]:.6e} Дж/м²")
print(f"    W_numerical  = {W_total_numerical[idx_10nm]:.6e} Дж/м²")
print(f"    Относительная ошибка = {rel_error_energy[idx_10nm]:.2e}")

print(f"\n  Максимальная относительная ошибка по всему диапазону: {max_rel_error_energy:.2e}")

if max_rel_error_energy < 1e-10:
    print("  ✅ ТЕСТ 1 ПРОЙДЕН: энергия совпадает с аналитикой")
else:
    print(f"  ❌ ТЕСТ 1 НЕ ПРОЙДЕН: ошибка {max_rel_error_energy:.2e} > 1e-10")

# ============================================================
# 7. ТЕСТ 2: ЧИСЛЕННАЯ ПРОИЗВОДНАЯ
# ============================================================

print("\n" + "=" * 70)
print("ТЕСТ 2: Сравнение численной производной с аналитическим давлением")
print("=" * 70)

# Аналитическое давление (из производной локальной энергии)
P_analytical = dlvo_pressure_analytical(D_range, A_H, kappa_D, gamma1, gamma2, n)

# Относительная ошибка для давления
rel_error_pressure = np.abs(P_analytical - F_per_area_numerical) / (np.abs(P_analytical) + 1e-30)
max_rel_error_pressure = np.max(rel_error_pressure)

print(f"  При H0 = 10 нм:")
print(f"    P_analytical = {P_analytical[idx_10nm]:.6e} Па")
print(f"    P_numerical  = {F_per_area_numerical[idx_10nm]:.6e} Па")
print(f"    Относительная ошибка = {rel_error_pressure[idx_10nm]:.2e}")

print(f"\n  Максимальная относительная ошибка по всему диапазону: {max_rel_error_pressure:.2e}")

if max_rel_error_pressure < 1e-8:
    print("  ✅ ТЕСТ 2 ПРОЙДЕН: численная производная совпадает с аналитической")
else:
    print(f"  ⚠️ ТЕСТ 2: ошибка {max_rel_error_pressure:.2e} (зависит от шага дискретизации)")

# ============================================================
# 8. ТЕСТ 3: СИЛА (прямая аналитическая производная локального DLVO)
# ============================================================

print("\n" + "=" * 70)
print("ТЕСТ 3: Сравнение численной силы с аналитической (интегрированной)")
print("=" * 70)

# Аналитическая сила на ячейку = P_analytical * Lx * Ly
F_analytical_cell = P_analytical * (Lx * Ly)

# Относительная ошибка для силы
rel_error_force = np.abs(F_analytical_cell - F_numerical) / (np.abs(F_analytical_cell) + 1e-30)
max_rel_error_force = np.max(rel_error_force)

print(f"  При H0 = 10 нм:")
print(f"    F_analytical_cell = {F_analytical_cell[idx_10nm]:.6e} Н")
print(f"    F_numerical       = {F_numerical[idx_10nm]:.6e} Н")
print(f"    Относительная ошибка = {rel_error_force[idx_10nm]:.2e}")

print(f"\n  Максимальная относительная ошибка по всему диапазону: {max_rel_error_force:.2e}")

if max_rel_error_force < 1e-10:
    print("  ✅ ТЕСТ 3 ПРОЙДЕН: сила совпадает с аналитической")
else:
    print(f"  ❌ ТЕСТ 3 НЕ ПРОЙДЕН: ошибка {max_rel_error_force:.2e} > 1e-10")

# ============================================================
# 9. ИТОГОВЫЙ ВЕРДИКТ
# ============================================================

print("\n" + "=" * 70)
print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
print("=" * 70)

all_passed = (max_rel_error_energy < 1e-10) and (max_rel_error_force < 1e-10)

if all_passed:
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("   Численная реализация DLVO полностью согласована с аналитикой.")
    print("   Код готов для расчётов шероховатых поверхностей.")
else:
    print("⚠️ НЕ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("   Проверьте параметры нормировки и дискретизации.")

# ============================================================
# 10. ВИЗУАЛИЗАЦИЯ
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) Энергия на единицу площади
ax1 = axes[0, 0]
ax1.plot(D_range * 1e9, W_total_analytical, 'b-', lw=2, label='Аналитическая')
ax1.plot(D_range * 1e9, W_total_numerical, 'ro', ms=3, label='Численная')
ax1.set_xlabel('H0 [нм]')
ax1.set_ylabel('Энергия/площадь [Дж/м²]')
ax1.set_title('Тест 1: Энергия')
ax1.grid(True)
ax1.legend()

# (b) Относительная ошибка энергии
ax2 = axes[0, 1]
ax2.semilogy(D_range * 1e9, rel_error_energy, 'g-', lw=2)
ax2.axhline(y=1e-10, color='r', linestyle='--', label='Порог 1e-10')
ax2.set_xlabel('H0 [нм]')
ax2.set_ylabel('Относительная ошибка')
ax2.set_title('Тест 1: Ошибка энергии')
ax2.grid(True)
ax2.legend()

# (c) Давление (сила/площадь)
ax3 = axes[0, 2]
ax3.plot(D_range * 1e9, P_analytical, 'b-', lw=2, label='Аналитическая')
ax3.plot(D_range * 1e9, F_per_area_numerical, 'ro', ms=3, label='Численная')
ax3.axhline(y=0, color='gray', linestyle=':', lw=1)
ax3.set_xlabel('H0 [нм]')
ax3.set_ylabel('Давление [Па]')
ax3.set_title('Тест 2: Давление (сила/площадь)')
ax3.grid(True)
ax3.legend()

# (d) Относительная ошибка давления
ax4 = axes[1, 0]
ax4.semilogy(D_range * 1e9, rel_error_pressure, 'g-', lw=2)
ax4.axhline(y=1e-8, color='r', linestyle='--', label='Порог 1e-8')
ax4.set_xlabel('H0 [нм]')
ax4.set_ylabel('Относительная ошибка')
ax4.set_title('Тест 2: Ошибка давления')
ax4.grid(True)
ax4.legend()

# (e) Сила на ячейку
ax5 = axes[1, 1]
ax5.plot(D_range * 1e9, F_analytical_cell, 'b-', lw=2, label='Аналитическая')
ax5.plot(D_range * 1e9, F_numerical, 'ro', ms=3, label='Численная')
ax5.axhline(y=0, color='gray', linestyle=':', lw=1)
ax5.set_xlabel('H0 [нм]')
ax5.set_ylabel('Сила [Н]')
ax5.set_title('Тест 3: Сила')
ax5.grid(True)
ax5.legend()

# (f) Относительная ошибка силы
ax6 = axes[1, 2]
ax6.semilogy(D_range * 1e9, rel_error_force, 'g-', lw=2)
ax6.axhline(y=1e-10, color='r', linestyle='--', label='Порог 1e-10')
ax6.set_xlabel('H0 [нм]')
ax6.set_ylabel('Относительная ошибка')
ax6.set_title('Тест 3: Ошибка силы')
ax6.grid(True)
ax6.legend()

plt.suptitle('Тестирование численной реализации DLVO (field = 0)', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()