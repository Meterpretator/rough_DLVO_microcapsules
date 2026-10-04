import numpy as np
import matplotlib.pyplot as plt

from white_noise_field_generator import (
    build_grids,
    gaussian_psd,
    generate_field_from_psd,
    compute_correlation_function,
    test_correlation_length,
    test_correlation_shape,
    check_hermitian_symmetry,
    white_noise_psd
)
from generate_surface.WhiteNoise.white_noise_field_generator import plot_results

# ============================================================
# 1. ПАРАМЕТРЫ
# ============================================================

N = 1024
Lx = 64.0
Ly = 64.0
sigma_f = 1.0
xi = 1.0
# np.random.seed(42)

def main():
    X, Y, dx, dy, x, y, KX, KY, K, kx, ky = build_grids(N, Lx, Ly)

    # БЕЛЫЙ ШУМ
    print("\n" + "=" * 60)
    print("БЕЛЫЙ ШУМ")
    print("=" * 60)

    S_white = white_noise_psd(K, sigma_f)
    f_white, F_white, S_white_corr, _ = generate_field_from_psd(
        S_white, dx, dy, sigma_target=sigma_f, seed=42
    )

    print(f"Среднее поля:     {np.mean(f_white):.8e}")
    print(f"Дисперсия поля:   {np.var(f_white):.8f}")
    print(f"Целевая:          {sigma_f ** 2:.8f}")

    # ГАУССОВО ПОЛЕ
    print("\n" + "=" * 60)
    print("ГАУССОВО ПОЛЕ")
    print("=" * 60)

    S_gauss = gaussian_psd(K, sigma_f, xi)
    f_gauss, F_gauss, S_gauss_corr, _ = generate_field_from_psd(
        S_gauss, dx, dy, sigma_target=sigma_f, seed=43
    )

    print(f"Среднее поля:     {np.mean(f_gauss):.8e}")
    print(f"Дисперсия поля:   {np.var(f_gauss):.8f}")
    print(f"Целевая:          {sigma_f ** 2:.8f}")

    # ПРОВЕРКА HERMITIAN SYMMETRY
    print("\n" + "=" * 60)
    print("ПРОВЕРКА HERMITIAN SYMMETRY")
    print("=" * 60)

    max_error, is_hermitian = check_hermitian_symmetry(F_gauss)
    print(f"Максимальная ошибка: {max_error:.3e}")
    print(f"Эрмитова симметрия: {'✅ PASS' if is_hermitian else '❌ FAIL'}")

    # ТЕСТ КОРРЕЛЯЦИОННОЙ ДЛИНЫ
    print("\n" + "=" * 60)
    print("ТЕСТ КОРРЕЛЯЦИОННОЙ ДЛИНЫ")
    print("=" * 60)

    xi_measured, xi_error_rel, passed = test_correlation_length(
        f_gauss, dx, xi, tolerance=0.10
    )

    print(f"Заданная xi:              {xi:.6f}")
    print(f"Измеренная xi:            {xi_measured:.6f}")
    print(f"Относительная ошибка:      {xi_error_rel:.2f} %")
    print(f"Уровень C(xi)/C(0):        {np.exp(-0.5):.6f}")
    print(f"РЕЗУЛЬТАТ: {'✅ PASS' if passed else '❌ FAIL'}")

    # ТЕСТ ФОРМЫ КОРРЕЛЯЦИОННОЙ ФУНКЦИИ
    print("\n" + "=" * 60)
    print("ТЕСТ ФОРМЫ КОРРЕЛЯЦИОННОЙ ФУНКЦИИ")
    print("=" * 60)

    rmse, max_error = test_correlation_shape(f_gauss, dx, xi, r_limit=3.0)
    print(f"RMSE:                     {rmse:.6f}")
    print(f"Максимальная ошибка:      {max_error:.6f}")

    # ВИЗУАЛИЗАЦИЯ
    plot_results(f_white, f_gauss, S_gauss_corr, kx, ky, xi, Lx, Ly, dx, KX, KY)


if __name__ == "__main__":
    main()