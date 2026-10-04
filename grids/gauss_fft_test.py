import numpy as np
import matplotlib.pyplot as plt
from r_grid import build_fft_r_grid
from k_grid import build_fft_k_grid


def test_gaussian_fft(N, Lx, Ly, sigma, center=(0, 0)):
    """
    Полный тест: гаусс -> FFT (числ. vs аналит.) -> обратное FFT

    Параметры:
        N      : число точек по каждому измерению
        Lx, Ly : физические размеры области
        sigma  : ширина гаусса (чем меньше, тем уже пик)
        center : координаты центра гаусса (x0, y0)
    """

    print("=" * 70)
    print("ТЕСТ: ГАУСС -> ПРЯМОЕ FFT -> ОБРАТНОЕ FFT")
    print("=" * 70)

    # ============================================================
    # 1. Построение сеток
    # ============================================================
    print("\n1. Построение сеток...")

    # Реальное пространство (r-сетка): координаты X, Y и шаги dx, dy
    X, Y, dx, dy, x, y = build_fft_r_grid(N, N, Lx, Ly)
    print(f"   dx = {dx:.4f}, dy = {dy:.4f}")
    print(f"   Область: x = [0, {Lx}], y = [0, {Ly}]")
    print(f"   Форма сетки: {X.shape}")

    # Частотное пространство (k-сетка): волновые числа KX, KY
    KX, KY, K, kx, ky = build_fft_k_grid(N, N, dx, dy)
    dkx = kx[1] - kx[0] if N > 1 else 2 * np.pi / Lx
    dky = ky[1] - ky[0] if N > 1 else 2 * np.pi / Ly
    print(f"   dkx = {dkx:.4f}, dky = {dky:.4f}")
    print(f"   Диапазон kx: [{kx}]")

    # ============================================================
    # 2. Задание гаусса в реальном пространстве
    # ============================================================
    print("\n2. Задание гауссовой функции...")
    x0, y0 = center
    print(f"   Центр: ({x0}, {y0})")
    print(f"   Ширина sigma = {sigma}")

    # Формула: f(r) = exp(-|r - r0|² / (2σ²))
    f_gaussian = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))

    print(f"   Максимальное значение: {f_gaussian.max():.6f} (должно быть 1)")
    print(f"   Минимальное значение: {f_gaussian.min():.6f}")

    # ============================================================
    # 3. Аналитическое Фурье-преобразование
    # ============================================================
    print("\n3. Вычисление аналитического Фурье-образа...")
    print("   Формула: F_anal(k) = 2πσ² · exp(-k²σ²/2) · exp(-i·k·r0)")

    # Амплитудная часть (не зависит от сдвига)
    envelope = 2 * np.pi * sigma ** 2 * np.exp(-(KX ** 2 + KY ** 2) * sigma ** 2 / 2)

    # Фазовая часть (зависит от сдвига)
    phase = np.exp(-1j * (KX * x0 + KY * y0))

    F_analytical = envelope * phase

    print(f"   |F_anal| max: {np.abs(F_analytical).max():.6f}")
    print(f"   |F_anal| min: {np.abs(F_analytical).min():.6f}")

    # ============================================================
    # 4. Численное Фурье-преобразование (FFT)
    # ============================================================
    print("\n4. Вычисление численного Фурье-образа (FFT)...")
    print("   F_num = fft2(f) · dx · dy")

    # Прямое FFT (без нормировки) даёт сумму по точкам
    F_fft = np.fft.fft2(f_gaussian)

    # Умножаем на dx·dy, чтобы аппроксимировать непрерывный интеграл
    F_numerical = F_fft * dx * dy

    print(f"   |F_num| max: {np.abs(F_numerical).max():.6f}")
    print(f"   |F_num| min: {np.abs(F_numerical).min():.6f}")

    # ============================================================
    # 5. Сравнение численного и аналитического спектров
    # ============================================================
    print("\n5. Сравнение спектров (модули)...")

    abs_F_num = np.abs(F_numerical)
    abs_F_anal = np.abs(F_analytical)

    # Относительная ошибка (только по точкам, где аналит. значение не слишком мало)
    mask = abs_F_anal > 1e-8 * np.max(abs_F_anal)
    if np.any(mask):
        rel_errors = np.abs(abs_F_num[mask] - abs_F_anal[mask]) / abs_F_anal[mask]
        mean_rel_error = np.mean(rel_errors)
        max_rel_error = np.max(rel_errors)
        print(f"   Средняя относительная ошибка: {mean_rel_error:.4e} ({mean_rel_error * 100:.4f}%)")
        print(f"   Максимальная относительная ошибка: {max_rel_error:.4e}")
    else:
        mean_rel_error = float('inf')
        print("   Ошибка: не найдено точек для сравнения")

    # Максимальная абсолютная разница
    max_abs_diff = np.max(np.abs(abs_F_num - abs_F_anal))
    print(f"   Максимальная абсолютная разница: {max_abs_diff:.6e}")

    # ============================================================
    # 6. Обратное преобразование (восстановление)
    # ============================================================
    print("\n6. Обратное преобразование...")
    print("   f_recovered = ifft2(F_num / (dx·dy))")

    # Из численного спектра
    f_recovered_num = np.fft.ifft2(F_numerical / (dx * dy)).real

    # Из аналитического спектра (дискретизированного)
    f_recovered_anal = np.fft.ifft2(F_analytical / (dx * dy)).real

    # Ошибки восстановления
    max_err_num = np.max(np.abs(f_gaussian - f_recovered_num))
    max_err_anal = np.max(np.abs(f_gaussian - f_recovered_anal))
    mean_err_num = np.mean(np.abs(f_gaussian - f_recovered_num))
    mean_err_anal = np.mean(np.abs(f_gaussian - f_recovered_anal))

    print(f"\n   Восстановление из ЧИСЛЕННОГО спектра:")
    print(f"      Макс. ошибка: {max_err_num:.6e}")
    print(f"      Средняя ошибка: {mean_err_num:.6e}")

    print(f"\n   Восстановление из АНАЛИТИЧЕСКОГО спектра:")
    print(f"      Макс. ошибка: {max_err_anal:.6e}")
    print(f"      Средняя ошибка: {mean_err_anal:.6e}")

    # ============================================================
    # 7. Визуализация
    # ============================================================
    print("\n7. Построение графиков...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    fig.suptitle(f'Тест преобразования Фурье гаусса (N={N}, σ={sigma})', fontsize=14)

    # ---------- Первая строка: исходный и спектры ----------

    # 1. Исходный гаусс
    im1 = axes[0, 0].imshow(f_gaussian, origin='lower',
                            extent=[0, Lx, 0, Ly], cmap='viridis')
    axes[0, 0].set_title(f'1. Исходный гаусс\nЦентр: ({x0},{y0}), σ={sigma}')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. Численный спектр |F_num| (центрированный)
    kx_shifted = np.fft.fftshift(kx)
    ky_shifted = np.fft.fftshift(ky)
    im2 = axes[0, 1].imshow(np.fft.fftshift(abs_F_num), origin='lower',
                            extent=[kx_shifted[0], kx_shifted[-1],
                                    ky_shifted[0], ky_shifted[-1]],
                            cmap='plasma')
    axes[0, 1].set_title('2. Численный спектр |F_num|')
    axes[0, 1].set_xlabel('kx')
    axes[0, 1].set_ylabel('ky')
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. Аналитический спектр |F_anal| (центрированный)
    im3 = axes[0, 2].imshow(np.fft.fftshift(abs_F_anal), origin='lower',
                            extent=[kx_shifted[0], kx_shifted[-1],
                                    ky_shifted[0], ky_shifted[-1]],
                            cmap='plasma')
    axes[0, 2].set_title('3. Аналитический спектр |F_anal|')
    axes[0, 2].set_xlabel('kx')
    axes[0, 2].set_ylabel('ky')
    plt.colorbar(im3, ax=axes[0, 2])

    # ---------- Вторая строка: восстановление и ошибки ----------

    # 4. Восстановленный из численного спектра
    im4 = axes[1, 0].imshow(f_recovered_num, origin='lower',
                            extent=[0, Lx, 0, Ly], cmap='viridis')
    axes[1, 0].set_title(f'4. Восстановлен из числ. спектра\nОшибка: {max_err_num:.2e}')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 0])

    # 5. Восстановленный из аналитического спектра
    im5 = axes[1, 1].imshow(f_recovered_anal, origin='lower',
                            extent=[0, Lx, 0, Ly], cmap='viridis')
    axes[1, 1].set_title(f'5. Восстановлен из анал. спектра\nОшибка: {max_err_anal:.2e}')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 1])

    # 6. Разница между исходным и восстановленным (из численного)
    diff = np.abs(f_gaussian - f_recovered_num)
    im6 = axes[1, 2].imshow(diff, origin='lower',
                            extent=[0, Lx, 0, Ly], cmap='Reds')
    axes[1, 2].set_title(f'6. Абсолютная ошибка\nМакс: {max_err_num:.2e}')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im6, ax=axes[1, 2])

    plt.tight_layout()
    plt.show()

    # ============================================================
    # 8. Итоговый вердикт
    # ============================================================
    print("\n" + "=" * 70)
    print("ИТОГИ ТЕСТА")
    print("=" * 70)

    max_abs_diff = np.max(np.abs(abs_F_num - abs_F_anal))
    print(f"Макс. абсолютная разница: {max_abs_diff:.3e}")
    max_idx = np.unravel_index(np.argmax(np.abs(abs_F_num - abs_F_anal)), abs_F_num.shape)
    print(f"Максимальная разница в точке: kx={kx[max_idx[0]]:.2f}, ky={ky[max_idx[1]]:.2f}")
    total_sum = np.sum(f_gaussian)
    print(f"Сумма значений гаусса: {total_sum:.6f}")
    print(f"Интеграл по области (сумма * dx*dy): {total_sum * dx * dy:.6f}")
    print(f"Аналитический интеграл (2πσ²): {2 * np.pi * sigma ** 2:.6f}")

    # Критерии успеха (подобраны эмпирически)
    recovery_num_ok = max_err_num < 1e-10
    recovery_anal_ok = max_err_anal < 1e-2  # Аналитический восстановит хуже из-за дискретизации
    # Вместо сравнения mean_rel_error < 0.05:
    abs_error_threshold = 1e-4  # Абсолютная ошибка меньше 0.0001
    max_abs_diff = np.max(np.abs(abs_F_num - abs_F_anal))

    spectrum_ok = max_abs_diff < abs_error_threshold

    print(
        f"✓ Сравнение спектров (макс. абс. ошибка < {abs_error_threshold}): {'ДА' if spectrum_ok else 'НЕТ'} ({max_abs_diff:.2e})")

    print(
        f"✓ Восстановление из численного спектра (ошибка < 1e-10): {'ДА' if recovery_num_ok else 'НЕТ'} ({max_err_num:.2e})")
    print(
        f"✓ Восстановление из аналитического спектра (ошибка < 1e-2): {'ДА' if recovery_anal_ok else 'НЕТ'} ({max_err_anal:.2e})")

    if spectrum_ok and recovery_num_ok:
        print("\n✅ ТЕСТ ПРОЙДЕН! Численное и аналитическое преобразования согласованы.")
    else:
        print("\n⚠️ ТЕСТ НЕ ПРОЙДЕН. Проверьте параметры (особенно sigma относительно Lx/Ly).")

    return {
        'f_gaussian': f_gaussian,
        'F_numerical': F_numerical,
        'F_analytical': F_analytical,
        'f_recovered_num': f_recovered_num,
        'f_recovered_anal': f_recovered_anal,
        'mean_rel_error': mean_rel_error,
        'max_err_num': max_err_num,
        'max_err_anal': max_err_anal
    }


# ============================================================
# ЗАПУСК ТЕСТА
# ============================================================
if __name__ == "__main__":
    # Параметры (можно менять)
    N = 128  # чем больше, тем точнее
    Lx, Ly = 20.0, 20.0  # размер области
    sigma = 1 # ширина гаусса  (должна быть < L/2, чтобы не обрезалась)
    center = (Lx / 2, Ly / 2)  # центр в середине области

    result = test_gaussian_fft(N, Lx, Ly, sigma, center)