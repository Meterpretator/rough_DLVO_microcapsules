import numpy as np
import matplotlib.pyplot as plt

from generate_surface.grids.k_grid import build_fft_k_grid
from generate_surface.grids.r_grid import build_fft_r_grid

# ============================================================
# 1. ПАРАМЕТРЫ
# ============================================================

N = 1024
Lx = 20.0
Ly = 20.0
sigma_f = 1.0
xi = 1.0
# np.random.seed(42)  # не рандом

# ============================================================
# 2. ПОСТРОЕНИЕ СЕТОК
# ============================================================

def build_grids(N, Lx, Ly):
    X, Y, dx, dy, x, y = build_fft_r_grid(N, N, Lx, Ly)
    KX, KY, K, kx, ky = build_fft_k_grid(N, N, dx, dy)
    return X, Y, dx, dy, x, y, KX, KY, K, kx, ky

# ============================================================
# 3. PSD ФУНКЦИИ
# ============================================================

def gaussian_psd(K, sigma, xi):
    return (
            2.0 * np.pi * sigma ** 2 * xi ** 2
            * np.exp(-0.5 * (K * xi) ** 2)
    )


def white_noise_psd(K, sigma):
    return np.full_like(K, sigma ** 2)


# ============================================================
# 4. ГЕНЕРАЦИЯ ЭРМИТОВА СПЕКТРА
# ============================================================
def generate_hermitian_spectrum(nx, ny, seed=None):
    if seed is not None:
        np.random.seed(seed)

    Z = (
        np.random.normal(size=(nx, ny))
        + 1j * np.random.normal(size=(nx, ny))
    ) / np.sqrt(2.0)

    # Тестовый режим:
    #Z = np.ones((nx, ny), dtype=complex)  # детерминир поле
    Z_sym = np.empty_like(Z)

    for i in range(nx):
        for j in range(ny):
            ii = (-i) % nx
            jj = (-j) % ny

            if i < ii or (i == ii and j <= jj):
                if i == 0 and j == 0:
                    z = 0.0       # DC = 0
                else:
                    z = Z[i, j]

                Z_sym[i, j] = z
                Z_sym[ii, jj] = np.conj(z)

    return Z_sym


# ============================================================
# 5. ГЕНЕРАЦИЯ ПОЛЯ ИЗ PSD
# ============================================================
def generate_field_from_psd(S, dx, dy, sigma_target=None, seed=None, verbose=True):
    """
    Генерирует вещественное случайное поле с заданной PSD S(k).

    Основные условия:
    1. DC-компонента исключается из PSD до нормировки.
    2. PSD нормируется так, чтобы интегральная дисперсия
       по ненулевым модам была равна sigma_target^2.
    3. generate_hermitian_spectrum() задаёт Z[0, 0] = 0,
       поэтому среднее генерируемого поля строго равно нулю.
    4. Эрмитова симметрия спектра обеспечивает вещественное поле.

    Параметры
    ---------
    S : 2D ndarray
        Целевая спектральная плотность мощности S(kx, ky).
    dx, dy : float
        Шаги пространственной сетки.
    sigma_target : float or None
        Требуемое стандартное отклонение поля.
        Если None, PSD не масштабируется.
    seed : int or None
        Seed генератора случайных чисел.
    verbose : bool
        Выводить диагностическую информацию.

    Возвращает
    ----------
    f : 2D ndarray
        Сгенерированное вещественное поле.
    F : 2D ndarray
        Его комплексный Fourier-спектр.
    S_corrected : 2D ndarray
        PSD после удаления DC и нормировки.
    variance_psd_without_dc : float
        Дисперсия, соответствующая исходной PSD без DC.
    """

    nx, ny = S.shape

    # ------------------------------------------------------------
    # 1. Геометрия области
    # ------------------------------------------------------------
    area = (nx * dx) * (ny * dy)

    dkx = 2.0 * np.pi / (nx * dx)
    dky = 2.0 * np.pi / (ny * dy)

    # Масштаб для перехода от спектральных коэффициентов
    # к нормировке np.fft.ifft2().
    fft_scale = nx * ny / np.sqrt(area)

    # ------------------------------------------------------------
    # 2. Проверка PSD
    # ------------------------------------------------------------
    if np.any(S < 0):
        raise ValueError("PSD S содержит отрицательные значения.")

    # ------------------------------------------------------------
    # 3. Удаляем DC непосредственно из PSD
    # ------------------------------------------------------------
    S_work = np.array(S, dtype=float, copy=True)

    dc_original = S_work[0, 0]

    S_work[0, 0] = 0.0

    # ------------------------------------------------------------
    # 4. Дисперсия исходной PSD БЕЗ DC
    #
    # sigma^2 =
    # ∫ S(k) d^2k / (2π)^2
    #
    # В дискретной форме:
    # sum(S) * dkx * dky / (2π)^2
    # ------------------------------------------------------------
    variance_psd_without_dc = (
        np.sum(S_work)
        * dkx
        * dky
        / (2.0 * np.pi) ** 2
    )

    if verbose:
        variance_psd_full = (
            np.sum(S)
            * dkx
            * dky
            / (2.0 * np.pi) ** 2
        )

        print(
            f"Полная дисперсия из PSD: "
            f"{variance_psd_full:.8e}"
        )

        print(
            f"Дисперсия без DC: "
            f"{variance_psd_without_dc:.8e}"
        )

        print(
            f"DC-вклад: "
            f"{dc_original * dkx * dky / (2.0 * np.pi) ** 2:.8e}"
        )

    if variance_psd_without_dc <= 0.0:
        raise ValueError(
            "Дисперсия PSD без DC должна быть положительной."
        )

    # ------------------------------------------------------------
    # 5. Нормировка PSD
    # ------------------------------------------------------------
    if sigma_target is not None:

        if sigma_target < 0:
            raise ValueError(
                "sigma_target должен быть >= 0."
            )

        scale = (
            sigma_target ** 2
            / variance_psd_without_dc
        )

        S_corrected = S_work * scale

        if verbose:
            print(
                f"Масштабный коэффициент: "
                f"{scale:.8f}"
            )

            new_variance = (
                np.sum(S_corrected)
                * dkx
                * dky
                / (2.0 * np.pi) ** 2
            )

            print(
                f"Новая дисперсия без DC: "
                f"{new_variance:.8f} "
                f"(цель {sigma_target ** 2:.8f})"
            )

    else:
        S_corrected = S_work

    # ------------------------------------------------------------
    # 6. Генерация случайного эрмитова спектра
    #
    # Внутри generate_hermitian_spectrum():
    #
    # Z[0,0] = 0
    #
    # поэтому DC Fourier-компонента будет равна нулю.
    # ------------------------------------------------------------
    Z_sym = generate_hermitian_spectrum(
        nx,
        ny,
        seed=seed
    )

    # ------------------------------------------------------------
    # 7. Формирование Fourier-спектра
    # ------------------------------------------------------------
    F = (
        fft_scale
        * np.sqrt(S_corrected)
        * Z_sym
    )

    # ------------------------------------------------------------
    # 8. Обратное FFT
    #
    # Благодаря:
    #
    #   Z_sym[0,0] = 0
    #
    # имеем:
    #
    #   F[0,0] = 0
    #
    # и, следовательно:
    #
    #   mean(f) = 0
    #
    # с точностью до машинного округления.
    # ------------------------------------------------------------
    f = np.fft.ifft2(F).real

    return (
        f,
        F,
        S_corrected,
        variance_psd_without_dc
    )


# ============================================================
# 6. ПРОВЕРКА ЭРМИТОВОЙ СИММЕТРИИ
# ============================================================

def check_hermitian_symmetry(F, tol=1e-12):
    nx, ny = F.shape
    F_check = np.empty_like(F)

    for i in range(nx):
        for j in range(ny):
            ii = (-i) % nx
            jj = (-j) % ny
            F_check[i, j] = np.conj(F[ii, jj])

    max_error = np.max(np.abs(F - F_check))
    is_hermitian = max_error < tol

    return max_error, is_hermitian


# ============================================================
# 7. КОРРЕЛЯЦИОННАЯ ФУНКЦИЯ
# ============================================================
def compute_correlation_function(f, dx, center=True):
    """
    Вычисляет периодическую автокорреляционную функцию:

        C(dx, dy) = IFFT2(F * conj(F)) / (Nx * Ny)

    где:
        F = FFT2(f)

    При center=True нулевая задержка переносится в центр массива.
    """

    nx, ny = f.shape

    # Fourier transform поля
    F = np.fft.fft2(f)

    # Автокорреляция:
    # F * F*
    C = np.fft.ifft2(F * np.conj(F)).real

    # Нормировка
    C /= nx * ny

    if center:
        C = np.fft.fftshift(C)

        rx = (np.arange(nx) - nx // 2) * dx
        ry = (np.arange(ny) - ny // 2) * dx
    else:
        rx = np.arange(nx) * dx
        ry = np.arange(ny) * dx

    return C, rx, ry


# ============================================================
# 8. ТЕСТ КОРРЕЛЯЦИОННОЙ ДЛИНЫ
# ============================================================

def test_correlation_length(f, dx, xi, tolerance=0.10):
    C, rx, _ = compute_correlation_function(f, dx, center=True)

    center_idx = len(rx) // 2
    C_x = C[center_idx, :]
    C_x /= C_x[center_idx]

    correlation_level = np.exp(-0.5)
    r_positive = rx[center_idx:]
    C_positive = C_x[center_idx:]

    idx = np.where(C_positive <= correlation_level)[0]

    if len(idx) == 0:
        return np.nan, np.nan, False

    i = idx[0]
    if i == 0:
        xi_measured = r_positive[0]
    else:
        r1 = r_positive[i - 1]
        r2 = r_positive[i]
        c1 = C_positive[i - 1]
        c2 = C_positive[i]
        xi_measured = r1 + (correlation_level - c1) * (r2 - r1) / (c2 - c1)

    xi_error_rel = 100.0 * (xi_measured - xi) / xi
    passed = abs(xi_error_rel) <= 100.0 * tolerance

    return xi_measured, xi_error_rel, passed


# ============================================================
# 9. ТЕСТ ФОРМЫ КОРРЕЛЯЦИОННОЙ ФУНКЦИИ
# ============================================================

def test_correlation_shape(f, dx, xi, r_limit=3.0):
    C, rx, _ = compute_correlation_function(f, dx, center=True)

    center_idx = len(rx) // 2
    C_x = C[center_idx, :]
    C_x /= C_x[center_idx]

    C_theory = np.exp(-rx ** 2 / (2.0 * xi ** 2))
    mask = (np.abs(rx) <= r_limit * xi)

    rmse = np.sqrt(np.mean((C_x[mask] - C_theory[mask]) ** 2))
    max_error = np.max(np.abs(C_x[mask] - C_theory[mask]))

    return rmse, max_error


# ============================================================
# 10. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_results(f_white, f_gauss, S_gauss, kx, ky, xi, Lx, Ly, dx, KX, KY):
    fig = plt.figure(figsize=(14, 10))

    # Белый шум
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(f_white, extent=[0, Lx, 0, Ly], origin="lower", aspect="equal")
    ax1.set_title(f"Белый шум\nVar = {np.var(f_white):.4f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1)

    # Гауссово поле
    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(f_gauss, extent=[0, Lx, 0, Ly], origin="lower", aspect="equal")
    ax2.set_title(f"Гауссово поле\nVar = {np.var(f_gauss):.4f}, xi = {xi}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.colorbar(im2, ax=ax2)

    # PSD
    ax3 = fig.add_subplot(223)
    S_plot = np.fft.fftshift(S_gauss)
    im3 = ax3.imshow(
        np.log10(S_plot + 1e-30),
        extent=[-np.max(np.abs(kx)), np.max(np.abs(kx)),
                -np.max(np.abs(ky)), np.max(np.abs(ky))],
        origin="lower",
        aspect="equal"
    )
    ax3.set_title("log10 PSD")
    ax3.set_xlabel("kx")
    ax3.set_ylabel("ky")
    plt.colorbar(im3, ax=ax3)

    # АКФ
    ax4 = fig.add_subplot(224)
    C_gauss, rx, _ = compute_correlation_function(f_gauss, dx, center=True)
    center_idx = len(rx) // 2
    C_x = C_gauss[center_idx, :]
    C_x /= C_x[center_idx]
    C_theory = np.exp(-rx ** 2 / (2.0 * xi ** 2))

    ax4.plot(rx, C_x, label="численная")
    ax4.plot(rx, C_theory, "--", label="теоретическая")
    ax4.set_xlim(-5.0 * xi, 5.0 * xi)
    ax4.set_ylim(-0.1, 1.1)
    ax4.set_xlabel("r")
    ax4.set_ylabel("C(r) / C(0)")
    ax4.set_title("Корреляционная функция")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()