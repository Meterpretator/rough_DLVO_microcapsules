import numpy as np
import matplotlib.pyplot as plt

from generate_surface.grids.k_grid import build_fft_k_grid
from generate_surface.grids.r_grid import build_fft_r_grid


# ============================================================
# ПАРАМЕТРЫ
# ============================================================

N = 128

Lx = 20.0
Ly = 20.0

sigma_f = 1.0
xi = 1.0

# np.random.seed(42)


# ============================================================
# REAL-SPACE GRID
# ============================================================

X, Y, dx, dy, x, y = build_fft_r_grid(
    N, N, Lx, Ly
)


# ============================================================
# K-SPACE GRID
# ============================================================

KX, KY, K, kx, ky = build_fft_k_grid(
    N, N, dx, dy
)


# ============================================================
# ОСНОВНЫЕ FFT ПАРАМЕТРЫ
# ============================================================

nx = N
ny = N

area = Lx * Ly

dkx = 2.0 * np.pi / Lx
dky = 2.0 * np.pi / Ly

# Для numpy FFT:
#
# ifft2(F) = 1/(Nx Ny) * sum(F exp(ikr))
#
# Поэтому спектральный коэффициент имеет масштаб
#
# Nx Ny / sqrt(Lx Ly)
#
# при использовании PSD в непрерывной нормировке.
#
fft_scale = nx * ny / np.sqrt(area)


# ============================================================
# PSD
# ============================================================

def gaussian_psd(K, sigma, xi):
    """
    PSD для корреляционной функции

        C(r) = sigma^2 exp[-r^2/(2 xi^2)]

    при соглашении

        C(r) =
        integral S(k) exp(i k r) dk / (2*pi)^2
    """

    return (
        2.0
        * np.pi
        * sigma**2
        * xi**2
        * np.exp(
            -0.5 * (K * xi)**2
        )
    )


# ============================================================
# GENERATE REAL GAUSSIAN FIELD
# ============================================================

def generate_field_from_psd(S, sigma_target=None):
    """
    Генерация вещественного Gaussian random field
    из PSD S(k).

    K-grid находится в стандартном порядке numpy FFT:

        [0, +k, ..., -k]

    поэтому fftshift НЕ используется.
    """

    # --------------------------------------------------------
    # Проверяем дискретную PSD
    # --------------------------------------------------------

    variance_psd = (
        np.sum(S)
        * dkx
        * dky
        / (2.0 * np.pi)**2
    )

    print(
        f"Дискретная дисперсия из PSD: "
        f"{variance_psd:.8f}"
    )

    # --------------------------------------------------------
    # При конечной сетке интеграл PSD немного отличается
    # от бесконечного аналитического интеграла.
    #
    # Если sigma_target задан, корректируем PSD так,
    # чтобы именно дискретная сумма давала sigma_target^2.
    # --------------------------------------------------------

    if sigma_target is not None:

        S = S * (
            sigma_target**2
            / variance_psd
        )

        variance_psd = (
            np.sum(S)
            * dkx
            * dky
            / (2.0 * np.pi)**2
        )

        print(
            f"Дискретная дисперсия после "
            f"нормировки: {variance_psd:.8f}"
        )

    # --------------------------------------------------------
    # Complex Gaussian noise
    #
    # E[|Z|^2] = 1
    # --------------------------------------------------------

    Z = (
        np.random.normal(
            size=(nx, ny)
        )
        +
        1j * np.random.normal(
            size=(nx, ny)
        )
    ) / np.sqrt(2.0)

    # --------------------------------------------------------
    # Hermitian symmetry
    #
    # F[-k] = conj(F[k])
    #
    # Индекс -i в FFT соответствует:
    #
    # (-i) % N
    # --------------------------------------------------------

    Z_sym = np.empty_like(Z)

    for i in range(nx):
        for j in range(ny):

            ii = (-i) % nx
            jj = (-j) % ny

            # Берём один коэффициент из пары
            if (
                i < ii
                or
                (i == ii and j <= jj)
            ):

                z = Z[i, j]

                # Если точка сама себе сопряжённая
                # (DC или Nyquist/Nyquist),
                # коэффициент должен быть вещественным.

                if i == ii and j == jj:
                    z = np.random.normal()

                Z_sym[i, j] = z
                Z_sym[ii, jj] = np.conj(z)

    # --------------------------------------------------------
    # FFT coefficients
    # --------------------------------------------------------

    F = (
        fft_scale
        * np.sqrt(S)
        * Z_sym
    )

    # --------------------------------------------------------
    # Нулевая частота.
    #
    # Убираем среднее значение поля.
    # --------------------------------------------------------

    F[0, 0] = 0.0

    # --------------------------------------------------------
    # Inverse FFT
    # --------------------------------------------------------

    f = np.fft.ifft2(F).real

    return f, F, S


# ============================================================
# 1. БЕЛЫЙ ШУМ
# ============================================================

print()
print("=" * 60)
print("БЕЛЫЙ ШУМ")
print("=" * 60)

S_white = np.full_like(
    K,
    sigma_f**2
)

f_white, F_white, S_white = \
    generate_field_from_psd(
        S_white,
        sigma_target=sigma_f
    )

print(
    f"Среднее поля:     "
    f"{np.mean(f_white):.8e}"
)

print(
    f"Дисперсия поля:   "
    f"{np.var(f_white):.8f}"
)

print(
    f"Целевая:          "
    f"{sigma_f**2:.8f}"
)


# ============================================================
# 2. GAUSSIAN RANDOM FIELD
# ============================================================

print()
print("=" * 60)
print("ГАУССОВО ПОЛЕ")
print("=" * 60)

S_gauss = gaussian_psd(
    K,
    sigma_f,
    xi
)

f_gauss, F_gauss, S_gauss = \
    generate_field_from_psd(
        S_gauss,
        sigma_target=sigma_f
    )

print(
    f"Среднее поля:     "
    f"{np.mean(f_gauss):.8e}"
)

print(
    f"Дисперсия поля:   "
    f"{np.var(f_gauss):.8f}"
)

print(
    f"Целевая:          "
    f"{sigma_f**2:.8f}"
)


# ============================================================
# 3. ПРОВЕРКА HERMITIAN SYMMETRY
# ============================================================

F_check = np.empty_like(F_gauss)

for i in range(nx):
    for j in range(ny):

        ii = (-i) % nx
        jj = (-j) % ny

        F_check[i, j] = np.conj(
            F_gauss[ii, jj]
        )

hermitian_error = np.max(
    np.abs(
        F_gauss - F_check
    )
)

print()
print(
    f"Максимальная ошибка "
    f"Hermitian symmetry: "
    f"{hermitian_error:.3e}"
)


# ============================================================
# 4. КОРРЕЛЯЦИОННАЯ ФУНКЦИЯ
# ============================================================

# Автокорреляция через Wiener-Khinchin

F = np.fft.fft2(
    f_gauss
)

C = np.fft.ifft2(
    np.abs(F)**2
).real

C /= N * N

# Перемещаем r=0 в центр
C = np.fft.fftshift(C)


# Расстояния относительно центра
rx = (
    np.arange(N) - N // 2
) * dx

ry = (
    np.arange(N) - N // 2
) * dy


# Сечение вдоль x
C_x = C[N // 2, :]

# Нормируем
C_x /= C_x[N // 2]


# Теоретическая корреляция
C_theory = np.exp(
    -rx**2
    /
    (2.0 * xi**2)
)

# ============================================================
# 5. ТЕСТ КОРРЕЛЯЦИОННОЙ ДЛИНЫ
# ============================================================

# Теоретическая корреляционная функция:
#
# C(r) / C(0) = exp(-r^2 / (2*xi^2))
#
# При r = xi:
#
# C(xi) / C(0) = exp(-1/2)
#

correlation_level = np.exp(-0.5)

# ------------------------------------------------------------
# Берём только положительные расстояния
# ------------------------------------------------------------

center = N // 2

r_positive = rx[center:]
C_positive = C_x[center:]

# ------------------------------------------------------------
# Находим первое пересечение уровня exp(-1/2)
# ------------------------------------------------------------

idx = np.where(
    C_positive <= correlation_level
)[0]

if len(idx) == 0:

    xi_measured = np.nan

    print()
    print("ОШИБКА: корреляционная длина не найдена.")

else:

    i = idx[0]

    if i == 0:

        xi_measured = r_positive[0]

    else:

        # Линейная интерполяция между двумя точками

        r1 = r_positive[i - 1]
        r2 = r_positive[i]

        c1 = C_positive[i - 1]
        c2 = C_positive[i]

        xi_measured = (
            r1
            +
            (correlation_level - c1)
            * (r2 - r1)
            / (c2 - c1)
        )


# ------------------------------------------------------------
# Ошибка корреляционной длины
# ------------------------------------------------------------

if np.isfinite(xi_measured):

    xi_error_abs = (
        xi_measured - xi
    )

    xi_error_rel = (
        100.0
        * xi_error_abs
        / xi
    )

else:

    xi_error_abs = np.nan
    xi_error_rel = np.nan


# ------------------------------------------------------------
# Вывод
# ------------------------------------------------------------

print()
print("=" * 60)
print("ТЕСТ КОРРЕЛЯЦИОННОЙ ДЛИНЫ")
print("=" * 60)

print(
    f"Заданная xi:              {xi:.6f}"
)

print(
    f"Измеренная xi:            {xi_measured:.6f}"
)

print(
    f"Абсолютная ошибка:         "
    f"{xi_error_abs:.6f}"
)

print(
    f"Относительная ошибка:      "
    f"{xi_error_rel:.2f} %"
)

print(
    f"Уровень C(xi)/C(0):        "
    f"{correlation_level:.6f}"
)

# ------------------------------------------------------------
# Допуск
# ------------------------------------------------------------

xi_tolerance = 0.10   # 10 %

if (
    np.isfinite(xi_measured)
    and abs(xi_error_rel) <= 100.0 * xi_tolerance
):

    print(
        "РЕЗУЛЬТАТ: PASS — "
        "корреляционная длина совпадает "
        "с заданной в пределах допуска."
    )

else:

    print(
        "РЕЗУЛЬТАТ: FAIL — "
        "корреляционная длина отличается "
        "от заданной."
    )
# ============================================================
# ТЕСТ ФОРМЫ КОРРЕЛЯЦИОННОЙ ФУНКЦИИ
# ============================================================

# Используем только область, где корреляция ещё заметна.
# Это уменьшает влияние статистического шума на больших r.

mask = (
    (np.abs(rx) <= 3.0 * xi)
)

correlation_rmse = np.sqrt(
    np.mean(
        (
            C_x[mask]
            -
            C_theory[mask]
        )**2
    )
)

correlation_max_error = np.max(
    np.abs(
        C_x[mask]
        -
        C_theory[mask]
    )
)

print()
print("=" * 60)
print("ТЕСТ ФОРМЫ КОРРЕЛЯЦИОННОЙ ФУНКЦИИ")
print("=" * 60)

print(
    f"RMSE:                     "
    f"{correlation_rmse:.6f}"
)

print(
    f"Максимальная ошибка:      "
    f"{correlation_max_error:.6f}"
)
# ============================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================

fig = plt.figure(
    figsize=(14, 10)
)


# ------------------------------------------------------------
# White noise
# ------------------------------------------------------------

ax1 = fig.add_subplot(221)

im1 = ax1.imshow(
    f_white,
    extent=[
        0,
        Lx,
        0,
        Ly
    ],
    origin="lower",
    aspect="equal"
)

ax1.set_title(
    f"Белый шум\n"
    f"Var = {np.var(f_white):.4f}"
)

ax1.set_xlabel("x")
ax1.set_ylabel("y")

plt.colorbar(
    im1,
    ax=ax1
)


# ------------------------------------------------------------
# Gaussian field
# ------------------------------------------------------------

ax2 = fig.add_subplot(222)

im2 = ax2.imshow(
    f_gauss,
    extent=[
        0,
        Lx,
        0,
        Ly
    ],
    origin="lower",
    aspect="equal"
)

ax2.set_title(
    f"Гауссово поле\n"
    f"Var = {np.var(f_gauss):.4f}, "
    f"xi = {xi}"
)

ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.colorbar(
    im2,
    ax=ax2
)


# ------------------------------------------------------------
# PSD
# ------------------------------------------------------------

ax3 = fig.add_subplot(223)

# Для отображения PSD делаем shift,
# но только для visualization.
#
# Это НЕ используется при генерации FFT.

S_plot = np.fft.fftshift(
    S_gauss
)

K_plot = np.fft.fftshift(
    K
)

im3 = ax3.imshow(
    np.log10(
        S_plot + 1e-30
    ),
    extent=[
        -np.max(np.abs(kx)),
        np.max(np.abs(kx)),
        -np.max(np.abs(ky)),
        np.max(np.abs(ky))
    ],
    origin="lower",
    aspect="equal"
)

ax3.set_title(
    "log10 PSD"
)

ax3.set_xlabel("kx")
ax3.set_ylabel("ky")

plt.colorbar(
    im3,
    ax=ax3
)


# ------------------------------------------------------------
# Correlation
# ------------------------------------------------------------

ax4 = fig.add_subplot(224)

ax4.plot(
    rx,
    C_x,
    label="численная"
)

ax4.plot(
    rx,
    C_theory,
    "--",
    label="теоретическая"
)

ax4.set_xlim(
    -5.0 * xi,
    5.0 * xi
)

ax4.set_ylim(
    -0.1,
    1.1
)

ax4.set_xlabel("r")
ax4.set_ylabel("C(r) / C(0)")

ax4.set_title(
    "Корреляционная функция"
)

ax4.grid(True)

ax4.legend()


plt.tight_layout()

plt.show()