import numpy as np
import matplotlib.pyplot as plt

from r_grid import build_fft_r_grid
from k_grid import build_fft_k_grid
from k_grid import test_k_meshgrids
from r_grid import test_r_meshgrids

N = 6        # число точек (чётное для наглядности, можно 5 для нечётного)
Lx, Ly = 6, 6   # размер области
L=Lx

X, Y, dx, dy, x, y = build_fft_r_grid(N,N, Lx, Ly)   # Построение r-сетки (передаём N, N, L, L)
KX, KY, K, kx, ky= build_fft_k_grid(N,N, dx, dy)   # Построение k-сетки

# функции тестов
test_k_meshgrids(N, Lx,Ly)
test_r_meshgrids(N, Lx, Ly)
print(type(kx), len(kx))
print(type(ky), len(ky))


# ============================================================
# ПРОВЕРКА СОГЛАСОВАННОСТИ
# ============================================================

# 1. Шаг по пространству
dx_expected = Lx / N
dy_expected = Ly / N

print("=" * 50)
print("1. Шаги r-сетки:")
print(f"   dx (из функции) = {dx:.6f}, ожидается {dx_expected:.6f} (Lx / N)")
print(f"   dy (из функции) = {dy:.6f}, ожидается {dy_expected:.6f} (Ly / N)")
assert abs(dx - dx_expected) < 1e-12, "Ошибка: dx не совпадает с Lx/N"
assert abs(dy - dy_expected) < 1e-12, "Ошибка: dy не совпадает с Ly/N"

# 2. Шаг по k-сетке
dkx_expected = 2 * np.pi / Lx
dky_expected = 2 * np.pi / Ly
# Убеждаемся, что kx и ky — одномерные массивы
kx = np.asarray(kx).flatten()  # преобразуем в плоский одномерный массив
ky = np.asarray(ky).flatten()

# Теперь dkx — скаляр
dkx_actual = kx[1] - kx[0]
dky_actual = ky[1] - ky[0]

print("kx[:5]:", kx[:5])
print("kx[-5:]:", kx[-5:])
print("dkx_actual =", dkx_actual)
print("dkx_expected =", dkx_expected)

print("\n2. Шаги k-сетки:")
print(f"   dkx (факт) = {dkx_actual:.6f}, ожидается {dkx_expected:.6f} (2 * np.pi / Lx)")
print(f"   dky (факт) = {dky_actual:.6f}, ожидается {dky_expected:.6f} (2 * np.pi / Ly)")
assert abs(dkx_actual - dkx_expected) < 1e-12, "Ошибка: dkx не совпадает с 2π/Lx"
assert abs(dky_actual - dky_expected) < 1e-12, "Ошибка: dky не совпадает с 2π/Ly"

# 3. Основное соотношение dx * dk = 2π/N
product_x = dx * dkx_actual
product_y = dy * dky_actual
expected = 2 * np.pi / N

print("\n3. Проверка основного соотношения dx * dk = 2π/N:")
print(f"   dx * dkx = {product_x:.8f}, 2π/N = {expected:.8f}")
print(f"   dy * dky = {product_y:.8f}, 2π/N = {expected:.8f}")
assert abs(product_x - expected) < 1e-12, "Ошибка: dx * dkx != 2π/N"
assert abs(product_y - expected) < 1e-12, "Ошибка: dy * dky != 2π/N"

# 4. Размерности и форма
print("\n4. Проверка размерностей:")
print(f"   len(kx) = {len(kx)}, N = {N}")
print(f"   len(ky) = {len(ky)}, N = {N}")
print(f"   X.shape = {X.shape}, ожидается ({N}, {N})")
assert len(kx) == N, "Ошибка: длина kx не равна N"
assert len(ky) == N, "Ошибка: длина ky не равна N"
assert X.shape == (N, N), "Ошибка: форма X не N×N"

# 5. Нулевая частота
print("\n5. Нулевая частота:")
print(f"   kx[0] = {kx[0]:.6f} (должно быть 0)")
print(f"   ky[0] = {ky[0]:.6f} (должно быть 0)")
assert abs(kx[0]) < 1e-12, "Ошибка: kx[0] != 0"
assert abs(ky[0]) < 1e-12, "Ошибка: ky[0] != 0"

print("\n" + "=" * 50)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ! Сетки согласованы.")

def plot_1d_grids(x, kx):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(x, 'o-')
    ax1.set_title('Пространственная сетка x (r-пространство)')
    ax1.set_xlabel('Индекс')
    ax1.set_ylabel('x')
    ax1.grid(True)
    # Добавляем подписи значений для x
    for i, val in enumerate(x):
        ax1.annotate(f'{val:.3f}', (i, val),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8)

    ax2.plot(kx, 'o-')
    ax2.set_title('Частотная сетка kx (k-пространство, стандартный порядок)')
    ax2.set_xlabel('Индекс')
    ax2.set_ylabel('kx')
    ax2.grid(True)
    # Добавляем подписи значений для kx
    for i, val in enumerate(kx):
        ax2.annotate(f'{val:.3f}', (i, val),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8)
    plt.tight_layout()

def plot_2d_grids(X, Y, KX, KY, K):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(f"FFTSHIFTED GRIDS N={N}x{N} L={Lx}x{Ly} ")
    # X сетка (центрированная)
    im1 = axes[0, 0].imshow(np.fft.fftshift(X), cmap='RdBu')
    axes[0, 0].set_title('X (r-пространство, центрированная)')
    plt.colorbar(im1, ax=axes[0, 0])

    # Y сетка (центрированная)
    im2 = axes[0, 1].imshow(np.fft.fftshift(Y), cmap='RdBu')
    axes[0, 1].set_title('Y (r-пространство, центрированная)')
    plt.colorbar(im2, ax=axes[0, 1])

    # KX сетка (центрированная)
    im3 = axes[1, 0].imshow(np.fft.fftshift(KX), cmap='RdBu')
    axes[1, 0].set_title('KX (k-пространство, центрированная)')
    plt.colorbar(im3, ax=axes[1, 0])

    # KY сетка (центрированная)
    im4 = axes[1, 1].imshow(np.fft.fftshift(KY), cmap='RdBu')
    axes[1, 1].set_title('KY (k-пространство, центрированная)')
    plt.colorbar(im4, ax=axes[1, 1])

    # K (радиальная) сетка (центрированная)
    im5 = axes[1, 2].imshow(np.fft.fftshift(K), cmap='viridis')
    axes[1, 2].set_title('K = sqrt(KX²+KY²) (центрированная)')
    plt.colorbar(im5, ax=axes[1, 2])
    axes[0, 2].axis('off')
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    plt.tight_layout()
    plt.show()


def test_grids_with_linear_functions(N, Lx, Ly):
    """
    Тестирование согласованности сеток на линейных функциях.

    Проверяет, что:
    - f(x) = x правильно отображается в r-пространстве
    - f(y) = y правильно отображается в r-пространстве
    - f(kx) = kx правильно отображается в k-пространстве
    - f(ky) = ky правильно отображается в k-пространстве
    """

    # 1. Строим согласованные сетки
    dx = Lx / N
    dy = Ly / N

    X, Y, _, _, x, y = build_fft_r_grid(N, N, Lx, Ly)
    KX, KY, K, kx, ky = build_fft_k_grid(N, N, dx, dy)

    # 2. Создаём фигуру с 4 подграфиками
    fig = plt.figure(figsize=(16, 12))

    # График 1: f(x) = x в r-пространстве
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, X, cmap='viridis', edgecolor='none')
    ax1.set_title('f(x) = x в реальном пространстве', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # График 2: f(y) = y в r-пространстве
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Y, cmap='plasma', edgecolor='none')
    ax2.set_title('f(y) = y в реальном пространстве', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(y)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    # График 3: f(kx) = kx в k-пространстве (центрированное)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    KX_shifted = np.fft.fftshift(KX)
    KY_shifted = np.fft.fftshift(KY)
    surf3 = ax3.plot_surface(KX_shifted, KY_shifted, KX_shifted,
                             cmap='coolwarm', edgecolor='none')
    ax3.set_title('f(kx) = kx в k-пространстве (центрированное)', fontsize=12)
    ax3.set_xlabel('kx')
    ax3.set_ylabel('ky')
    ax3.set_zlabel('f(kx)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)

    # График 4: f(ky) = ky в k-пространстве (центрированное)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    surf4 = ax4.plot_surface(KX_shifted, KY_shifted, KY_shifted,
                             cmap='Spectral', edgecolor='none')
    ax4.set_title('f(ky) = ky в k-пространстве (центрированное)', fontsize=12)
    ax4.set_xlabel('kx')
    ax4.set_ylabel('ky')
    ax4.set_zlabel('f(ky)')
    fig.colorbar(surf4, ax=ax4, shrink=0.5)

    plt.suptitle(f'Тестирование сеток на линейных функциях (N={N}, L={Lx})',
                 fontsize=14, y=0.98)
    plt.tight_layout()


    # 3. Дополнительная проверка: сечения
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Сечение f(x)=x при y=0 (средняя строка)
    mid_y = N // 2
    axes[0, 0].plot(x, X[:, mid_y], 'bo-', label='X сетка')
    axes[0, 0].plot(x, x, 'r--', label='Теоретическая f(x)=x')
    axes[0, 0].set_title('Сечение f(x)=x при y=0')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Сечение f(y)=y при x=0
    mid_x = N // 2
    axes[0, 1].plot(y, Y[mid_x, :], 'go-', label='Y сетка')
    axes[0, 1].plot(y, y, 'r--', label='Теоретическая f(y)=y')
    axes[0, 1].set_title('Сечение f(y)=y при x=0')
    axes[0, 1].set_xlabel('y')
    axes[0, 1].set_ylabel('f(y)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Сечение f(kx)=kx при ky=0
    axes[1, 0].plot(kx, KX[:, mid_y], 'bo-', label='KX сетка')
    axes[1, 0].plot(kx, kx, 'r--', label='Теоретическая f(kx)=kx')
    axes[1, 0].set_title('Сечение f(kx)=kx при ky=0 (центрированное)')
    axes[1, 0].set_xlabel('kx')
    axes[1, 0].set_ylabel('f(kx)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Сечение f(ky)=ky при kx=0
    axes[1, 1].plot(ky, KY[mid_x, :], 'go-', label='KY сетка')
    axes[1, 1].plot(ky, ky, 'r--', label='Теоретическая f(ky)=ky')
    axes[1, 1].set_title('Сечение f(ky)=ky при kx=0 (центрированное)')
    axes[1, 1].set_xlabel('ky')
    axes[1, 1].set_ylabel('f(ky)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle('Проверка линейных зависимостей (сечения)', fontsize=14)
    plt.tight_layout()
    plt.show()

    return True


plot_1d_grids(x, kx)
plot_2d_grids(X, Y, KX, KY, K)
test_grids_with_linear_functions(N, Lx, Ly)