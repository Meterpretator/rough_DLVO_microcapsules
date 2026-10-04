import numpy as np
import math

def build_1d_r_grid(N, L):
    """
    Построение одномерной центрированной пространственной сетки.

    Параметры:
        N : int – количество точек
        L : float – физический размер области

    Возвращает:
        x : np.ndarray (N,) – одномерные координаты
        dx : float – шаг дискретизации
    """
    dx = L / N
    x = np.linspace(-L / 2, L / 2 - dx, N)
    return x, dx


def build_2d_r_grid(N, Lx, Ly):
    """
    Построение двумерной центрированной пространственной сетки.

    Параметры:
        N : int – количество точек по каждому измерению (квадратная сетка N x N)
        Lx, Ly : float – физические размеры области по осям X и Y

    Возвращает:
        X, Y : np.ndarray (N, N) – двумерные массивы координат (центрированные)
        x, y : np.ndarray (N,) – одномерные координаты
        dx, dy : float – шаги дискретизации
    """
    # Одномерные координаты
    x, dx = build_1d_r_grid(N, Lx)
    y, dy = build_1d_r_grid(N, Ly)

    # Двумерная сетка (индексное соглашение 'ij')
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y, x, y, dx, dy


def build_1d_k_grid(nx, lx):
    """
    Build FFT-compatible 1D reciprocal grid.

    Parameters
    ----------
    nx : int
        Number of real-space points.
    lx : float
        Physical system size.

    Returns
    -------
    kx : np.ndarray
        FFT-compatible grids (1D array).
    """
    # Шаг в k-пространстве: Δk = 2π / Lx
    dkx = (2.0 * math.pi) / lx

    half = nx // 2

    if nx % 2 == 0:
        # Чётное количество точек
        # Положительные частоты (включая ноль) до N/2 - 1
        kx = np.arange(half) * dkx
        # Отрицательные частоты от -N/2 до -1
        kx_neg = np.arange(-half, 0) * dkx
        kx = np.concatenate([kx, kx_neg])
    else:
        # Нечётное количество точек
        kx_pos = np.arange(half + 1) * dkx
        kx_neg = np.arange(-half, 0) * dkx
        kx = np.concatenate([kx_pos, kx_neg])

    return kx


def build_2d_k_grid(nx, ny, lx, ly):
    """
    Build 2D FFT-compatible reciprocal grid.

    Parameters
    ----------
    nx, ny : int
        Number of points in x and y directions.
    lx, ly : float
        Physical system sizes in x and y.

    Returns
    -------
    KX, KY, K : np.ndarray (2D)
        Grids of wave numbers and radial magnitudes.
    """
    # Получаем одномерные k-сетки
    kx = build_1d_k_grid(nx, lx)
    ky = build_1d_k_grid(ny, ly)

    # Создаём 2D сетки с помощью meshgrid (индексное соглашение 'ij')
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Радиальное волновое число
    K = np.sqrt(KX**2 + KY**2)

    return KX, KY, K