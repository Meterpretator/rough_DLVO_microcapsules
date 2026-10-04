import numpy as np

def build_fft_k_grid(nx, ny, dx, dy):    # 4. SHIFTED (CENTERED) K-GRID
    """
    Build FFT-compatible reciprocal grid.

    NumPy FFT order:
        [0, +, +, ..., -, -]

    Returns
    -------
    KX, KY : 2D arrays
    K : 2D array
        Radial wave number
    kx, ky : 1D arrays
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)

    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    K = np.sqrt(KX**2 + KY**2)

    return KX, KY, K, kx, ky

def test_k_meshgrids(N, dx, dy):
    KX, KY, K, k2x, k2y = build_fft_k_grid(N, N, dx, dy)
    print("#################")
    print("#################")
    print('\n build_fft_k_grid_output:\n')
    print(f"kx_fft_modes\n{KX}")
    print(f"kx_fft_shifted\n{np.fft.fftshift(np.array(KX))}")
    print(f"ky_fft_modes\n{KY}")
    print(f"ky_fft_shifted\n{np.fft.fftshift(np.array(KY))}")
    print(f"k_fft_shifted\n{np.fft.fftshift(np.array(K))}")
