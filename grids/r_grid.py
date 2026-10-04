import numpy as np

# FFT-COMPATIBLE REAL-SPACE GRID
def build_fft_r_grid(nx, ny, lx, ly):
    """
    Build FFT-compatible real-space grid.

    Grid:
        x in [0, Lx)
        y in [0, Ly)

    Zero is located in CORNER [0,0].

    Returns
    -------
    X, Y : 2D arrays
    x, y : 1D arrays
    dx, dy : float
    """

    dx = lx / nx
    dy = ly / ny
     # область [0, L - dx]
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy

    X, Y = np.meshgrid(x, y, indexing='ij')

    return X, Y, dx, dy, x, y

def test_r_meshgrids(N, Lx, Ly):
    X, Y, x, y, dx, dy = build_fft_r_grid(N, N, Lx, Ly)
    print("#################")
    print("#################")
    print(f"build_fft_r_grid output\n")
    print("dx:\n", dx, "\ndy:\n", dy)
    print(f'X \n{X}')
    print(f"X_shifted\n{np.fft.ifftshift(X)}")
    print(f'Y \n{Y}')
    print(f"Y_shifted\n{np.fft.ifftshift(Y)}")
