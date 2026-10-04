import numpy as np
import matplotlib.pyplot as plt

from k_grid import build_fft_k_grid
from r_grid import build_fft_r_grid


def test_fft_inverse_identity(
    N,
    Lx,
    Ly,
    seed=42,
    tol=1e-12
):
    """
    Проверка FFT -> inverse FFT identity.

    Проверяется:

        ifft2(fft2(f)) = f

    с точностью floating-point arithmetic.
    """

    #
    # 1. Build real-space grid
    #

    X, Y, dx, dy, x, y = build_fft_r_grid(
        N, N, Lx, Ly
    )

    #
    # 2. Generate random real field
    #

    np.random.seed(seed)

    f_original = np.random.randn(N, N)

    #
    # 3. Forward FFT
    #

    F = np.fft.fft2(f_original)

    #
    # 4. Inverse FFT
    #

    f_reconstructed = np.fft.ifft2(F)

    #
    # 5. Compute errors
    #

    abs_error = np.abs(
        f_original - f_reconstructed
    )

    max_abs_error = np.max(abs_error)

    mean_abs_error = np.mean(abs_error)

    #
    # 6. Imaginary part check
    #

    max_imag = np.max(
        np.abs(np.imag(f_reconstructed))
    )

    #
    # 7. Diagnostics
    #

    print("\n=== FFT Inverse Identity Test ===")

    print(f"N           = {N}")

    print(f"Lx, Ly      = {Lx}, {Ly}")

    print(f"dx, dy      = {dx}, {dy}")

    print(f"\nmax abs error   = {max_abs_error:.16e}")

    print(f"mean abs error  = {mean_abs_error:.16e}")

    print(f"max imag part   = {max_imag:.16e}")

    #
    # 8. Assertions
    #

    assert max_abs_error < tol, (
        f"FFT inverse identity FAILED: "
        f"max_abs_error={max_abs_error}"
    )

    assert max_imag < tol, (
        f"Imaginary part too large: "
        f"{max_imag}"
    )

    print("\n✅ FFT inverse identity PASSED")

    #
    # ==========================================================
    # 9. Visualization
    # ==========================================================
    #

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    #
    # Original field
    #

    im0 = axes[0, 0].imshow(
        f_original,
        origin='lower',
        extent=[0, Lx, 0, Ly]
    )

    axes[0, 0].set_title("Original field")

    axes[0, 0].set_xlabel("x")

    axes[0, 0].set_ylabel("y")

    plt.colorbar(im0, ax=axes[0, 0])

    #
    # Reconstructed field
    #

    im1 = axes[0, 1].imshow(
        np.real(f_reconstructed),
        origin='lower',
        extent=[0, Lx, 0, Ly]
    )

    axes[0, 1].set_title("Reconstructed field")

    axes[0, 1].set_xlabel("x")

    axes[0, 1].set_ylabel("y")

    plt.colorbar(im1, ax=axes[0, 1])

    #
    # Absolute error
    #

    im2 = axes[1, 0].imshow(
        abs_error,
        origin='lower',
        extent=[0, Lx, 0, Ly]
    )

    axes[1, 0].set_title("Absolute reconstruction error")

    axes[1, 0].set_xlabel("x")

    axes[1, 0].set_ylabel("y")

    plt.colorbar(im2, ax=axes[1, 0])

    #
    # Imaginary part
    #

    im3 = axes[1, 1].imshow(
        np.imag(f_reconstructed),
        origin='lower',
        extent=[0, Lx, 0, Ly]
    )

    axes[1, 1].set_title("Imaginary part after IFFT")

    axes[1, 1].set_xlabel("x")

    axes[1, 1].set_ylabel("y")

    plt.colorbar(im3, ax=axes[1, 1])

    #
    # Figure title
    #

    plt.suptitle(
        "FFT ⇄ inverse FFT identity test",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()

    return True

def test_parseval_correct(
    N,
    Lx,
    Ly,
    seed=42,
    tol=1e-12
):
    """
    Проверка теоремы Парсеваля
    + визуализация энергий.

    Проверяется:

        sum |f(r)|^2 dxdy
            =
        (1/N^2) sum |F(k)|^2 dxdy
    """

    #
    # 1. Build grids
    #

    X, Y, dx, dy, x, y = build_fft_r_grid(
        N, N, Lx, Ly
    )

    KX, KY, K, kx, ky = build_fft_k_grid(
        N, N, dx, dy
    )

    #
    # 2. Random field
    #

    np.random.seed(seed)

    f_r = np.random.randn(N, N)

    #
    # 3. Forward FFT
    #

    F = np.fft.fft2(f_r)

    #
    # 4. Energy densities
    #

    energy_r_density = np.abs(f_r)**2

    energy_k_density = np.abs(F)**2 / (N * N)

    #
    # 5. Total energies
    #

    E_r = np.sum(energy_r_density) * dx * dy

    E_k = np.sum(energy_k_density) * dx * dy

    #
    # 6. Errors
    #

    abs_error = np.abs(E_r - E_k)

    rel_error = abs_error / E_r

    #
    # 7. Diagnostics
    #

    print("\n=== Parseval Test ===")

    print(f"N           = {N}")
    print(f"Lx, Ly      = {Lx}, {Ly}")
    print(f"dx, dy      = {dx}, {dy}")

    print(f"\nE_real      = {E_r:.16e}")
    print(f"E_fourier   = {E_k:.16e}")

    print(f"\nabs error   = {abs_error:.16e}")
    print(f"rel error   = {rel_error:.16e}")

    #
    # 8. Parseval check
    #

    assert rel_error < tol, (
        f"Parseval FAILED: rel_error={rel_error}"
    )

    print("\n✅ Parseval theorem PASSED")
    print("dx =", dx)

    print("dk =", 2 * np.pi / Lx)

    print("kx min/max =", kx.min(), kx.max())

    print("expected kmax =", np.pi / dx)
    #
    # ==========================================================
    # 9. Visualization
    # ==========================================================
    #

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    #
    # Real-space field
    #

    im0 = axes[0, 0].imshow(
        f_r,
        origin='lower',
        extent=[0, Lx, 0, Ly]
    )

    axes[0, 0].set_title("Random field $f(x,y)$")

    axes[0, 0].set_xlabel("x")

    axes[0, 0].set_ylabel("y")

    plt.colorbar(im0, ax=axes[0, 0])

    #
    # Real-space energy density
    #

    im1 = axes[0, 1].imshow(
        energy_r_density,
        origin='lower',
        extent=[0, Lx, 0, Ly]
    )

    axes[0, 1].set_title(r"Real-space energy density $|f(x,y)|^2$")

    axes[0, 1].set_xlabel("x")

    axes[0, 1].set_ylabel("y")

    plt.colorbar(im1, ax=axes[0, 1])

    #
    # Fourier energy density
    #

    spectral_energy_shifted = np.fft.fftshift(
        energy_k_density
    )

    kx_shifted = np.fft.fftshift(kx)

    ky_shifted = np.fft.fftshift(ky)

    im2 = axes[1, 0].imshow(
        spectral_energy_shifted,
        origin='lower',
        extent=[
            kx_shifted[0],
            kx_shifted[-1],
            ky_shifted[0],
            ky_shifted[-1]
        ]
    )

    axes[1, 0].set_title(
        r"Fourier energy density $|F(k_x,k_y)|^2/N^2$"
    )

    axes[1, 0].set_xlabel(r"$k_x$")

    axes[1, 0].set_ylabel(r"$k_y$")

    plt.colorbar(im2, ax=axes[1, 0])

    #
    # Energy comparison
    #

    axes[1, 1].bar(
        ["Real space", "Fourier space"],
        [E_r, E_k]
    )

    axes[1, 1].set_title("Parseval energy comparison")

    axes[1, 1].set_ylabel("Total energy")

    #
    # Figure title
    #

    plt.suptitle(
        f"Parseval theorem test (N={N})",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()
    return True

N = 65
Lx=Ly=10

test_parseval_correct(N, Lx, Ly)
test_fft_inverse_identity(N, Lx, Ly)