import numpy as np
import matplotlib.pyplot as plt

def build_centered_grid(N, L):
    dx = L / N
    x = np.linspace(-L/2, L/2 - dx, N)
    return x, dx

def build_target_acf(N, L, rms, clx, cly):
    x, dx = build_centered_grid(N, L)
    X, Y = np.meshgrid(x, x, indexing='ij')
    r = np.sqrt((X/clx)**2 + (Y/cly)**2)
    R = rms**2 * np.exp(-r**2)   # гауссова АКФ
    return R, x

def compute_spectrum(R):
    R_shifted = np.fft.ifftshift(R)
    R_fft = np.fft.fft2(R_shifted)
    return R_fft, R_shifted

# Параметры
N = 256
L = 50.0
rms = 1.0
clx = cly = 10.0

# Построение АКФ и спектра
R, x = build_target_acf(N, L, rms, clx, cly)
R_fft, R_shifted = compute_spectrum(R)
R_fft_real = np.real(R_fft)
R_fft_imag = np.imag(R_fft)
R_fft_mag = np.abs(R_fft)

# Профили через центр
center = N // 2
profile_r_center = R[center, :]
profile_r_shifted_center = R_shifted[center, :]
profile_fft_real_center = R_fft_real[center, :]
profile_fft_imag_center = R_fft_imag[center, :]

# Волновые числа для оси k
dk = 2 * np.pi / L
k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi   # стандартный порядок (после fftshift для графика)

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. АКФ центрированная
im1 = axes[0, 0].imshow(R, cmap='viridis', extent=[x[0], x[-1], x[0], x[-1]])
axes[0, 0].set_title("1. АКФ в r-пространстве\n(центрированная, нуль в центре)")
axes[0, 0].set_xlabel("τx")
axes[0, 0].set_ylabel("τy")
plt.colorbar(im1, ax=axes[0, 0])

# 2. АКФ после ifftshift (перед БПФ)
im2 = axes[0, 1].imshow(R_shifted, cmap='viridis')
axes[0, 1].set_title("2. АКФ после ifftshift\n(нуль в левом верхнем углу)")
axes[0, 1].set_xlabel("τx (индекс)")
axes[0, 1].set_ylabel("τy (индекс)")
plt.colorbar(im2, ax=axes[0, 1])

# 3. Спектр (PSD) – действительная часть
im3 = axes[0, 2].imshow(R_fft_real, cmap='RdBu')
axes[0, 2].set_title("3. Действительная часть спектра\n(после БПФ)")
axes[0, 2].set_xlabel("k_x (индекс)")
axes[0, 2].set_ylabel("k_y (индекс)")
plt.colorbar(im3, ax=axes[0, 2])

# 4. Профиль АКФ через центр (r-пространство)
axes[1, 0].plot(x, profile_r_center, 'b-', label='АКФ (центрированная)')
axes[1, 0].set_title("4. Профиль АКФ через центр\n(до сдвига)")
axes[1, 0].set_xlabel("τ")
axes[1, 0].set_ylabel("R(τ,0)")
axes[1, 0].grid(True)

# 5. Профиль спектра (действительная и мнимая части)
# Для графика по k удобно использовать центрированные частоты
k_shifted = np.fft.fftshift(k)
profile_real_shifted = np.fft.fftshift(profile_fft_real_center)
profile_imag_shifted = np.fft.fftshift(profile_fft_imag_center)

axes[1, 1].plot(k_shifted, profile_real_shifted, 'b-', label='Re(FFT)')
axes[1, 1].plot(k_shifted, profile_imag_shifted, 'r--', label='Im(FFT)')
axes[1, 1].set_title("5. Профиль спектра\n(действительная и мнимая части)")
axes[1, 1].set_xlabel("k (рад/ед.длины)")
axes[1, 1].set_ylabel("F(k)")
axes[1, 1].legend()
axes[1, 1].grid(True)

# 6. Мнимая часть спектра (отдельно, чтобы оценить её величину)
im6 = axes[1, 2].imshow(R_fft_imag, cmap='RdBu')
axes[1, 2].set_title(f"6. Мнимая часть спектра\n(макс |Im| = {np.max(np.abs(R_fft_imag)):.2e})")
axes[1, 2].set_xlabel("k_x (индекс)")
axes[1, 2].set_ylabel("k_y (индекс)")
plt.colorbar(im6, ax=axes[1, 2])

plt.tight_layout()
plt.show()

# Дополнительная информация
print(f"Максимальное значение |Im(F)|: {np.max(np.abs(R_fft_imag)):.2e}")
print(f"Отношение max|Im| / max|Re|: {np.max(np.abs(R_fft_imag)) / np.max(np.abs(R_fft_real)):.2e}")