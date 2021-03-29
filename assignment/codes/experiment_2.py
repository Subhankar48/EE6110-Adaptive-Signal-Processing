import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve


c = np.array([1, -2, 3, -4, 5, 5, -4, 3, -2, 1])[::-1]/100
M_TAPS = len(c)
N_SAMPLES = int(4e6)
NOISE_MEAN = 0
DATA_MEAN = np.zeros(M_TAPS)
NOISE_STD = np.sqrt(0.001)
DATA_STD = np.diag([2, 3, 4, 5, 6, 7, 8, 9, 10, 5])/100
N_EXPERIMENTS = 3
N_SAMPLES_FOR_CURVE = 500
START_MU_VAL = 1e-4
END_MU_VAL = 1e-2
STEPS = 3
mu_vals = np.geomspace(START_MU_VAL, END_MU_VAL, STEPS)

np.random.seed(1299)


# exp_vals = np.zeros_like(mu_vals)

# for n in tqdm(range(N_EXPERIMENTS)):
#     u = np.random.multivariate_normal(DATA_MEAN, DATA_STD, N_SAMPLES)
#     noise = np.random.normal(NOISE_MEAN, NOISE_STD, N_SAMPLES)
#     d = u.dot(c)+noise
#     for k in tqdm(range(STEPS)):
#         w = np.zeros((M_TAPS,1), dtype=np.float64)
#         e = np.zeros((N_EXPERIMENTS, N_SAMPLES))
#         mu = mu_vals[k]
#         for i in tqdm(range(N_SAMPLES)):
#             u_i = u[i].reshape(-1,1)
#             w = w + mu*(d[i]-np.sum(u_i*w))*u_i
#             e[n,i] = (d[i]-np.sum(u_i*w))**2

#         exp_vals[k] = np.mean(np.mean(e[:, -N_SAMPLES_FOR_CURVE:], axis=0))


# plt.semilogx(mu_vals, 10*np.log10(exp_vals), '-o')
# plt.grid(True, which='both')
# plt.ylabel("MSE(dB)", fontsize=13)
# plt.xlabel(r"$\mu$ value")
# plt.title("LMS: Gaussian regressor without shift structure")
# plt.show()


c = np.array([1, -2, 3, -4, 5, -5, 4, -3, 2, -1])[::-1]/10
M_TAPS = len(c)
N_SAMPLES = int(4e5)
NOISE_MEAN = 0
DATA_MEAN = 0
NOISE_STD = np.sqrt(0.001)
DATA_STD = 1
N_EXPERIMENTS = 5
N_SAMPLES_FOR_CURVE = 5000
START_MU_VAL = 1e-4
END_MU_VAL = 1e-2
STEPS = 3
mu_vals = np.geomspace(START_MU_VAL, END_MU_VAL, STEPS)
a = 0.8

exp_vals = np.zeros_like(mu_vals)

# np.random.seed(1299)
e = np.zeros((N_EXPERIMENTS, STEPS, N_SAMPLES), dtype=np.float)

for n in tqdm(range(N_EXPERIMENTS)):
    s = np.random.normal(DATA_MEAN, DATA_STD, N_SAMPLES+M_TAPS)
    u = np.zeros_like(s)
    u[0] = s[0]
    for i in range(1, len(s)):
        u[i] = a*u[i-1]+np.sqrt(1-a**2)*s[i]
    u = u[1:]
    d = convolve(u, c, mode='valid')
    noise = np.random.normal(NOISE_MEAN, NOISE_STD, np.shape(d))
    d += noise
    for k in tqdm(range(STEPS)):
        w = np.zeros((M_TAPS, 1), dtype=np.float)
        mu = mu_vals[k]
        for i in tqdm(range(len(d))):
            u_i = u[i:i+M_TAPS].reshape(-1, 1)
            w = w + mu*(d[i]-np.sum(u_i*w))*u_i
            e[n, k, i] = (d[i]-np.sum(u_i*w))**2


for k in range(STEPS):
    exp_vals[k] = np.mean(e[:, k, -N_SAMPLES_FOR_CURVE:])

plt.semilogx(mu_vals, 10*np.log10(exp_vals), '-o')
plt.grid(True, which='both')
plt.ylabel("MSE(dB)", fontsize=13)
plt.xlabel(r"$\mu$ value")
plt.title("LMS: Gaussian regressor without shift structure")
plt.show()
