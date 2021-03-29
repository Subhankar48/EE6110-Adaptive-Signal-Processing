import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from tqdm import tqdm

# LMS
M_TAPS = 4
N_SAMPLES = 6000
NOISE_MEAN = 0
DATA_MEAN = 0
NOISE_STD = 0.1
DATA_STD = 1
MU_LMS = 0.01

np.random.seed(1300)


# LMS experiment 1
u = np.random.normal(DATA_MEAN, DATA_STD, N_SAMPLES)
# to match the way it is defined in the book
c = np.array([1, 0.5, -1, 2])[::-1]
d = convolve(u, c, mode='valid')
noise = np.random.normal(NOISE_MEAN, NOISE_STD, np.shape(d))
d += noise

w = np.zeros((M_TAPS, 1), dtype=np.float64)
e = []

for i in tqdm(range(len(d))):
    u_i = u[i:i+M_TAPS].reshape(-1, 1)
    e.append((d[i]-np.sum(u_i*w))**2)
    w = w + MU_LMS*(d[i]-np.sum(u_i*w))*u_i

e.append((d[i]-np.sum(u_i*w))**2)


plt.plot(e[:200])
plt.title("LMS")
plt.ylabel(r'$|e(i)|^{2}$', fontsize=13)
plt.xlabel("Number of iterations (i)", fontsize=13)
plt.grid(True)
plt.show()


# LMS experiment 2
u = np.random.normal(DATA_MEAN, DATA_STD, N_SAMPLES)
# to match the way it is defined in the book
c = np.array([1, 0.5, -1, 2])[::-1]
d = convolve(u, c, mode='valid')
noise = np.random.normal(NOISE_MEAN, NOISE_STD, np.shape(d))
d += noise

w = np.zeros((M_TAPS, 1), dtype=np.float64)
e = []

for i in tqdm(range(len(d))):
    u_i = u[i:i+M_TAPS].reshape(-1, 1)
    e.append((d[i]-np.sum(u_i*w))**2)
    w = w + MU_LMS*(d[i]-np.sum(u_i*w))*u_i

e.append((d[i]-np.sum(u_i*w))**2)

plt.plot(e[:200])
plt.title("LMS")
plt.ylabel(r'$|e(i)|^{2}$', fontsize=13)
plt.xlabel("Number of iterations (i)", fontsize=13)
plt.grid(True)
plt.show()
# end of LMS


# e-NLMS
M_TAPS = 4
N_SAMPLES = 603
NOISE_MEAN = 0
DATA_MEAN = 0
NOISE_STD = 0.1
DATA_STD = 1
MU_ELMS = 0.2
EPSILON = 0.001

np.random.seed(1300)


# e-NLMS experiment 1
u = np.random.normal(DATA_MEAN, DATA_STD, N_SAMPLES)
# to match the way it is defined in the book
c = np.array([1, 0.5, -1, 2])[::-1]
d = convolve(u, c, mode='valid')
noise = np.random.normal(NOISE_MEAN, NOISE_STD, np.shape(d))
d += noise

w = np.zeros((M_TAPS, 1), dtype=np.float64)
e = []

for i in tqdm(range(len(d))):
    u_i = u[i:i+M_TAPS].reshape(-1, 1)
    e.append((d[i]-np.sum(u_i*w))**2)
    w = w + MU_ELMS*(d[i]-np.sum(u_i*w))*u_i/(EPSILON+np.sum(u_i**2))

e.append((d[i]-np.sum(u_i*w))**2)

plt.plot(e[:200])
plt.title(r'$\epsilon$-NLMS')
plt.ylabel(r'$|e(i)|^{2}$', fontsize=13)
plt.xlabel("Number of iterations (i)", fontsize=13)
plt.grid(True)
plt.show()


# e-NLMS experiment 2
u = np.random.normal(DATA_MEAN, DATA_STD, N_SAMPLES)
# to match the way it is defined in the book
c = np.array([1, 0.5, -1, 2])[::-1]
d = convolve(u, c, mode='valid')
noise = np.random.normal(NOISE_MEAN, NOISE_STD, np.shape(d))
d += noise

w = np.zeros((M_TAPS, 1), dtype=np.float64)
e = []

for i in tqdm(range(len(d))):
    u_i = u[i:i+M_TAPS].reshape(-1, 1)
    e.append((d[i]-np.sum(u_i*w))**2)
    w = w + MU_ELMS*(d[i]-np.sum(u_i*w))*u_i/(EPSILON+np.sum(u_i**2))

e.append((d[i]-np.sum(u_i*w))**2)

plt.plot(e[:200])
plt.title(r'$\epsilon$-NLMS')
plt.ylabel(r'$|e(i)|^{2}$', fontsize=13)
plt.xlabel("Number of iterations (i)", fontsize=13)
plt.grid(True)
plt.show()
# end of e-NLMS


# RLS
M_TAPS = 4
N_SAMPLES = 603
NOISE_MEAN = 0
DATA_MEAN = 0
NOISE_STD = 0.1
DATA_STD = 1
EPSILON = 0.001
LAMBDA = 0.995

np.random.seed(1310)

# RLS experiment 1
u = np.random.normal(DATA_MEAN, DATA_STD, N_SAMPLES)
# to match the way it is defined in the book
c = np.array([1, 0.5, -1, 2])[::-1]
d = convolve(u, c, mode='valid')
noise = np.random.normal(NOISE_MEAN, NOISE_STD, np.shape(d))
d += noise

w = np.zeros((M_TAPS, 1), dtype=np.float64)
e = []
P = np.eye(M_TAPS)/EPSILON

for i in tqdm(range(len(d))):
    u_i = u[i:i+M_TAPS].reshape(1, -1)
    e.append((d[i]-np.asscalar(u_i@w))**2)
    P = (1/LAMBDA)*(P-((P@(u_i.T)@u_i@P)/(LAMBDA+u_i@P@(u_i.T))))
    w = w + P@(u_i.T)*(d[i]-u_i@w)

e.append((d[i]-np.sum(u_i@w))**2)


plt.plot(e[:200])
plt.title('RLS')
plt.ylabel(r'$|e(i)|^{2}$', fontsize=13)
plt.xlabel("Number of iterations (i)", fontsize=13)
plt.grid(True)
plt.show()


# RLS experiment 1
u = np.random.normal(DATA_MEAN, DATA_STD, N_SAMPLES)
# to match the way it is defined in the book
c = np.array([1, 0.5, -1, 2])[::-1]
d = convolve(u, c, mode='valid')
noise = np.random.normal(NOISE_MEAN, NOISE_STD, np.shape(d))
d += noise

w = np.zeros((M_TAPS, 1), dtype=np.float64)
e = []
P = np.eye(M_TAPS)/EPSILON

for i in tqdm(range(len(d))):
    u_i = u[i:i+M_TAPS].reshape(1, -1)
    e.append((d[i]-np.asscalar(u_i@w))**2)
    P = (1/LAMBDA)*(P-((P@(u_i.T)@u_i@P)/(LAMBDA+u_i@P@(u_i.T))))
    w = w + P@(u_i.T)*(d[i]-u_i@w)

e.append((d[i]-np.sum(u_i@w))**2)

plt.plot(e[:200])
plt.title('RLS')
plt.ylabel(r'$|e(i)|^{2}$', fontsize=13)
plt.xlabel("Number of iterations (i)", fontsize=13)
plt.grid(True)
plt.show()
# end of RLS


# ensemble average
N_SAMPLES = 603
N_EXPERIMENTS = 300
e_lms = np.zeros((N_EXPERIMENTS, N_SAMPLES+1-M_TAPS), dtype=np.float64)
e_nlms = np.zeros_like(e_lms)
e_rls = np.zeros_like(e_lms)

for n in tqdm(range(N_EXPERIMENTS)):
    u = np.random.normal(DATA_MEAN, DATA_STD, N_SAMPLES)
    d = convolve(u, c, mode='valid')
    noise = np.random.normal(NOISE_MEAN, NOISE_STD, np.shape(d))
    d += noise

    w = np.zeros((M_TAPS, 1), dtype=np.float64)

    for i in range(len(d)):
        u_i = u[i:i+M_TAPS].reshape(-1, 1)
        e_lms[n, i] = (d[i]-np.sum(u_i*w))**2
        w = w + MU_LMS*(d[i]-np.sum(u_i*w))*u_i

    w = np.zeros((M_TAPS, 1), dtype=np.float64)

    for i in range(len(d)):
        u_i = u[i:i+M_TAPS].reshape(-1, 1)
        e_nlms[n, i] = (d[i]-np.sum(u_i*w))**2
        w = w + MU_ELMS*(d[i]-np.sum(u_i*w))*u_i/(EPSILON+np.sum(u_i**2))

    w = np.zeros((M_TAPS, 1), dtype=np.float64)
    P = np.eye(M_TAPS)/EPSILON

    for i in range(len(d)):
        u_i = u[i:i+M_TAPS].reshape(1, -1)
        e_rls[n, i] = (d[i]-np.asscalar(u_i@w))**2
        P = (1/LAMBDA)*(P-((P@(u_i.T)@u_i@P)/(LAMBDA+u_i@P@(u_i.T))))
        w = w + P@(u_i.T)*(d[i]-u_i@w)


plt.plot(10*np.log10(np.mean(e_lms, axis=0)))
plt.plot(10*np.log10(np.mean(e_nlms, axis=0)))
plt.plot(10*np.log10(np.mean(e_rls, axis=0)))
plt.title("Ensemble average learning curve.")
plt.ylabel("MSE (dB)", fontsize=13)
plt.xlabel("Number of iterations (i)", fontsize=13)
plt.legend(['LMS', 'ε-NLMS', 'RLS'])
plt.grid(True)
plt.show()
