import numpy as np
from tqdm import tqdm
import multiprocessing


c = np.array([1, -2, 3, -4, 5, -5, 4, -3, 2, -1])[::-1]
M_TAPS = len(c)
N_SAMPLES = int(4e5)
NOISE_MEAN = 0
DATA_MEAN = np.zeros(M_TAPS)
NOISE_STD = np.sqrt(0.001)
DATA_STD = np.diag([2, 3, 4, 5, 6, 7, 8, 9, 10, 5])/10
N_EXPERIMENTS = 80
N_SAMPLES_FOR_CURVE = 500

np.random.seed(1331)

all_u = np.random.multivariate_normal(DATA_MEAN, DATA_STD, (N_EXPERIMENTS, N_SAMPLES))
all_d = np.zeros((N_EXPERIMENTS, N_SAMPLES), dtype=np.float)
all_noise = np.random.normal(NOISE_MEAN, NOISE_STD, (N_EXPERIMENTS, N_SAMPLES))

for k in range(N_EXPERIMENTS):
    u = all_u[k]
    noise = all_noise[k]
    d = u.dot(c)+noise
    all_d[k] = d

print("mu,mse")

mu_vals = [0.0001, 0.00011721, 0.00013738, 0.00016103, 0.00018874,
           0.00022122, 0.00025929, 0.00030392, 0.00035622, 0.00041753,
           0.00048939, 0.00057362, 0.00067234, 0.00078805, 0.00092367,
           0.00108264, 0.00126896, 0.00148735, 0.00174333, 0.00204336,
           0.00239503, 0.00280722, 0.00329034, 0.00385662, 0.00452035,
           0.00529832, 0.00621017, 0.00727895, 0.00853168, 0.01]


def lms(mu):
    e = np.zeros((N_EXPERIMENTS, N_SAMPLES), dtype=np.float)
    for n in range(N_EXPERIMENTS):
        u = all_u[n]
        d = all_d[n]
        w = np.random.randint(min(c), max(c), (M_TAPS, 1)).astype('float64')
        for i in range(len(d)):
            u_i = u[i].reshape(-1, 1)
            e[n, i] = (d[i]-np.sum(u_i*w))**2
            w = w + mu*(d[i]-np.sum(u_i*w))*u_i
    
    mse = np.mean(e[:, N_SAMPLES-N_SAMPLES_FOR_CURVE:])
    print(f"{mu},{mse}")

"""
Uncomment and run only those many processes as the number of cores you have.
"""

t1 = multiprocessing.Process(target=lms, args=(mu_vals[0],))
t2 = multiprocessing.Process(target=lms, args=(mu_vals[1],))
t3 = multiprocessing.Process(target=lms, args=(mu_vals[2],))
t4 = multiprocessing.Process(target=lms, args=(mu_vals[3],))
t5 = multiprocessing.Process(target=lms, args=(mu_vals[4],))
t6 = multiprocessing.Process(target=lms, args=(mu_vals[5],))
t7 = multiprocessing.Process(target=lms, args=(mu_vals[6],))
t8 = multiprocessing.Process(target=lms, args=(mu_vals[7],))
# t9 = multiprocessing.Process(target=lms, args=(mu_vals[8],))
# t10 = multiprocessing.Process(target=lms, args=(mu_vals[9],))
# t11 = multiprocessing.Process(target=lms, args=(mu_vals[10],))
# t12 = multiprocessing.Process(target=lms, args=(mu_vals[11],))
# t13 = multiprocessing.Process(target=lms, args=(mu_vals[12],))
# t14 = multiprocessing.Process(target=lms, args=(mu_vals[13],))
# t15 = multiprocessing.Process(target=lms, args=(mu_vals[14],))
# t16 = multiprocessing.Process(target=lms, args=(mu_vals[15],))
# t17 = multiprocessing.Process(target=lms, args=(mu_vals[16],))
# t18 = multiprocessing.Process(target=lms, args=(mu_vals[17],))
# t19 = multiprocessing.Process(target=lms, args=(mu_vals[18],))
# t20 = multiprocessing.Process(target=lms, args=(mu_vals[19],))
# t21 = multiprocessing.Process(target=lms, args=(mu_vals[20],))
# t22 = multiprocessing.Process(target=lms, args=(mu_vals[21],))
# t23 = multiprocessing.Process(target=lms, args=(mu_vals[22],))
# t24 = multiprocessing.Process(target=lms, args=(mu_vals[23],))
# t25 = multiprocessing.Process(target=lms, args=(mu_vals[24],))
# t26 = multiprocessing.Process(target=lms, args=(mu_vals[25],))
# t27 = multiprocessing.Process(target=lms, args=(mu_vals[26],))
# t28 = multiprocessing.Process(target=lms, args=(mu_vals[27],))
# t29 = multiprocessing.Process(target=lms, args=(mu_vals[28],))
# t30 = multiprocessing.Process(target=lms, args=(mu_vals[29],))

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t8.start()
# t9.start()
# t10.start()
# t11.start()
# t12.start()
# t13.start()
# t14.start()
# t15.start()
# t16.start()
# t17.start()
# t18.start()
# t19.start()
# t20.start()
# t21.start()
# t22.start()
# t23.start()
# t24.start()
# t25.start()
# t26.start()
# t27.start()
# t28.start()
# t29.start()
# t30.start()