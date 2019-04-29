import numpy as np
import matplotlib.pyplot as plt
from transforms import *
# Number of samplepoints
N = 1024

# sample spacing
T = 1.0 / 800.0
T2 = T*100

x = np.linspace(0.0, N*T, N)
y = np.sin(20.0 * 2.0*np.pi*x) + np.sin(80.0 * 2.0*np.pi*x)

m = np.arange(0,N,1)
n = np.arange(0,N,1)
F_y = form_F_li(1.0, y, np.pi*m, n/N, 0, N)
F_y = F_y
f = np.linspace(0.0, 1.0/(2*T), N//2)

w_fake = np.ones((N))
back = form_P_meas(w_fake, F_y, np.pi*m, n/N, 0, N)
dif = np.sqrt((back-y)**2)
plt.plot(x, back)
#plt.plot(x, dif)
#plt.plot(f[:N//2], 2./N * np.abs(F_y[:N//2]))
plt.show(block=True)