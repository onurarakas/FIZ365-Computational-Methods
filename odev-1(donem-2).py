import numpy as np

import matplotlib.pyplot as plt

import bilYonMod as bil

import scipy.integrate as spint

plt.style.use('dark_background')



"Analitik Çözüm:"
"y''(t) = t^3 + t + 5 => y'(t) = (t^4)/4 + (t^2)/2 + 5t + c0 => y(t) = (t^5)/20 + (t^3)/6 + 5(t^2)/2 + c0*t + c1 => y(0) = 1 = c1"
"y(6) =548.8 => c0 = (548.8 - (6**5)/20 - 36 - 36*5/2 - 1)/6 = 5.499"

############################################################## 1. Soru #######################################################

def fonk_yVek_x(t, yVek):
    "y''(t) = t**3 + t + 5"
    "y'(t) = g(t)"
    "g'(t) = t**3 + t + 5"
    "bu işlemler yapılırsa 2.derece ode 2 tane 1. derece ode ye dönüşür"
    dydt = yVek[1]
    return np.array([dydt , t**3 + t + 5])

y0 = np.array([1, 5])  

t_tum = np.linspace(0, 10, 100)

t0 = 0

tson = 10

y_tum_scipy = spint.solve_ivp(fonk_yVek_x, (t0,tson), y0, t_eval=t_tum, method='RK45')

def analitik(t):
    return ((t**5)/20 + (t**3)/6 + (5*t**2)/2 + 5*t + 1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(y_tum_scipy.y[0], "ro", label='solve_ivp RK45 sonucu')
plt.plot(analitik(t_tum), label = 'analitik çözüm')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title("1.soru")


##################################################################### 2. Soru ##############################################################










##################################################################### 3. Soru ##############################################################
"Kaynaklar:"
"1-)https://youtu.be/ay0zZ8SUMSk?si=VRgfsYAsZRocQnxR"
"2-)https://youtu.be/o1Rt9zYptUM?si=SvC_YvhxfBEerAFK"

def second_dev(x):
    return x**3 + x + 5

h = 0.05

t = np.arange(h,6.0, 0.05)

m= len(t) 
L = 1
katsayi_mat = np.eye(m, m, k=-1) + -2.0*np.eye(m, m) + np.eye(m, m, k=1)
# k=1 veya -1 ile diagonal olması gereken den daha ileri veya geri kaydırabiliyorum 
# https://stackoverflow.com/questions/47761102/block-tridiagonal-matrices-how-to-program-this-kind-of-matrix



sonuc_vec = np.zeros([len(t)])

sonuc_vec[0] = (h**2 )* second_dev(t[0]) -1

sonuc_vec[-1] = (h**2 )* second_dev(t[-1]) -548.8



for i in range(1,len(t)-1):
    sonuc_vec[i] = (h**2)* second_dev(t[i])


solution_hopefully = np.linalg.solve(katsayi_mat, sonuc_vec)

def analitik(t):
    return (1/20) * t**5 + (1/2) * t**3 + (5/2) * t**2 - 5.499 * t + 1


plt.subplot(1, 2, 2)
plt.plot(t, solution_hopefully, '.', label='Sonlu Farklar', color = "purple")
plt.plot(t, analitik(t), label = 'analitik çözüm', color = "green")
plt.xlabel('t')
plt.ylabel('y')
plt.legend()

plt.title("3.soru")
plt.show()