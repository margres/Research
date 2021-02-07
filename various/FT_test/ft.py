# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2020-09-22 15:01:17
# @Last Modified by:   lshuns
# @Last Modified time: 2020-09-24 15:18:12

### test FFT

import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.interpolate as si
import matplotlib.pyplot as plt

Ftd_ori = np.loadtxt('Ftd_1d.txt')
t_ori = np.loadtxt('t_1d.txt')

# Ftd_ori = np.loadtxt('Ftd.txt')
# t_ori = np.loadtxt('t.txt')

# 1. FFT requires equal difference
dt = 1e-3
t = np.arange(dt, t_ori[-1], dt)
# # poly smooth
# fitting_order = 15
# coefs = poly.polyfit(t_ori, Ftd_ori, fitting_order)
# Ftd = poly.polyval(t, coefs)
## interpolate
f = si.interp1d(t_ori, Ftd_ori)
Ftd = f(t)
# ##
# plt.plot(t_ori, Ftd_ori, '.')
# plt.plot(t, Ftd, '.')
# plt.show()
# plt.close()

# 2. log fit to get high t results
t_cut = 0.6
t_max = 100
tail_mask = t>t_cut
log_t = np.log(t[tail_mask])
A = np.vstack([log_t, np.ones(len(log_t))]).T
m, c = np.linalg.lstsq(A, Ftd[tail_mask], rcond=None)[0]
t_new = np.arange(t[tail_mask][0], t_max, dt)
log_t_new = np.log(t_new)
Ftd_new = m*log_t_new+c
# plt.plot(t, Ftd, '.')
# plt.plot(t_new, Ftd_new, '.')
# plt.show()
# plt.close()

## combine
t_final = np.concatenate([t, t_new[t_new>(t[-1]+dt/2)]])
Ftd_final = np.concatenate([Ftd, Ftd_new[t_new>(t[-1]+dt/2)]])
N = len(t_final)
# plt.plot(t_final, Ftd_final, '.')
# plt.show()
# plt.close()

# # 3. add zero to both sides
# N_zeros = 100
# t_tail = np.linspace(t_final[-1]+dt, t_final[-1]+dt*N_zeros, N_zeros)
# t_head = np.linspace(t_final[0]-N_zeros*dt, t_final[0]-dt, N_zeros)
# t_final = np.concatenate([t_head, t_final, t_tail])
# Ftd_final = np.concatenate([np.zeros(N_zeros), Ftd_final, np.zeros(N_zeros)])
# N = len(t_final)


# 4. FFT
## note: Ftd_final is real, so first half of the FFT series gives usable information
##      you can either remove the second half of ifft results, or use a dedicated function ihfft
Fw = np.fft.ihfft(Ftd_final)
## multiply back what is divided
Fw *= N
## multiply sampling interval to transfer sum to integral
Fw *= dt
## the corresponding frequency 
dw= 2.*np.pi/(t_final[-1]-t_final[0]) # sampling interval
freq = np.fft.rfftfreq(N, d=dt)
w = freq*2.*np.pi

# w_max = 2.*np.pi/t_final[0]
# w_max = 100
# mask = w<w_max
# w = w[mask]
# Fw = Fw[mask]
## remove first point suffered from edge effect
w = w[1:]
Fw = Fw[1:]
# plt.plot(w, np.abs(Fw), '.')
# plt.show()
# plt.close()

# constants
# Fw = Fw*w/(2j*np.pi)-Ftd_ori[0]
t2 = t_final[-1]
t1 = t_final[0]
Ftd2 = Ftd_final[-1]
Ftd1 = Ftd_final[0]
first_term = np.exp(1j*w*t2)*Ftd2 - np.exp(1j*w*t1)*Ftd1
# Fw = (first_term + Fw*w/(2j*np.pi))
# Fw = Fw*w/(2j*np.pi)-Ftd1*np.exp(1j*w*t1)/2/np.pi
Fw = Fw*w/(2j*np.pi)-Ftd_ori[0]*np.exp(1j*w*t_ori[0])/2/np.pi

# # print(t_final[1])
# # # # print(Fw)

wc = np.loadtxt('w.txt')
Fwc = np.loadtxt('Fwc.txt', dtype='cfloat')

# trancate Fw
mask = (w<wc[-1]) & (w>wc[0])
w = w[mask]
Fw = Fw[mask]

# resample wc
f = si.interp1d(wc, Fwc)
Fwc = f(w)

wa = np.arange(0.01, 200, 0.001)
Fwa = np.loadtxt('Fw_analytical.txt', dtype='cfloat')
plt.plot(wa, np.abs(Fwa))
plt.plot(w, np.abs(Fwc))
plt.plot(w, np.abs(Fw), '.')
#plt.plot(w, np.abs(Fw+Fwc)**2)
# plt.plot(wa, np.angle(Fwa))
# plt.plot(w, np.angle(Fwc))
# plt.plot(w, np.angle(Fw), '.')
plt.xscale('log')
plt.show()

