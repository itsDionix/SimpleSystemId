import ctypes as C
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sio


class RecCovData(C.Structure):
    _fields_ = [("C", C.c_double),
                ("Cx", C.c_double),
                ("Cy", C.c_double),
                ("xMean", C.c_double),
                ("yMean", C.c_double),
                ("n", C.c_int)]


class RecLinRegData(C.Structure):
    _fields_ = [("m", C.c_double),
                ("c", C.c_double),
                ("CovData", C.POINTER(RecCovData))]


# class RecSysIdData(C.Structure):
#     _fields_ = [("yf", C.c_double),
#                 ("m", C.c_double),
#                 ("RegData", C.POINTER(RecLinRegData)),
#                 ("n", C.c_int)]


mylib = C.cdll.LoadLibrary("./simplesysid.so")
RecCovIter = mylib.RecursiveCovariance
RecCovIter.argtypes = [C.c_double, C.c_double,
                       C.POINTER(RecCovData)]
RecRegIter = mylib.RecursiveLinReg
RecRegIter.argtypes = [C.c_double, C.c_double,
                       C.POINTER(RecLinRegData)]
# RecSysIdIter = mylib.RecursiveSysId
# RecSysIdIter.argtypes = [C.c_double,
#                          C.POINTER(RecSysIdData)]

print(" ** Exponential identification test ** ")

x0 = 2
xf = 1
tau = 10


def real_x(t):
    return np.exp(-t/tau)*(x0 - xf) + xf


rng = np.random.default_rng()
predT = 7
totalT = predT*3
N = 200000
Ts = predT/(N-1)
# Interval between samples used for regression
dn = int(N/2)
print(f"dn is {dn:d}")
dt = dn*Ts
m = np.exp(-dt/tau)
c = xf*(1-m)
n = np.linspace(0, predT, N)
noise = rng.normal(size=N)*(x0 - real_x(predT/tau))*0.5
x = real_x(n)
y = x + noise

# fit with scipy


def f(t, x0, xf, tau):
    return np.exp(-t/tau)*(x0 - xf) + xf


fit = sio.curve_fit(f, n, y, (x0, xf, tau))[0]
fy0, fyf, ftau = fit

fig = plt.figure()
ax = fig.gca()
plott = np.linspace(0, totalT, 10001)
ax.plot(plott, real_x(plott), linewidth=1)
ax.scatter(n, y, marker=".", c="C01")

# Fit with C
CovData = RecCovData()
CovData.n = CovData.C = CovData.xMean = CovData.yMean = 0
RegData = RecLinRegData()
RegData.CovData = C.pointer(CovData)
# IdData = RecSysIdData()
# IdData.n = 0
# IdData.RegData = C.pointer(RegData)
px = []
py = []
mark = y * 0
for i in range(len(n)-dn):
    if (mark[i] == 0 and mark[i+dn] == 0):
        RecRegIter(y[i], y[i+dn], C.byref(RegData))
        px.append(y[i])
        py.append(y[i+dn])
        mark[i] = mark[i+dn] = 1

cm = RegData.m
cc = RegData.c
cyf = cc/(1-cm)
ctau = -dt/np.log(cm)
# px = y[0::2]
# py = y[1::2]
# pm, pc = np.polyfit(px, py, 1)
# pyf = pc/(1-pm)
print(f"\n Real tau yf c m are \t\t{tau:0.3f}\t\t{xf:0.3f}\t\t{c:0.3f}\t\t{m:0.3f}")
# print(f"Pfit m and c are \t\t{pm:0.3f}\t\t{pyf:0.3f}\t\t{pc:0.3f}")
print(f"SciOpt are \t\t\t{ftau:0.3f}\t\t{fyf:0.3f}\t\t-\t\t-")
print(f"From C are \t\t\t{ctau:0.3f}\t\t{cyf:0.3f}\t\t{cc:0.3f}\t\t{cm:0.3f}")


# Note the cheeky use of the real x0 in some plots
ax.plot(plott, f(plott, x0, cyf, ctau), c='C02')
ax.plot(plott, f(plott, fy0, fyf, ftau), c='C03')
# ax.plot(plotn, (pm**plotn)*(y0 - pyf) + pyf, c='C04')


fig = plt.figure()
ax = fig.gca()
ax.scatter(px, py, marker='.')
ax.scatter(x[:-dn], x[dn:], marker='.')
xx = np.array(ax.get_xlim())
ax.plot(xx, cm*xx + cc)

# fig = plt.figure()
# ax = fig.gca()
# ax.scatter(px, py, marker='.')
# ax.scatter(y_clean_total[:-1], y_clean_total[1:], marker='.')
# xx = np.array(ax.get_xlim())
# ax.plot(xx, pm*xx + pc)

# fig = plt.figure()
# ax = fig.gca()
# ax.scatter(np.arange(0, len(noise),), noise, marker='.')


plt.show(block=False)
input("Press ENTER to quit")
