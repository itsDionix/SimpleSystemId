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


# class RecLinRegData(C.Structure):
#     _fields_ = [("m", C.c_double),
#                 ("c", C.c_double),
#                 ("CovData", C.POINTER(RecCovData))]


# class RecSysIdData(C.Structure):
#     _fields_ = [("yf", C.c_double),
#                 ("m", C.c_double),
#                 ("RegData", C.POINTER(RecLinRegData)),
#                 ("n", C.c_int)]


mylib = C.cdll.LoadLibrary("./simplesysid.so")
RecCovIter = mylib.RecursiveCovariance
RecCovIter.argtypes = [C.c_double, C.c_double,
                       C.POINTER(RecCovData)]
GetLine = mylib.GetLineFromCovData
GetLine.argtypes = [C.POINTER(RecCovData),
                    C.POINTER(C.c_double),
                    C.POINTER(C.c_double)]
# RecRegIter = mylib.RecursiveLinReg
# RecRegIter.argtypes = [C.c_double, C.c_double,
#                        C.POINTER(RecLinRegData)]
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
predT = tau*1
totalT = tau*5
N = 2000
Ts = predT/(N-1)
# Interval between samples used for regression
dn = int((N/2)/4)
print(f"dn is {dn:d}")
dt = dn*Ts
m = np.exp(-dt/tau)
c = xf*(1-m)
n = np.linspace(0, predT, N)
noise_p = 0.5
noise = rng.normal(size=N)*(x0 - real_x(predT/tau))*noise_p
x = real_x(n)
y = x + noise

# fit with scipy


def f(t, x0, xf, tau):
    return np.exp(-t/tau)*(x0 - xf) + xf


fit = sio.curve_fit(f, n, y, (1, 1, 1))[0]
fy0, fyf, ftau = fit

fig = plt.figure()
ax = fig.gca()
plott = np.linspace(0, totalT, 10001)
ax.plot(plott, real_x(plott), linewidth=1, label="Original exponential")
ax.scatter(n, y, marker=".", c="C01", s=1)

# Fit with C no reps
CovData = RecCovData()
CovData.n = CovData.C = CovData.Cx = CovData.Cy = CovData.xMean = CovData.yMean = 0
# RegData = RecLinRegData()
# RegData.CovData = C.pointer(CovData)
# IdData = RecSysIdData()
# IdData.n = 0
# IdData.RegData = C.pointer(RegData)
px = []
py = []
mark = y * 0
for i in range(len(n)-dn):
    if (mark[i] == 0 and mark[i+dn] == 0):
        RecCovIter(y[i], y[i+dn], C.byref(CovData))
        px.append(y[i])
        py.append(y[i+dn])
        mark[i] = mark[i+dn] = 1
px = np.array(px)
py = np.array(py)

cm = C.c_double()
cc = C.c_double()
GetLine(C.byref(CovData), C.byref(cm), C.byref(cc))
cm = cm.value
cc = cc.value
cyf = cc/(1-cm)
ctau = -dt/np.log(cm)

# Fit with C with reps
CovData.n = CovData.C = CovData.Cx = CovData.Cy = CovData.xMean = CovData.yMean = 0
for i in range(len(n)-dn):
    RecCovIter(y[i], y[i+dn], C.byref(CovData))
crm = C.c_double()
crc = C.c_double()
GetLine(C.byref(CovData), C.byref(crm), C.byref(crc))
crm = crm.value
crc = crc.value
cryf = crc/(1-crm)
crtau = -dt/np.log(crm)

cov = np.cov(px, py)
l, v = np.linalg.eig(cov)
idx = l.argsort()[::-1]
l = l[idx]
v = v[:, idx]
pm = v[1, 0] / v[0, 0]
pc = np.mean(py) - pm*np.mean(px)
pyf = pc/(1-pm)
ptau = -dt/np.log(pm)

print(f"\n Real tau yf c m are \t\t{tau:0.3f}\t\t{xf:0.3f}\t\t{c:0.3f}\t\t{m:0.3f}")
print(f"SciOpt are \t\t\t{ftau:0.3f}\t\t{fyf:0.3f}\t\t-\t\t-")
print(f"P are \t\t\t\t{ptau:0.3f}\t\t{pyf:0.3f}\t\t{pc:0.3f}\t\t{pm:0.3f}")
print(f"From C no reps are \t\t{ctau:0.3f}\t\t{cyf:0.3f}\t\t{cc:0.3f}\t\t{cm:0.3f}")
print(f"From C with reps are \t\t{crtau:0.3f}\t\t{cryf:0.3f}\t\t{crc:0.3f}\t\t{crm:0.3f}")


# Note the cheeky use of the real x0 in some plots
xx = np.array(ax.get_xlim())
# ax.plot(xx, xx*0+cyf, c='C02', linestyle="-.", label="Cov method in C (no reps)")
ax.plot(xx, xx*0+cryf, c='C05', linestyle="-.", label="Cov method in C (with reps)")
ax.plot(plott, f(plott, fy0, fyf, ftau), c='C03', linestyle="--", label="SciPy method")
# ax.plot(xx, xx*0+pyf, c='C04', linestyle="--", label="Cov method in Python")
ax.legend()

fig = plt.figure()
ax = fig.gca()
ax.scatter(px, py, marker='.', s=1)
ax.scatter(x[:-dn], x[dn:], marker='.')
xx = np.array(ax.get_xlim())
ax.plot(xx, cm*xx + cc)
ax.plot(xx, pm*xx + pc)

# fig = plt.figure()
# ax = fig.gca()
# ax.scatter(px, py, marker='.')
# ax.scatter(y_clean_total[:-1], y_clean_total[1:], marker='.')
# xx = np.array(ax.get_xlim())
# ax.plot(xx, pm*xx + pc)

# fig = plt.figure()
# ax = fig.gca()
# ax.scatter(np.arange(0, len(noise),), noise, marker='.')

plt.pause(0.5)

print(" ** Statistical characterization ** ")
tests = 100
fyfs = np.empty(shape=((tests,)))
ftaus = np.empty_like(fyfs)
cyfs = np.empty_like(fyfs)
ctaus = np.empty_like(fyfs)
cryfs = np.empty_like(fyfs)
crtaus = np.empty_like(fyfs)

for test in range(tests):
    noise = rng.normal(size=N)*(x0 - real_x(predT/tau))*noise_p
    y = x + noise

    # Fit scipy
    _, fyfs[test], ftaus[test] = sio.curve_fit(f, n, y, (1, 1, 1))[0]

    # Fit C without repeated points
    CovData.n = CovData.C = CovData.Cx = CovData.Cy = CovData.xMean = CovData.yMean = 0
    mark = y * 0
    for i in range(len(n)-dn):
        if (mark[i] == 0 and mark[i+dn] == 0):
            RecCovIter(y[i], y[i+dn], C.byref(CovData))
            mark[i] = mark[i+dn] = 1
    cm = C.c_double()
    cc = C.c_double()
    GetLine(C.byref(CovData), C.byref(cm), C.byref(cc))
    cm = cm.value
    cc = cc.value
    cyfs[test] = cc/(1-cm)
    ctaus[test] = -dt/np.log(cm)

    # Fit C with repeated points
    CovData.n = CovData.C = CovData.Cx = CovData.Cy = CovData.xMean = CovData.yMean = 0
    for i in range(len(n)-dn):
        RecCovIter(y[i], y[i+dn], C.byref(CovData))
    cm = C.c_double()
    cc = C.c_double()
    GetLine(C.byref(CovData), C.byref(cm), C.byref(cc))
    cm = cm.value
    cc = cc.value
    cryfs[test] = cc/(1-cm)
    crtaus[test] = -dt/np.log(cm)


efyfs = fyfs - xf
eftaus = ftaus - tau
ecyfs = cyfs - xf
ectaus = ctaus - tau
ecryfs = cryfs - xf
ecrtaus = crtaus - tau


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


print(f"\n Results \t\t\tav eyf\t\trms eyf\t\tav etau\t\trms etau")
print(f"SciOpt are \t\t\t{np.mean(efyfs):0.3e}\t{rms(efyfs):0.3e}\t{np.mean(eftaus):0.3e}\t{rms(eftaus):0.3e}")
print(f"C no rep are \t\t\t{np.mean(ecyfs):0.3e}\t{rms(ecyfs):0.3e}\t{np.mean(ectaus):0.3e}\t{rms(ectaus):0.3e}")
print(f"C with rep are \t\t\t{np.mean(ecryfs):0.3e}\t{rms(ecryfs):0.3e}\t{np.mean(ecrtaus):0.3e}\t{rms(ecrtaus):0.3e}")


input("Press ENTER to quit")
