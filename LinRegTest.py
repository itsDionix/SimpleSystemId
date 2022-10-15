import ctypes as C
import numpy as np
import matplotlib.pyplot as plt


class RecursiveCovarianceData(C.Structure):
    _fields_ = [("C", C.c_double),
                ("Cx", C.c_double),
                ("Cy", C.c_double),
                ("xMean", C.c_double),
                ("yMean", C.c_double),
                ("n", C.c_int)]


class RecLinRegData(C.Structure):
    _fields_ = [("m", C.c_double),
                ("c", C.c_double),
                ("CovData", C.POINTER(RecursiveCovarianceData))]


mylib = C.cdll.LoadLibrary("./simplesysid.so")
RecCovIter = mylib.RecursiveCovariance
RecCovIter.argtypes = [C.c_double, C.c_double,
                       C.POINTER(RecursiveCovarianceData)]
RecRegIter = mylib.RecursiveLinReg
RecRegIter.argtypes = [C.c_double, C.c_double,
                       C.POINTER(RecLinRegData)]

N = 200000
rng = np.random.default_rng()
m = -54
c = 33
x = rng.uniform(size=N)
noise = rng.normal(size=N)
y = x*m + c + noise
fig = plt.figure()
ax = fig.gca()
ax.scatter(x, y, marker=".")

# Bias to use population cov (devidie by N, not N-1)
cov = np.cov(x, y, bias=True)

pfit = np.polyfit(x, y, 1)

print(f"The varx, vary and cov are:  "
      f"{cov[0][0]:0.3f}, {cov[1][1]:0.3f}, {cov[0][1]:0.3f}")

CovData = RecursiveCovarianceData()
CovData.n = CovData.C = CovData.xMean = CovData.yMean = 0
RegData = RecLinRegData()
RegData.CovData = C.pointer(CovData)

for xi, yi in zip(x, y):
    RecRegIter(xi, yi, C.byref(RegData))

cCov = CovData.C / CovData.n
cVarx = CovData.Cx / CovData.n
cVary = CovData.Cy / CovData.n
print(f"From C, the varx, vary and cov are:  "
      f"{cVarx:0.3f}, {cVary:0.3f}, {cCov:0.3f}")
cm = RegData.m
cc = RegData.c
print(f"\n Real m and c are \t\t{m:0.3f}\t\t{c:0.3f}")
print(f"Pfit m and c are \t\t{pfit[0]:0.3f}\t\t{pfit[1]:0.3f}")
print(f"From C the m and c are \t\t{cm:0.3f}\t\t{cc:0.3f}")

xx = np.array(ax.get_xlim())
ax.plot(xx, cm*xx + cc)


plt.show(block=False)
input("Press ENTER to quit")
