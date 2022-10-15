# SimpleSystemId
Implementation of online system identification given by y[n] = ay[n-1] + b for unkown a and b. Implementted in C, includes a python tests script.

One example use case is estimating the final value of a signal of type exp(-t/tau)(v0 - vf) + vf from noisy samples. The next image presents example results. 

The blue line is the original signal, the orange dots are the noisy samples, the red dashed exponential is the prediction from a SciPy fitting method to the samples and the horizontal line is the predicted vf from this code.

![Example system identification](readme_fig.jpeg?raw=true "Example system identification")

The method consists on creating data of the form y = mx + c by considering y = v((n+d)T) and x = v(nT), where T is the sampling period. This data would form a straight line without any measurement noise. Since there is measurement noise, the line is recovered as the principal component of the data samples. Tau and xf are then recovered from d, m and c.