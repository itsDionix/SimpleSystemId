#ifndef SIMPLESYSID_H_INCLUDED
#define SIMPLESYSID_H_INCLUDED

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct RecursiveCovarianceData {
    /* Data for covariance matrix of two variables x and y
     */
    double C;     // sum (xi - x_mean)(yi - y_mean) so far
    double Cx;    // sum (xi - x_mean)^2 so far
    double Cy;    // sum (yi - y_mean)^2 so far
    double xMean; // x mean so far
    double yMean; // y mean so far
    int n;        // points so far
};

struct RecursiveLinRegData {
    /* Data for y = mx + c regression - unkown m and c
     */
    double m;
    double c;
    struct RecursiveCovarianceData *CovData;
};

struct RecursiveSysIdData {
    /* Data for 'y[n] = m^n * (x0 - xf) + xf + noise' fitting for unkown m < 1, xf and x0
        Assumes the first data point given is y[0]
     */
    double m;
    double c;
    struct RecursiveCovarianceData *CovData;
};

int RecursiveCovariance(double new_x, double new_y, struct RecursiveCovarianceData *DataPtr);
/* One iteration of calculating a covariance matrix of two variables recursevily
 */

int RecursiveLinReg(double new_x, double new_y, struct RecursiveLinRegData *DataPtr);
/* One iteration of calculating a simple linear regression
 */

#endif