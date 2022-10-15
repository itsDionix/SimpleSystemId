#include "simplesysid.h"

int RecursiveCovariance(double new_x, double new_y, struct RecursiveCovarianceData *DataPtr) {
    double dx, dy;

    DataPtr->n = DataPtr->n + 1;
    dx = new_x - DataPtr->xMean;
    dy = new_y - DataPtr->yMean;
    DataPtr->xMean = DataPtr->xMean + dx / DataPtr->n;
    DataPtr->yMean = DataPtr->yMean + (new_y - DataPtr->yMean) / DataPtr->n;
    DataPtr->C = DataPtr->C + dx * (new_y - DataPtr->yMean);
    DataPtr->Cx = DataPtr->Cx + dx * (new_x - DataPtr->xMean);
    DataPtr->Cy = DataPtr->Cy + dy * (new_y - DataPtr->yMean);

    return 0;
}

int RecursiveLinReg(double new_x, double new_y, struct RecursiveLinRegData *DataPtr) {
    RecursiveCovariance(new_x, new_y, DataPtr->CovData);
    if (DataPtr->CovData->n > 0) {
        DataPtr->m = DataPtr->CovData->C / DataPtr->CovData->Cx;
        DataPtr->c = DataPtr->CovData->yMean - DataPtr->m * DataPtr->CovData->xMean;
    } else {
        DataPtr->c = new_y;
    }
    return 0;
}

int GetLineFromCovData(struct RecursiveCovarianceData *DataPtr, double *m, double *c) {
    double lambda;

    // Get biggest eigvalue from Cov matrix
    lambda = (DataPtr->Cx + DataPtr->Cy) / 2 +
             sqrt(pow((DataPtr->Cx - DataPtr->Cy) / 2, 2) + pow(DataPtr->C, 2));
    *m = (lambda - DataPtr->Cx) / DataPtr->C;
    *c = DataPtr->yMean - DataPtr->xMean * (*m);

    return 0;
}
