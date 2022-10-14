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
    if (DataPtr->CovData->n > 1) {
	DataPtr->m = DataPtr->CovData->C / DataPtr->CovData->Cx;
	DataPtr->c = DataPtr->CovData->yMean - DataPtr->m * DataPtr->CovData->xMean;
    } else {
	DataPtr->c = new_y;
    }
    return 0;
    ;
}

int RecursiveSysId(double new_x, struct RecursiveLinRegData *DataPtr) {
    RecursiveCovariance(new_x, new_x, DataPtr->CovData);
    DataPtr->m = DataPtr->CovData->C / DataPtr->CovData->Cx;
    DataPtr->c = DataPtr->CovData->yMean - DataPtr->m * DataPtr->CovData->xMean;

    return DataPtr->CovData->n;
    ;
}
