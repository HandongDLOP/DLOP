#include "Tensor.h"

namespace temp {
int Argmax(double *data, int Dimension) {
    int index  = 0;
    double max = data[0];

    for (int dim = 1; dim < Dimension; dim++) {
        if (data[dim] > max) {
            max   = data[dim];
            index = dim;
        }
    }

    return index;
}

double Accuracy(Tensor *pred, Tensor *ans, int Batch) {
    double *****pred_data = pred->GetData();
    double *****ans_data  = ans->GetData();

    double accuracy = 0.0;

    int pred_index = 0;
    int ans_index  = 0;

    for (int ba = 0; ba < Batch; ba++) {
        pred_index = Argmax(pred_data[0][ba][0][0], 10);
        ans_index  = Argmax(ans_data[0][ba][0][0], 10);

        if (pred_index == ans_index) {
            accuracy += 1.0 / Batch;
        } else {
            // std::cout << pred_index << '\n';
        }
    }

    return accuracy;
}
}
