#include "Tensor.h"

namespace temp {
template<typename DTYPE>
int Argmax(DTYPE *data, int Dimension) {
    int   index = 0;
    DTYPE max   = data[0];

    for (int dim = 1; dim < Dimension; dim++) {
        if (data[dim] > max) {
            max   = data[dim];
            index = dim;
        }
    }

    return index;
}

template<typename DTYPE>
float Accuracy(Tensor<DTYPE> *pred, Tensor<DTYPE> *ans, int Batch) {
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;

    TENSOR_DTYPE pred_data = pred->GetData();
    TENSOR_DTYPE ans_data  = ans->GetData();

    float accuracy = 0.0;

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
