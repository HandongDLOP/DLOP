#include "Tensor.h"

namespace temp {
template<typename DTYPE> int Argmax(Tensor<DTYPE> *data, int ba, int Dimension) {
    int   index = 0;
    DTYPE max   = (*data)[ba * 10];
    int   start = ba * 10;
    int   end   = ba * 10 + 10;

    for (int dim = start + 1; dim < end; dim++) {
        if ((*data)[dim] > max) {
            max   = (*data)[dim];
            index = dim - start;
        }
    }

    // std::cout << index << ' ';

    return index;
}

template<typename DTYPE> float Accuracy(Tensor<DTYPE> *pred, Tensor<DTYPE> *ans, int Batch) {
    // typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;

    float accuracy = 0.0;

    int pred_index = 0;
    int ans_index  = 0;

    for (int ba = 0; ba < Batch; ba++) {
        pred_index = Argmax(pred, ba, 10);
        ans_index  = Argmax(ans, ba, 10);

        if (pred_index == ans_index) {
            accuracy += 1 / (float)Batch;
        } else {
            std::cout << pred_index << '\n';
        }
    }

    return accuracy;
}
}
