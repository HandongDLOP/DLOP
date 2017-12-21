#include "Tensor.h"

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

template<typename DTYPE>
int Tensor<DTYPE>::Alloc() {
    return 1;
}

template<typename DTYPE>
int Tensor<DTYPE>::Alloc(int pTime, int pBatch, int pChannel, int pRow, int pCol) {
    return 1;
}

template<typename DTYPE>
int Tensor<DTYPE>::Delete() {
    return 1;
}

template<typename DTYPE>
void Tensor<DTYPE>::Reset() {}

// ===========================================================================================

template<typename DTYPE>
Tensor<DTYPE> *Tensor<DTYPE>::Truncated_normal(int pTime, int pBatch, int pChannel, int pRow, int pCol, float mean, float stddev) {
    std::cout << "Tensor<DTYPE>::Truncated_normal()" << '\n';

    Tensor<DTYPE> *temp_Tensor = new Tensor(pTime, pBatch, pChannel, pRow, pCol);


    return temp_Tensor;
}

template<typename DTYPE>
Tensor<DTYPE> *Tensor<DTYPE>::Zeros(int pTime, int pBatch, int pChannel, int pRow, int pCol) {
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';

    Tensor<DTYPE> *temp_Tensor = new Tensor(pTime, pBatch, pChannel, pRow, pCol);

    return temp_Tensor;
}

template<typename DTYPE>
Tensor<DTYPE> *Tensor<DTYPE>::Constants(int pTime, int pBatch, int pChannel, int pRow, int pCol, DTYPE constant) {
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';

    Tensor<DTYPE> *temp_Tensor = new Tensor(pTime, pBatch, pChannel, pRow, pCol);
    return temp_Tensor;
}

template<typename DTYPE>
void Tensor<DTYPE>::PrintData(int forceprint) {}

template<typename DTYPE>
void Tensor<DTYPE>::PrintShape() {}
