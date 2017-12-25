#include "Tensor.h"

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

////////////////////////////////////////////////////////////////////Class Tensor

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Shape *pShape) {
    if (pShape == NULL) {
        printf("Receive invalid Shape value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape = pShape;

        int rank = pShape->GetRank();
        int size = 1;

        for (int i = 0; i < rank; i++) {
            size *= (*pShape)[i];
        }
        m_aData = new Data<DTYPE>(size);
    }

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::Delete() {
    if (m_aShape) {
        delete m_aShape;
        m_aShape = NULL;
    }

    if (m_aData) {
        delete m_aData;
        m_aData = NULL;
    }
    return TRUE;
}

template<typename DTYPE> unsigned int Tensor<DTYPE>::Index(int rank, ...) {
    va_list ap;
    va_start(ap, rank);



    return 0;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Truncated_normal(Shape *pShape, float mean, float stddev) {
    std::cout << "Tensor<DTYPE>::Truncated_normal()" << '\n';

    return NULL;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Zeros(Shape *pShape) {
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';

    return NULL;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Constants(Shape *pShape, DTYPE constant) {
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';

    return NULL;
}
