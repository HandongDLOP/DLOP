#include "Tensor.h"

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

////////////////////////////////////////////////////////////////////Class Tensor
template<typename DTYPE> Tensor<DTYPE>::Tensor() {
    std::cout << "Tensor::Tensor()" << '\n';
    m_aShape = NULL;
    m_aData  = NULL;
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
    std::cout << "Tensor::Tensor(Shape*)" << '\n';
    m_aShape = NULL;
    m_aData  = NULL;
    Alloc(new Shape(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize));
}

template<typename DTYPE> Tensor<DTYPE>::~Tensor() {
    Delete();
}

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Shape *pShape) {
    if (pShape == NULL) {
        printf("Receive invalid Shape value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape = pShape;

        int rank = pShape->GetRank();

        if (rank < 5) {
            delete m_aShape;
            m_aShape = NULL;
            printf("Receive invalid rank value %d in %s (%s %d)\n", rank, __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        } else {
            int capacity = 1;

            for (int i = 0; i < rank; i++) {
                capacity *= (*pShape)[i];
            }
            m_aData = new Data<DTYPE>(capacity);
        }
    }

    return TRUE;
}

template<typename DTYPE> void Tensor<DTYPE>::Delete() {
    if (m_aShape) {
        delete m_aShape;
        m_aShape = NULL;
    }

    if (m_aData) {
        delete m_aData;
        m_aData = NULL;
    }
}

template<typename DTYPE> Shape *Tensor<DTYPE>::GetShape() {
    return m_aShape;
}

template<typename DTYPE> Data<DTYPE> *Tensor<DTYPE>::GetData() {
    return m_aData;
}

template<typename DTYPE> int Tensor<DTYPE>::GetTimeSize() {
    return (*m_aShape)[0];
}

template<typename DTYPE> int Tensor<DTYPE>::GetBatchSize() {
    return (*m_aShape)[1];
}

template<typename DTYPE> int Tensor<DTYPE>::GetChannelSize() {
    return (*m_aShape)[2];
}

template<typename DTYPE> int Tensor<DTYPE>::GetRowSize() {
    return (*m_aShape)[3];
}

template<typename DTYPE> int Tensor<DTYPE>::GetColSize() {
    return (*m_aShape)[4];
}

///////////////////////////////////////////////////////////////////

template<typename DTYPE> DTYPE& Tensor<DTYPE>::operator[](unsigned int index) {
    return (*m_aData)[index];
}

//////////////////////////////////////////////////////////////////static method

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Truncated_normal(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, float mean, float stddev) {
    std::cout << "Tensor<DTYPE>::Truncated_normal()" << '\n';

    return NULL;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Zeros(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';

    return new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Constants(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, DTYPE constant) {
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

    int capacity = temp->GetData()->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = constant;
    }

    return temp;
}

///////////////////////////////////////////////////////////////////

unsigned int Index5D(Shape *pShape, int ti, int ba, int ch, int ro, int co) {
    return (((ti * (*pShape)[1] + ba) * (*pShape)[2] + ch) * (*pShape)[3] + ro) * (*pShape)[4] + co;
}

//// example code
// int main(int argc, char const *argv[]) {
//// Tensor<int> *temp = new Tensor<int>(1, 100, 1, 28, 28);
//
// Tensor<int> *temp = Tensor<int>::Constants(1, 100, 1, 28, 28, 3);
//
// std::cout << Index5D(temp->GetShape(), 0, 1, 0, 1, 1) << '\n';
//
// std::cout << (*temp)[Index5D(temp->GetShape(), 0, 1, 0, 1, 1)] << '\n';
//
// delete temp;
//
// return 0;
// }
