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
    // std::cout << "Tensor::Tensor(int, int, int, int, int)" << '\n';
    m_aShape = NULL;
    m_aData  = NULL;
    Alloc(new Shape(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize));
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(Shape *pShape) {
    std::cout << "Tensor::Tensor(Shape*)" << '\n';
    m_aShape = NULL;
    m_aData  = NULL;
    Alloc(pShape);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(Tensor *pTensor) {
    std::cout << "Tensor::Tensor(Shape*)" << '\n';
    m_aShape = NULL;
    m_aData  = NULL;
    Alloc(pTensor);
}

template<typename DTYPE> Tensor<DTYPE>::~Tensor() {
    // std::cout << "Tensor::~Tensor()" << '\n';
    Delete();
}

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Shape *pShape) {
    if (pShape == NULL) {
        printf("Receive NULL pointer of Shape class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
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

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Tensor<DTYPE> *pTensor) {
    if (pTensor == NULL) {
        printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape = new Shape(pTensor->GetShape());
        m_aData  = new Data<DTYPE>(pTensor->GetData());
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

//////////////////////////////////////////////////////////////////

template<typename DTYPE> int Tensor<DTYPE>::Reshape(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
    int cur_capacity = m_aData->GetCapacity();
    int new_capacity = pTimeSize * pBatchSize * pChannelSize * pRowSize * pColSize;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot Reshape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        (*m_aShape)[0] = pTimeSize;
        (*m_aShape)[1] = pBatchSize;
        (*m_aShape)[2] = pChannelSize;
        (*m_aShape)[3] = pRowSize;
        (*m_aShape)[4] = pColSize;
    }

    return TRUE;
}

template<typename DTYPE> void Tensor<DTYPE>::Reset() {
    int capacity = m_aData->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*m_aData)[i] = 0;
    }
}

///////////////////////////////////////////////////////////////////

template<typename DTYPE> DTYPE& Tensor<DTYPE>::operator[](unsigned int index) {
    return (*m_aData)[index];
}

//////////////////////////////////////////////////////////////////static method

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Truncated_normal(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, float mean, float stddev) {
    std::cout << "Tensor<DTYPE>::Truncated_normal()" << '\n';

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> rand(mean, stddev);

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

    int capacity = temp->GetData()->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = rand(gen);
    }

    return temp;
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


std::ostream& operator<<(std::ostream& pOS, Tensor<int> *pTensor) {
    int timesize    = pTensor->GetTimeSize();
    int batchsize   = pTensor->GetBatchSize();
    int channelsize = pTensor->GetChannelSize();
    int rowsize     = pTensor->GetRowSize();
    int colsize     = pTensor->GetColSize();

    Shape *shape = pTensor->GetShape();

    pOS << "[ ";

    for (int ti = 0; ti < timesize; ti++) {
        pOS << "[ \n";

        for (int ba = 0; ba < batchsize; ba++) {
            pOS << "[ ";

            for (int ch = 0; ch < channelsize; ch++) {
                pOS << "[ ";

                for (int ro = 0; ro < rowsize; ro++) {
                    pOS << "[ ";

                    for (int co = 0; co < colsize; co++) {
                        pOS << (*pTensor)[Index5D(shape, ti, ba, ch, ro, co)] << ", ";
                    }
                    pOS << " ]\n";
                }
                pOS << " ]";
            }
            pOS << " ]\n";
        }
        pOS << " ]\n";
    }
    pOS << " ]\n";

    return pOS;
}

std::ostream& operator<<(std::ostream& pOS, Tensor<float> *pTensor) {
    int timesize    = pTensor->GetTimeSize();
    int batchsize   = pTensor->GetBatchSize();
    int channelsize = pTensor->GetChannelSize();
    int rowsize     = pTensor->GetRowSize();
    int colsize     = pTensor->GetColSize();

    Shape *shape = pTensor->GetShape();

    pOS << "[ ";

    for (int ti = 0; ti < timesize; ti++) {
        pOS << "[ \n";

        for (int ba = 0; ba < batchsize; ba++) {
            pOS << "[ ";

            for (int ch = 0; ch < channelsize; ch++) {
                pOS << "[ ";

                for (int ro = 0; ro < rowsize; ro++) {
                    pOS << "[ ";

                    for (int co = 0; co < colsize; co++) {
                        pOS << (*pTensor)[Index5D(shape, ti, ba, ch, ro, co)] << ", ";
                    }
                    pOS << " ]\n";
                }
                pOS << " ]";
            }
            pOS << " ]\n";
        }
        pOS << " ]\n";
    }
    pOS << " ]\n";

    return pOS;
}

std::ostream& operator<<(std::ostream& pOS, Tensor<double> *pTensor) {
    int timesize    = pTensor->GetTimeSize();
    int batchsize   = pTensor->GetBatchSize();
    int channelsize = pTensor->GetChannelSize();
    int rowsize     = pTensor->GetRowSize();
    int colsize     = pTensor->GetColSize();

    Shape *shape = pTensor->GetShape();

    pOS << "[ ";

    for (int ti = 0; ti < timesize; ti++) {
        pOS << "[ \n";

        for (int ba = 0; ba < batchsize; ba++) {
            pOS << "[ ";

            for (int ch = 0; ch < channelsize; ch++) {
                pOS << "[ ";

                for (int ro = 0; ro < rowsize; ro++) {
                    pOS << "[ ";

                    for (int co = 0; co < colsize; co++) {
                        pOS << (*pTensor)[Index5D(shape, ti, ba, ch, ro, co)] << ", ";
                    }
                    pOS << " ]\n";
                }
                pOS << " ]";
            }
            pOS << " ]\n";
        }
        pOS << " ]\n";
    }
    pOS << " ]\n";

    return pOS;
}

//// example code
// int main(int argc, char const *argv[]) {
//// Tensor<int> *temp = new Tensor<int>(1, 100, 1, 28, 28);
//
// Tensor<int> *temp = Tensor<int>::Constants(100, 100, 100, 100, 100, 3);
//
// std::cout << Index5D(temp->GetShape(), 0, 1, 0, 1, 1) << '\n';
//
// std::cout << (*temp)[Index5D(temp->GetShape(), 0, 1, 0, 1, 1)] << '\n';
//
// delete temp;
//
// return 0;
// }
