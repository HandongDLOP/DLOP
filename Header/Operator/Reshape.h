#ifndef RESHAPE_H_
#define RESHAPE_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Reshape : public Operator<DTYPE>{
private:
    Shape *m_aRe;

public:
    Reshape(Operator<DTYPE> *pInput, int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Reshape::Reshape(Operator *)" << '\n';
        this->Alloc(pInput, pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);
    }

    ~Reshape() {
        std::cout << "Reshape::~Reshape()" << '\n';
        delete m_aRe;
    }

    int Alloc(Operator<DTYPE> *pInput, int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
        std::cout << "Reshape::Alloc(Operator *, Operator *)" << '\n';

        m_aRe = new Shape(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

        Tensor<DTYPE> *result = new Tensor<DTYPE>(pInput->GetResult());
        result->Reshape((*m_aRe)[0], (*m_aRe)[1], (*m_aRe)[2], (*m_aRe)[3], (*m_aRe)[4]);

        this->SetResult(result);  // copy data

        Shape *shapeOfDelta = new Shape(m_aRe);
        this->SetDelta(new Tensor<DTYPE>(shapeOfDelta));

        return TRUE;
    }

    int ComputeForwardPropagate() {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int capacity = result->GetData()->GetCapacity();

        for(int i = 0; i < capacity; i++){
            (*result)[i] = (*input)[i];
        }

        return TRUE;
    }

    int ComputeBackPropagate() {
        int capacity = this->GetDelta()->GetData()->GetCapacity();

        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        for (int i = 0; i < capacity; i++) {
            (*input_delta)[i] = (*this_delta)[i];
        }

        return TRUE;
    }
};

#endif  // RESHAPE_H_
