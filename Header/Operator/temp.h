#ifndef TEMP_H
#define TEMP_H    value

#include "..//Operator.h"

template<typename DTYPE>
class Temp : public Operator<DTYPE>{
public:
    Temp(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Temp::Temp(Operator<DTYPE> *)" << '\n';
        this->Alloc(pInput);
    }

    ~Temp() {
        std::cout << "Temp::~Temp()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput) {
        std::cout << "Temp::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        Shape *shapeOfResult = new Shape(pInput->GetResult()->GetShape());
        this->SetResult(new Tensor<DTYPE>(shapeOfResult));

        Shape *shapeOfDelta = new Shape(pInput->GetResult()->GetShape());
        this->SetDelta(new Tensor<DTYPE>(shapeOfDelta));

        return TRUE;
    }

    int ComputeForwardPropagate() {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        int capacity          = input->GetData()->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*result)[i] = this->MAX((*input)[i], 0.f);
        }

        return TRUE;
    }

    int ComputeBackPropagate() {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        int capacity               = result->GetData()->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            if ((*result)[i] > 0.0) (*input_delta)[i] = (*this_delta)[i];
            else (*input_delta)[i] = 0;
        }

        return TRUE;
    }

    inline DTYPE MAX(DTYPE data1, DTYPE data2) {
        if (data1 >= data2) return data1;
        else return data2;
    }
};

#endif  // TEMP_H
