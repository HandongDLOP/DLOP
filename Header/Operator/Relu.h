#ifndef RELU_H_
#define RELU_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Relu : public Operator<DTYPE>{
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Relu::Alloc())
    Relu(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Relu::Relu(Operator<DTYPE> *)" << '\n';
        this->Alloc(pInput);
    }

    ~Relu() {
        std::cout << "Relu::~Relu()" << '\n';
    }

    bool Alloc(Operator<DTYPE> *pInput) {
        std::cout << "Relu::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        Tensor<DTYPE> *output = new Tensor<DTYPE>(pInput->GetOutput()->GetShape());
        this->SetOutput(output);
        Tensor<DTYPE> *delta = new Tensor<DTYPE>(pInput->GetOutput()->GetShape());
        this->SetDelta(delta);

        return true;
    }

    bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int *shape        = this->GetInputOperator()[0]->GetOutput()->GetShape();
        DTYPE *****input  = this->GetInputOperator()[0]->GetOutput()->GetData();
        DTYPE *****output = this->GetOutput()->GetData();

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            output[ti][ba][ch][ro][co] = this->Max(input[ti][ba][ch][ro][co], 0.0);
                        }
                    }
                }
            }
        }

        return true;
    }

    bool ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int *shape        = this->GetOutput()->GetShape();
        DTYPE *****output = this->GetOutput()->GetData();
        DTYPE *****delta  = this->GetDelta()->GetData();

        this->GetInputOperator()[0]->GetDelta()->Reset();
        DTYPE *****delta_input = this->GetInputOperator()[0]->GetDelta()->GetData();

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            if (output[ti][ba][ch][ro][co] > 0.0) {
                                delta_input[ti][ba][ch][ro][co] = delta[ti][ba][ch][ro][co];
                            } else {
                                delta_input[ti][ba][ch][ro][co] = 0;
                            }
                        }
                    }
                }
            }
        }

        // GetInputOperator()[0]->GetDelta()->PrintData();

        // GetDelta()->Reset();

        return true;
    }

    // for relu
    DTYPE Max(DTYPE data1, DTYPE data2) {
        if (data1 >= data2) return data1;
        else return data2;
    }
};

#endif  // RELU_H_
