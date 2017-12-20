#ifndef RESHAPE_H_
#define RESHAPE_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Reshape : public Operator<DTYPE>{
private:
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;

public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Reshape::Alloc())
    Reshape(Operator<DTYPE> *pInput, int pChannel, int pRow, int pCol, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Reshape::Reshape(Operator *)" << '\n';
        this->Alloc(pInput, pChannel, pRow, pCol);
    }

    ~Reshape() {
        std::cout << "Reshape::~Reshape()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput, int pChannel, int pRow, int pCol) {
        std::cout << "Reshape::Alloc(Operator *, Operator *)" << '\n';

        int *shape = this->GetInputOperator()[0]->GetOutput()->GetShape();

        if (shape[2] * shape[3] * shape[4] != pChannel * pRow * pCol) {
            std::cout << "invalid shape" << '\n';
            exit(0);
        }

        Tensor<DTYPE> *output = new Tensor<DTYPE>(shape[0], shape[1], pChannel, pRow, pCol);
        this->SetOutput(output);
        Tensor<DTYPE> *delta = new Tensor<DTYPE>(shape[0], shape[1], pChannel, pRow, pCol);
        this->SetDelta(delta);

        return 1;
    }

    virtual int ComputeForwardPropagate() {
        std::cout << "Reshape_Forward" << '\n';
        int *shape          = this->GetInputOperator()[0]->GetOutput()->GetShape();
        TENSOR_DTYPE input  = this->GetInputOperator()[0]->GetOutput()->GetData();
        int *result_shape   = this->GetOutput()->GetShape();
        TENSOR_DTYPE output = this->GetOutput()->GetData();

        DTYPE temp[shape[2] * shape[3] * shape[4]] = { 0.0 };

        int count = 0;

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                count = 0;

                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            temp[count] = input[ti][ba][ch][ro][co];
                            count++;
                        }
                    }
                }

                count = 0;

                for (int ch = 0; ch < result_shape[2]; ch++) {
                    for (int ro = 0; ro < result_shape[3]; ro++) {
                        for (int co = 0; co < result_shape[4]; co++) {
                            output[ti][ba][ch][ro][co] = temp[count];
                            count++;
                        }
                    }
                }
            }
        }


        return 1;
    }

    virtual int ComputeBackPropagate() {
        int *shape          = this->GetOutput()->GetShape();
        TENSOR_DTYPE output = this->GetOutput()->GetData();
        int *input_shape    = this->GetInputOperator()[0]->GetOutput()->GetShape();
        TENSOR_DTYPE delta  = this->GetDelta()->GetData();

        this->GetInputOperator()[0]->GetDelta()->Reset();
        TENSOR_DTYPE delta_input = this->GetInputOperator()[0]->GetDelta()->GetData();

        DTYPE temp[shape[2] * shape[3] * shape[4]] = { 0.0 };

        int count = 0;

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                count = 0;

                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            temp[count] = delta[ti][ba][ch][ro][co];
                            count++;
                        }
                    }
                }

                count = 0;

                for (int ch = 0; ch < input_shape[2]; ch++) {
                    for (int ro = 0; ro < input_shape[3]; ro++) {
                        for (int co = 0; co < input_shape[4]; co++) {
                            delta_input[ti][ba][ch][ro][co] = temp[count];
                            count++;
                        }
                    }
                }
            }
        }

        return 1;
    }
};

#endif  // RESHAPE_H_
