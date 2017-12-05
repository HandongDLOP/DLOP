#ifndef VARIABLE_H_
#define VARIABLE_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Variable : public Operator<DTYPE>{
public:
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;
public:
    Variable(std::string pName) : Operator<DTYPE>(pName) {
        std::cout << "Variable::Variable(std::string)" << '\n';
    }

    Variable(Tensor<DTYPE> *pTensor, std::string pName, int pTrainable = 1) : Operator<DTYPE>(pTensor, pName) {
        std::cout << "Variable::Variable(Tensor<DTYPE> *, std::string)" << '\n';

        this->Alloc(pTensor, pTrainable);
    }

    ~Variable() {
        std::cout << "Variable::~Variable()" << '\n';
    }

    virtual bool Alloc(Tensor<DTYPE> *pTensor, int pTrainable) {
        if (pTensor->GetShape()[0] != 1) {
            std::cout << "data has unvalid time dimension" << '\n';
            exit(0);
        }

        this->SetOutput(pTensor);

        Tensor<DTYPE> *gradient = new Tensor<DTYPE>(pTensor->GetShape());

        this->SetGradient(gradient);

        Tensor<DTYPE> *delta = new Tensor<DTYPE>(pTensor->GetShape());

        this->SetDelta(delta);

        this->SetTrainable(pTrainable);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        return true;
    }

    virtual bool ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int *shape       = this->GetOutput()->GetShape();
        TENSOR_DTYPE delta = this->GetDelta()->GetData();
        TENSOR_DTYPE grad  = this->GetGradient()->GetData();

        // 이전에 구해져 있던 gradient와 합치기
        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            grad[ti][ba][ch][ro][co] += delta[ti][ba][ch][ro][co];
                        }
                    }
                }
            }
        }

        // Training
        // GetOutput()->PrintData();
        // GetGradient()->PrintData();
        // GetOptimizer()->UpdateWeight(GetOutput(), GetGradient());
        // GetOutput()->PrintData();
        // GetGradient()->PrintData();

        // GetDelta()->Reset();
        return true;
    }
};

#endif  // VARIABLE_H_
