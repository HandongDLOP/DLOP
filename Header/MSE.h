#ifndef MSE_H_
#define MSE_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class MSE : public Operator {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (MSE::Alloc())
    MSE(Operator *pInput1, Operator *pInput2) : Operator(pInput1, pInput2) {
        std::cout << "MSE::MSE(Operator *, MetaParameter *)" << '\n';
        Alloc(pInput1, pInput2);
    }

    MSE(Operator *pInput1, Operator *pInput2, std::string pName) : Operator(pInput1, pInput2, pName) {
        std::cout << "MSE::MSE(Operator *, MetaParameter *, std::string)" << '\n';
        Alloc(pInput1, pInput2);
    }

    virtual ~MSE() {
        std::cout << "MSE::~MSE()" << '\n';
    }

    virtual bool Alloc(Operator *pInput1, Operator *pInput2) {
        std::cout << "MSE::Alloc(Operator *, Operator *)" << '\n';
        // if pInput1 and pInput2의 shape가 다르면 abort

        TensorShape *InputDim0 = pInput1->GetOutput()->Getshape();
        TensorShape *InputDim1 = pInput2->GetOutput()->Getshape();

        if (InputDim0->Getdim()[0] != InputDim1->Getdim()[0]) {
            std::cout << "data has invalid dimension" << '\n';
            exit(0);
        }

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        GetInputOperator()[0]->GetOutput()->PrintData();

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        TensorShape *InputDim0 = GetInputOperator()[0]->GetOutput()->Getshape(); // 하나만 확인해도 된다.

        int output = InputDim0->Getdim()[0] * InputDim0->Getdim()[1]  /* * InputDim0->Getdim()[2] == ch*/;
        // int batch  = InputDim0->Getdim()[3];

        float *data0  = GetInputOperator()[0]->GetOutput()->GetData();
        float *data1  = GetInputOperator()[1]->GetOutput()->GetData();
        float *result = GetInputOperator()[0]->GetDelta()->GetData();

        for (int i = 0; i < output; i++) {
            result[i] = (data0[i] - data1[i]) / output;
        }

        GetInputOperator()[0]->GetDelta()->PrintData();

        return true;
    }
};

#endif  // MSE_H_
