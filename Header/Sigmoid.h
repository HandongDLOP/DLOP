#ifndef SIGMOID_H_
#define SIGMOID_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Sigmoid : public Operator {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Sigmoid::Alloc())
    Sigmoid(Operator *pInput, std::string pName) : Operator(pInput, pName) {
        std::cout << "/* Sigmoid::Sigmoid(Operator *) */" << '\n';
        Alloc(pInput);
    }

    virtual ~Sigmoid() {
        std::cout << "Sigmoid::~Sigmoid()" << '\n';
    }

    virtual bool Alloc(Operator *pInput) {
        std::cout << "Sigmoid::Alloc(Operator *, Operator *)" << '\n';

        Tensor *temp_output = new Tensor(GetInputOperator()[0]->GetOutput()->Getshape());

        SetOutput(temp_output);

        Tensor *temp_delta = new Tensor(GetInputOperator()[0]->GetOutput()->Getshape());

        SetDelta(temp_delta);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int size = GetInputOperator()[0]->GetOutput()->GetFlatDim();

        float *data = GetInputOperator()[0]->GetOutput()->GetData();

        float *result = GetOutput()->GetData();

        for (int i = 0; i < size; i++) {
            result[i] = sigmoid(data[i]);
        }

        // SetOutput(result);

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int size = GetOutput()->GetFlatDim();

        float *output = GetOutput()->GetData();

        float *delta = GetDelta()->GetData();

        float *delta_for_next = GetInputOperator()[0]->GetDelta()->GetData();

        for (int i = 0; i < size; i++) {
                delta_for_next[i] = delta[i] * output[i] * (1 - output[i]);
        }

        // GetInputOperator()[0]->SetDelta(delta_for_next);

        GetInputOperator()[0]->GetDelta()->PrintData();

        return true;
    }

    // for Sigmoid
    float sigmoid(float data) {
        return 1.F / (1.F + (float)exp(-data));
    }
};

#endif  // SIGMOID_H_
