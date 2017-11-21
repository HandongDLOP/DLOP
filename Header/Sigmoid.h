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

        Tensor *temp_output = new Tensor(GetInput()[0]->Getshape());

        SetOutput(temp_output);

        Tensor *temp_delta = new Tensor(GetInput()[0]->Getshape());

        SetDelta(temp_delta);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int size = GetInput()[0]->GetFlatDim();

        float *data = GetInput()[0]->GetData();

        float *result = new float[GetInput()[0]->GetFlatDim()];

        for (int i = 0; i < size; i++) {
            result[i] = sigmoid(data[i]);
        }

        SetOutput(result);

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int size = GetOutput()->GetFlatDim();

        float *output = GetOutput()->GetData();

        //// Test code
        // Tensor * temp = Tensor::Constants(6, 1, 0, 0, 0, 1.0);
        //
        // float *delta = temp->GetData();

        float *delta = GetDelta()->GetData();

        float *_delta = new float[size];

        for (int i = 0; i < size; i++) {
                _delta[i] = delta[i] * output[i] * (1 - output[i]);
        }

        GetInputOperator()[0]->SetDelta(_delta);

        GetInputOperator()[0]->GetDelta()->PrintData();

        return true;
    }

    // for Sigmoid
    float sigmoid(float data) {
        return 1.F / (1.F + (float)exp(-data));
    }
};

#endif  // SIGMOID_H_
