#ifndef RELU_H_
#define RELU_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Relu : public Operator {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Relu::Alloc())
    Relu(Operator *pInput, std::string pName) : Operator(pInput, pName) {
        std::cout << "/* Relu::Relu(Operator *) */" << '\n';
        Alloc(pInput);
    }

    virtual ~Relu() {
        std::cout << "Relu::~Relu()" << '\n';
    }

    virtual bool Alloc(Operator *pInput) {
        std::cout << "Relu::Alloc(Operator *, Operator *)" << '\n';

        Tensor *temp_output = new Tensor(GetInput()[0]->Getshape());

        SetOutput(temp_output);

        Tensor *temp_Gradient = new Tensor(GetInput()[0]->Getshape());

        SetGradient(temp_Gradient);

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
            result[i] = Max(data[i], 0.0);
        }

        SetOutput(result);

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int size = GetGradient()->GetFlatDim();

        float *output = GetOutput()->GetData();

        float *grad = GetGradient()->GetData();

        //// Test code
        // Tensor * temp = Tensor::Constants(6, 1, 0, 0, 0, 1.0);
        //
        // float *delta = temp->GetData();

        float *delta = GetDelta()->GetData();

        float *_grad  = new float[size];
        float *_delta = new float[size];

        for (int i = 0; i < size; i++) {
            if (output[i] > 0.0) {
                _grad[i]  = delta[i] + grad[i]; // 필요없으나 참고용으로 잠시 놔둠
                _delta[i] = delta[i];
            } else {
                _grad[i]  = grad[i]; // 필요없으나 참고용으로 잠시 놔둠
                _delta[i] = 0;
            }
        }

        SetGradient(_grad);

        GetGradient()->PrintData();

        GetInputOperator()[0]->SetDelta(_delta);

        GetInputOperator()[0]->GetDelta()->PrintData();

        return true;
    }

    // for relu
    float Max(float data1, float data2) {
        float temp = 0.0;

        if (data1 >= data2) temp = data1;
        else temp = data2;
        return temp;
    }
};

#endif  // RELU_H_
