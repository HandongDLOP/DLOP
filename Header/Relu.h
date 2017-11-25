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

        // Tensor *temp_output = new Tensor(GetInput()[0]->Getshape());

        SetOutput(new Tensor(GetInputOperator()[0]->GetOutput()->Getshape()));

        // Tensor *temp_Gradient = new Tensor(GetInput()[0]->Getshape());
        //
        // SetGradient(temp_Gradient);

        // Tensor *tempdelta_for_input = new Tensor(GetInput()[0]->Getshape());

        SetDelta(new Tensor(GetInputOperator()[0]->GetOutput()->Getshape()));

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int size = GetInputOperator()[0]->GetOutput()->GetFlatDim();

        float *data = GetInputOperator()[0]->GetOutput()->GetData();

        float *result = GetOutput()->GetData();

        for (int i = 0; i < size; i++) {
            result[i] = Max(data[i], 0.0);
        }

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int size = GetOutput()->GetFlatDim();

        float *output = GetOutput()->GetData();

        float *delta = GetDelta()->GetData();

        float *delta_for_input = GetInputOperator()[0]->GetDelta()->GetData();

        for (int i = 0; i < size; i++) {
            if (output[i] > 0.0) {
                delta_for_input[i] = delta[i];
            } else {
                delta_for_input[i] = 0;
            }
        }

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
