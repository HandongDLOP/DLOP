#ifndef ADD_H_
#define ADD_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Add : public Operator {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Add::Alloc())
    Add(Operator *pInput1, Operator *pInput2) : Operator(pInput1, pInput2) {
        std::cout << "Add::Add(Operator *, MetaParameter *)" << '\n';
        Alloc(pInput1, pInput2);
    }

    Add(Operator *pInput1, Operator *pInput2, std::string pName) : Operator(pInput1, pInput2, pName) {
        std::cout << "Add::Add(Operator *, MetaParameter *, std::string)" << '\n';
        Alloc(pInput1, pInput2);
    }

    virtual ~Add() {
        std::cout << "Add::~Add()" << '\n';
    }

    virtual bool Alloc(Operator *pInput1, Operator *pInput2) {
        std::cout << "Add::Alloc(Operator *, Operator *)" << '\n';
        // if pInput1 and pInput2의 shape가 다르면 abort

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

        // Operator Alloc 마저 구현
        // 텐서 클론 구현
        // 텐서 더하기 구현
        if (GetInput()[0]->GetFlatDim() != GetInput()[1]->GetFlatDim()) {
            std::cout << "data has different flat dimension" << '\n';
            exit(0);
        }

        int size = GetInput()[0]->GetFlatDim();

        float *data0 = GetInput()[0]->GetData();

        float *data1 = GetInput()[1]->GetData();

        float *result = new float[size];

        for (int i = 0; i < size; i++) {
            result[i] = data0[i] + data1[i];
        }

        SetOutput(result);

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int size = GetInput()[0]->GetFlatDim();

        // Test code
        // Tensor * temp = Tensor::Constants(5, 1, 0, 0, 0, 1.0);
        //
        // temp->PrintData();
        //
        // float *delta = temp->GetData();

        float *delta = GetDelta()->GetData();

        float *_delta0 = new float[size];
        float *_delta1 = new float[size];

        // 위에서 내려온 delta를 그대로 흘린다.
        for(int i = 0; i < size; i++){
            _delta0[i] = _delta1[i] = delta[i];
        }

        GetInputOperator()[0]->SetDelta(_delta0);

        GetInputOperator()[0]->GetDelta()->PrintData();

        GetInputOperator()[1]->SetDelta(_delta1);

        GetInputOperator()[1]->GetDelta()->PrintData();


        return true;
    }
};

#endif  // ADD_H_
