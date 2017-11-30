#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_    value

#include "..//Operator.h"

class Convolution : public Operator {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Convolution::Alloc())
    Convolution(Operator *pInput, MetaParameter *pParam) : Operator(pInput, pParam) {
        std::cout << "Convolution::Convolution(Operator *, MetaParameter *)" << '\n';
        // Alloc(pInput, pParam);
    }

    Convolution(Operator *pInput, MetaParameter *pParam, std::string pName) : Operator(pInput, pParam, pName) {
        std::cout << "Convolution::Convolution(Operator *, MetaParameter *, std::string)" << '\n';
        // Alloc(pInput, pParam);
    }

    virtual ~Convolution() {
        std::cout << "Convolution::~Convolution()" << '\n';
    }

    virtual bool Alloc(Operator *pInput, MetaParameter *pParam) {

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';


        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        return true;
    }

};

#endif  // CONVOLUTION_H_
