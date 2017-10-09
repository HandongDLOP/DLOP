#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_    value

#include <iostream>

#include "Ark.h"
#include "Operator.h"

class Convolution : public Operator {
public:
    Convolution(Ark *pInput, MetaParameter *pParam, LayerType LayerType = HIDDEN) : Operator(pInput, pParam, HIDDEN) {
        Alloc(pInput, pParam);
    }

    virtual ~Convolution() {
        std::cout << "Convolution::~Convolution()" << '\n';
    }
};

#endif  // CONVOLUTION_H_
