#ifndef OBJECTIVE_FUNCTION_H_
#define OBJECTIVE_FUNCTION_H_    value

#include "Tensor.h"
#include "Layer.h"

// class Objective {
// private:
///* data */
//
// public:
// Objective() {}
//
// virtual ~Objective() {}
//
// ComputeDeltaBar(Tensor * pDesiredOutput);
// };

class MSE : public Layer {
private:
    /* data */

public:
    MSE(){
        std::cout << "MSE::MSE() : public Layer" << '\n';
    }
    virtual ~MSE(){
        std::cout << "MSE::~MSE()" << '\n';
    }
};

class SoftMax : public Layer {
private:
    /* data */

public:
    SoftMax (){
        std::cout << "SoftMax::SoftMax() : public Layer" << '\n';
    }
    virtual ~SoftMax (){
        std::cout << "SoftMax::SoftMax() : public Layer" << '\n';
    }
};

#endif  // OBJECTIVE_FUNCTION_H_
