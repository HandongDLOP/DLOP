#ifndef OBJECTIVE_FROM_OBJECTIVE_H_
#define OBJECTIVE_FROM_OBJECTIVE_H_    value

#include "Tensor.h"
#include "Objective.h"

class SoftMax : public Objective {
private:
    /* data */

public:
    SoftMax (){
        std::cout << "SoftMax::SoftMax() : public Operator" << '\n';
    }
    virtual ~SoftMax (){
        std::cout << "SoftMax::~SoftMax() : public Operator" << '\n';
    }
};

#endif  // OBJECTIVE_FROM_OBJECTIVE_H_
