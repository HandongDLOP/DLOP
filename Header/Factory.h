#ifndef OPTIMIZER_FACTORY_H_
#define OPTIMIZER_FACTORY_H_    value

// #include <string>
#include "StochasticGradientDescent.h"

enum Optimizer_name {
    STOCHASTIC_GRADIENT_DESCENT,
    GD,
    MOMENTUM,
    ADAM
};

class Factory {
private:
    /* data */

public:
    Factory (){}
    virtual ~Factory (){}

    static Optimizer* OptimizerFactory(Optimizer_name pOptimizer_name){
        std::cout << "Factory::OptimizerFactory(Optimizer_name)" << '\n';

        if(pOptimizer_name == STOCHASTIC_GRADIENT_DESCENT){
            return new StochasticGradientDescent(0.6);
        }

        return NULL;
    }
};




#endif  // OPTIMIZER_FACTORY_H_
