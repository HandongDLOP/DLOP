#include <iostream>

enum Optimizer_name {
    STOCHASTIC_GRADIENT_DESCENT,
    GD,
    MOMENTUM,
    ADAM
};

class Optimizer {
private:
    /* data */

public:
    Optimizer (){}
    virtual ~Optimizer (){}
};

class StochasticGradientDescent : public Optimizer{
private:
    /* data */

public:
    StochasticGradientDescent (){}
    virtual ~StochasticGradientDescent (){}
};

Optimizer* OptimizerFactory(Optimizer_name pOptimizer_name){
    if(pOptimizer_name == STOCHASTIC_GRADIENT_DESCENT){
        return new StochasticGradientDescent();
    }

    return new Optimizer();
}

int main(int argc, char const *argv[]) {

    Optimizer* optimizer = OptimizerFactory(STOCHASTIC_GRADIENT_DESCENT);



    return 0;
}
