#ifndef ACTIVATION_H_
#define ACTIVATION_H_    value

#include <string>
#include "Tensor.h"

class Activation {
private:
    Tensor *m_input;
    Tensor *m_output;
    Tensor *m_aWeight;

    // Training 과정을 공부한 후 다시 확인해야 할 부분
    Tensor *m_aGradient;
    Tensor *m_aDelta;
    Tensor *m_aDeltabar;

    Tensor net;  // layer의 output_dim과 같을 것으로 예상
    // construct 할 때 생성 (초기값 0)

public:
    Activation() {}

    virtual ~Activation() {}
};

class Relu : public Activation {
private:
    /* data */

public:
    Relu() {}

    virtual ~Relu() {}
};

class Linear : public Activation {
private:
    /* data */

public:
    Linear() {}

    virtual ~Linear() {}
};

#endif  // ACTIVATION_H_
