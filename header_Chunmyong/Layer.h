#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>
#include <string>
#include "Tensor.h"

// Layer 의 연산에서 필요한 요소를 구현합니다.
class Layer {
    // 멤버로 가져야 하는 친구는 노드일 것으로 예상됩니다.

protected:
    int m_inputDim;
    int m_outputDim;

    Tensor *m_pInput;
    Tensor *m_aOutput;
    Tensor *m_aWeight;

    // Training 과정을 공부한 후 다시 확인해야 할 부분
    Tensor *m_aGradient;
    Tensor *m_aDelta;
    Tensor *m_aDeltabar;

public:
    Layer() {
        std::cout << "Layer::Layer()" << '\n';
        // Alloc();
    }

    virtual ~Layer() {
        std::cout << "Layer::~Layer()" << '\n';
    }

    int  Alloc();
    void Delete();

    // Get, Set
    void Getter() const;
    void Setter();

    //
};

#endif  // LAYER_H_
