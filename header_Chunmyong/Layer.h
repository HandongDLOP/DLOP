#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>
#include <string>
#include "Tensor.h"
#include "Activation.h"

    // Layer 의 연산에서 필요한 요소를 구현합니다.
    class Layer {
    // 멤버로 가져야 하는 친구는 노드일 것으로 예상됩니다.

private:
    int m_inputDim;
    int m_outputDim;

    Tensor m_pInput;
    Tensor m_aOutput;
    Tensor m_aWeight;

    // Training 과정을 공부한 후 다시 확인해야 할 부분
    Tensor m_aGradient;
    Tensor m_aDelta;
    Tensor m_aDeltabar;

    Activation *m_Activation;

public:
    Layer(Activation * p_Activation) {

        // Alloc();
    }

    virtual ~Layer() {
        // Delete();
    }

    // Alloc
    int Alloc() {
        return 0;
    }

    void Delete() {}

    // Get, Set
    void Getter() {}  // 추후 필요한 요소 정하기

    void Setter() {}  // 추후 필요한 요소 정하기


    int Propagate() {
        return 0;
    }

    int BackPropagation() {
        return 0;
    }

    int ComputeGradient() {
        return 0;
    }

};

#endif  // LAYER_H_
