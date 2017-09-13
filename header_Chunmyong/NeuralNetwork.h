#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Tensor.h"
#include "Layer.h"

class NeuralNetwork {
protected:
    // Layer 개수
    // 추후에는 noLayer를 정하지 않아도 되는 방법을 알아보고자 합니다.
    int countofLayer = 0;
    int m_noLayer;
    Layer *m_aLayer[];

public:
    // Layer의 개수를 정합니다.
    NeuralNetwork(int p_noLayer);
    ~NeuralNetwork();

    // 각 Layer 객체를 저장할 array를 만듭니다.
    bool Alloc();
    void Delete();

    bool CreateLayer(Layer * Type);
};

#endif  // NEURALNETWORK_H_
