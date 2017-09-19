#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Tensor.h"
#include "Operator.h"
#include "Objective.h"

class NeuralNetwork {
private:
    // Operator 개수
    // 추후에는 noOperator를 정하지 않아도 되는 방법을 알아보고자 합니다.
    int m_noOperator = 0;
    int maxOperator;

    // 그래프 형식으로 바꿔야 합니다.
    // 그래프가 되기 위해서는 다음 오퍼레이터의 링크를 건네는 Operator가 필요합니다.
    Operator *m_aOperator;

    // 각 Operator 객체를 저장할 array를 만듭니다.
    bool Alloc();
    void Delete();

public:
    // Operator의 개수를 정합니다.
    NeuralNetwork(int p_maxOperator);
    virtual ~NeuralNetwork();

    // Operator 전달하는 형식이 될 수 있게 할 수 있나?
    // 생각을 좀 더 해보기
    // identifier에 대한 부분을 추가해야함
    bool AddOperator(Operator *Type);
    bool AddObjective(Objective *Type);

    // Propagate
    bool Propagate();

    // BackPropagate
    bool BackPropagate();

    // For NeuralNetwork Training
    bool Training(const int p_maxEpoch);
    bool Testing();

};

#endif  // NEURALNETWORK_H_
