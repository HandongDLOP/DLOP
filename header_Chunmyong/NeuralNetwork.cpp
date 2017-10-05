#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
    std::cout << "NeuralNetwork::NeuralNetwork(int p_noOperator)" << '\n';
    Alloc();
}

NeuralNetwork::~NeuralNetwork() {
    Delete();
    std::cout << "NeuralNetwork::~NeuralNetwork()" << '\n';
}

bool NeuralNetwork::Alloc() {
    std::cout << "NeuralNetwork::Alloc()" << '\n';
    return true;
}

void NeuralNetwork::Delete() {
    std::cout << "NeuralNetwork::Delete()" << '\n';
}

bool NeuralNetwork::PutOperator(std::string Op, MetaParameter *pParam, LayerType) {
    std::cout << "NeuralNetwork::CreateOperator(Operator op)" << '\n';

    // Operator를 어떻게 정의할 것인지에 대해서 생각할 필요가 있음

    return true;
}

bool NeuralNetwork::ForwardPropagate() {
    if (m_aOperator == NULL) {
        std::cout << "There is no linked Operator!" << '\n';
        return false;
    }

    // 시작하는 주소의 Propagate를 실행
    // 시작하는 주소가 input Layer일 경우 (forward)Propagate는 Preorder의 형식
    // 시작하는 주소가 Ouput Layer일 경우는 Postorder의 형식
    m_aOperator->ForwardPropagate();

    return true;
}

bool NeuralNetwork::BackPropagate() {
    if (m_aOperator == NULL) {
        std::cout << "There is no linked Operator!" << '\n';
        return false;
    }

    // 시작하는 주소의 Backropagate를 실행
    // 시작하는 주소가 input Layer일 경우 BackPropagate는 Postorder의 형식
    // 시작하는 주소가 Ouput Layer일 경우는 Preorder의 형식
    m_aOperator->BackPropagate();

    return true;
}

bool NeuralNetwork::Training(const int p_maxEpoch) {
    for (int epoch = 0; epoch < p_maxEpoch; epoch++) {
        ForwardPropagate();
        BackPropagate();
        // print some data
    }

    return true;
}

bool NeuralNetwork::Testing() {
    ForwardPropagate();
    return true;
}
