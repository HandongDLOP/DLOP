#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const int p_maxOperator) : maxOperator(p_maxOperator) {
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

bool NeuralNetwork::AddOperator(Operator *Type) {
    std::cout << "NeuralNetwork::CreateOperator(Operator * Type)" << '\n';

    if (m_noOperator >= maxOperator) {
        std::cout << "Error!" << '\n';
        return false;
    }

    m_noOperator++;

    return true;
}

bool NeuralNetwork::AddObjective(Objective *Type) {
    std::cout << "NeuralNetwork::CreateOperator(Operator * Type)" << '\n';

    // Objective를 마지막에 쌓아올려야 한다.
    if (m_noOperator >= maxOperator) {
        std::cout << "Error!" << '\n';
        return false;
    }

    return true;
}

bool NeuralNetwork::Propagate() {
    if (m_aOperator == NULL) {
        std::cout << "There is no linked Operator!" << '\n';
        return false;
    }

    // 시작하는 주소의 Propagate를 실행
    // 시작하는 주소가 input Layer일 경우 (forward)Propagate는 Preorder의 형식
    // 시작하는 주소가 Ouput Layer일 경우는 Postorder의 형식
    m_aOperator->PrePropagate();

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
    m_aOperator->PreBackPropagate();

    return true;
}

bool NeuralNetwork::Training(const int p_maxEpoch) {
    for (int epoch = 0; epoch < p_maxEpoch; epoch++) {
        Propagate();
        BackPropagate();
        // print some data
    }

    return true;
}

bool NeuralNetwork::Testing(){
    Propagate();
    return true;
}
