#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
    std::cout << "NeuralNetwork::NeuralNetwork()" << '\n';
    Alloc();
}

NeuralNetwork::~NeuralNetwork() {
    Delete();
    std::cout << "NeuralNetwork::~NeuralNetwork()" << '\n';
}

// ===========================================================================================

bool NeuralNetwork::Alloc() {
    std::cout << "NeuralNetwork::Alloc()" << '\n';
    return true;
}

void NeuralNetwork::Delete() {
    std::cout << "NeuralNetwork::Delete()" << '\n';
    PropagateDelete();
    delete _m_aEnd;
}

// ===========================================================================================

bool NeuralNetwork::PropagateDelete() {
    _m_aEnd->PropagateDelete();
    return true;
}

// ===========================================================================================

Operator * NeuralNetwork::AddPlaceholder() {
    std::cout << "NeuralNetwork::Placeholder()" << '\n';

    Operator *temp = new Placeholder();

    // 쌍방향 연결관계 추가
    temp->_AddInputEdge(_m_pStart);
    _m_pStart->_AddOutputEdge(temp);

    return temp;
}

Operator * NeuralNetwork::AddPlaceholder(std::string pName) {
    std::cout << "NeuralNetwork::Placeholder()" << '\n';

    Operator *temp = new Placeholder(pName);

    // 쌍방향 연결관계 추가
    temp->_AddInputEdge(_m_pStart);
    _m_pStart->_AddOutputEdge(temp);

    return temp;
}

// ===========================================================================================

// 주소에 따라 조절되는 알고리즘 추가 필요
bool NeuralNetwork::ForwardPropagate(Operator *_pStart, Operator *_pEnd) {
    if (_pEnd == NULL) {
        if (_m_aEnd == NULL) {
            std::cout << "There is no linked Operator!" << '\n';
            return false;
        } else _pEnd = _m_aEnd;
    }

    // 시작하는 주소의 Propagate를 실행
    // 시작하는 주소가 input Layer일 경우 (forward)Propagate는 Preorder의 형식
    // 시작하는 주소가 Ouput Layer일 경우는 Postorder의 형식
    _pEnd->ForwardPropagate();

    return true;
}

bool NeuralNetwork::BackPropagate(Operator *_pStart, Operator *_pEnd) {
    if (_pEnd == NULL) {
        if (_m_aEnd == NULL) {
            std::cout << "There is no linked Operator!" << '\n';
            return false;
        } else _pEnd = _m_aEnd;
    }

    // 시작하는 주소의 Backropagate를 실행
    // 시작하는 주소가 input Layer일 경우 BackPropagate는 Postorder의 형식
    // 시작하는 주소가 Ouput Layer일 경우는 Preorder의 형식
    _pEnd->BackPropagate();

    return true;
}

// ===========================================================================================

bool NeuralNetwork::Training(Operator *_pStart, Operator *_pEnd) {
    std::cout << "\n<<<ForwardPropagate>>>\n" << '\n';

    ForwardPropagate(_pStart, _pEnd);

    std::cout << "\n<<<BackPropagate>>>\n" << '\n';

    BackPropagate(_pStart, _pEnd);

    std::cout << '\n';

    return true;
}

bool NeuralNetwork::Testing(Operator *_pStart, Operator *_pEnd) {
    std::cout << "\n<<<ForwardPropagate>>>\n" << '\n';

    ForwardPropagate(_pStart, _pEnd);

    std::cout << '\n';

    return true;
}
