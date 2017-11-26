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
    DeleteOperator();
}

// ===========================================================================================


bool NeuralNetwork::AllocOptimizer(Optimizer * pOptimizer) {
    _m_aEnd->AllocOptimizer(pOptimizer);

    // 마지막 Operator는 거의 100% Optimizer가 필요 없다.
    // _m_aEnd->SetOptimizer(pOptimizer);
    return true;
}

bool NeuralNetwork::DeleteOperator() {
    _m_aEnd->DeleteInputOperator();
    delete _m_aEnd;
    delete _m_aOptimizer;
    return true;
}

// ===========================================================================================

// Operator * NeuralNetwork::AddPlaceholder(TensorShape *pshape) {
//     std::cout << "NeuralNetwork::Placeholder(TensorShape *)" << '\n';
//
//     Operator *temp = new Placeholder(pshape);
//
//     temp->AddEdgebetweenOperators(_m_pStart);
//
//     return temp;
// }
//
// Operator * NeuralNetwork::AddPlaceholder(TensorShape *pshape, std::string pName) {
//     std::cout << "NeuralNetwork::Placeholder(TensorShape *, std::string )" << '\n';
//
//     Operator *temp = new Placeholder(pshape, pName);
//
//     temp->AddEdgebetweenOperators(_m_pStart);
//
//     return temp;
// }

Operator * NeuralNetwork::AddPlaceholder(Tensor *pTensor, std::string pName) {
    std::cout << "NeuralNetwork::Placeholder(Tensor *, std::string )" << '\n';

    // placeholder의 경우 trainable하지 않다.
    Operator *temp = new Placeholder(pTensor, pName);

    temp->AddEdgebetweenOperators(_m_pStart);

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

// ===========================================================================================

bool NeuralNetwork::CreateGraph(Optimizer* pOptimizer, Operator *pEnd){
    SetEndOperator(pEnd);
    SetOptimizer(pOptimizer);
    AllocOptimizer(pOptimizer);

    return true;
}

// bool NeuralNetwork::CreateGraph(){
//     // 추후에 만들 Optimizer는 그 자체가 Trainable Operator주소를 가질 수 있도록 만들 것이다.
//     // factory method 삭제예정
// }

bool NeuralNetwork::SetEndOperator() {
    _m_aEnd = _m_pStart->CheckEndOperator();

    // std::cout << _m_aEnd->GetName() << '\n';
    // std::cout << temp->GetName() << '\n';

    return true;
}
