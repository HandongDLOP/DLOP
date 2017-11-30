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
    // m_aEnd->AddEdgebetweenOperators(m_aStart);
    return true;
}

void NeuralNetwork::Delete() {
    std::cout << "NeuralNetwork::Delete()" << '\n';
    // DeleteOperator()
    DeletePlaceholder();
    delete m_aStart;
    // delete m_pOptimizer;
}

// ===========================================================================================


bool NeuralNetwork::AllocOptimizer(Optimizer *pOptimizer) {
    pOptimizer->GetObjectOperator()->AllocOptimizer(pOptimizer);
    pOptimizer->SetBatch(pOptimizer->GetObjectOperator()->GetOutput()->GetBatch());

    // Object Operator는 거의 100% Optimizer가 필요 없다.
    // m_aEnd->SetOptimizer(pOptimizer);
    return true;
}

// bool NeuralNetwork::DeleteOperator() {
// m_aEnd->DeleteInputOperator();
// return true;
// }

bool NeuralNetwork::DeletePlaceholder() {
    Operator **list_of_placeholder = m_aStart->GetOutputOperator();
    int num_of_placeholder         = m_aStart->GetOutputDegree();

    for (int i = 0; i < num_of_placeholder; i++) {
        delete list_of_placeholder[i];
        list_of_placeholder[i] = NULL;
    }

    return true;
}

// ===========================================================================================

Operator * NeuralNetwork::AddPlaceholder(Tensor *pTensor, std::string pName) {
    std::cout << "NeuralNetwork::Placeholder(Tensor *, std::string )" << '\n';

    // placeholder의 경우 trainable하지 않다.
    Operator *temp = new Placeholder(pTensor, pName);

    temp->AddEdgebetweenOperators(m_aStart);

    return temp;
}

// ===========================================================================================

// 주소에 따라 조절되는 알고리즘 추가 필요
// forwardPropagate Algorithm 수정 필요
bool NeuralNetwork::ForwardPropagate(Operator *pStart, Operator *pEnd) {
    pEnd->ForwardPropagate();

    return true;
}

bool NeuralNetwork::BackPropagate(Operator *pStart, Optimizer *pOptimizer) {
    // ObjectOperator로부터 시작한다
    pOptimizer->GetObjectOperator()->BackPropagate();

    return true;
}

// ===========================================================================================


bool NeuralNetwork::Run(Operator *pStart, Operator *pEnd) {
    ForwardPropagate(pStart, pEnd);
    return true;
}

bool NeuralNetwork::Run(Operator *pEnd) {
    ForwardPropagate(m_aStart, pEnd);
    return true;
}

bool NeuralNetwork::Run(Optimizer *pOptimizer) {
    ForwardPropagate(m_aStart, pOptimizer->GetObjectOperator());

    BackPropagate(m_aStart, pOptimizer);

    UpdateVariable(pOptimizer);
    return true;
}

// ===========================================================================================

// ===========================================================================================

void NeuralNetwork::PrintGraph(Operator *pEnd) {
    std::cout << '\n';
    pEnd->PrintGraph();
    std::cout << '\n';
}

void NeuralNetwork::PrintGraph(Optimizer *pOptimizer) {
    std::cout << '\n';
    pOptimizer->GetObjectOperator()->PrintGraph();
    std::cout << '\n';
}

void NeuralNetwork::PrintData(Optimizer *pOptimizer, int forceprint) {
    pOptimizer->GetObjectOperator()->PrintData(forceprint);
}

void NeuralNetwork::PrintData(Operator *pOperator, int forceprint) {
    std::cout << "\n\n" << pOperator->GetName() << ": PrintData()" << '\n';

    if (pOperator->GetOutput() != NULL) pOperator->PrintOutput(forceprint);

    if (pOperator->GetDelta() != NULL) pOperator->PrintDelta(forceprint);

    if (pOperator->GetGradient() != NULL) pOperator->PrintGradient(forceprint);

    std::cout << pOperator->GetName() << ": ~PrintData() \n\n";
}

// ===========================================================================================

bool NeuralNetwork::CreateGraph(Optimizer *pOptimizer) {
    // Final Operator

    // SetEndOperator();

    // =====================================

    // SetOptimizer(pOptimizer);
    AllocOptimizer(pOptimizer);

    // =====================================

    return true;
}

// bool NeuralNetwork::CreateGraph(){
//// 추후에 만들 Optimizer는 그 자체가 Trainable Operator주소를 가질 수 있도록 만들 것이다.
//// factory method 삭제예정
// }
