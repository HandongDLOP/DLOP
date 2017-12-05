#include "NeuralNetwork.h"

template class NeuralNetwork<int>;
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;

// template<typename DTYPE>
// NeuralNetwork<DTYPE>::NeuralNetwork() {
// std::cout << "NeuralNetwork<DTYPE>::NeuralNetwork()" << '\n';
// this->Alloc();
// }
//
// template<typename DTYPE>
// NeuralNetwork<DTYPE>::~NeuralNetwork() {
// this->Delete();
// std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';
// }

// ===========================================================================================

template<typename DTYPE>
int NeuralNetwork<DTYPE>::Alloc() {
    std::cout << "NeuralNetwork<DTYPE>::Alloc()" << '\n';
    // m_aEnd->AddEdgebetweenOperators(m_aStart);
    return 1;
}

template<typename DTYPE>
void NeuralNetwork<DTYPE>::Delete() {
    std::cout << "NeuralNetwork<DTYPE>::Delete()" << '\n';
    // DeleteOperator()
    this->DeletePlaceholder();
    delete m_aStart;
    // delete m_pOptimizer;
}

// ===========================================================================================

template<typename DTYPE>
int NeuralNetwork<DTYPE>::AllocOptimizer(Optimizer<DTYPE> *pOptimizer) {
    pOptimizer->GetObjectOperator()->AllocOptimizer(pOptimizer);
    // pOptimizer->SetBatch(pOptimizer->GetObjectOperator()->GetOutput()->GetBatch());

    // Object Operator는 거의 100% Optimizer가 필요 없다.
    // m_aEnd->SetOptimizer(pOptimizer);
    return 1;
}

// int NeuralNetwork<DTYPE>::DeleteOperator() {
// m_aEnd->DeleteInputOperator();
// return 1;
// }

template<typename DTYPE>
int NeuralNetwork<DTYPE>::DeletePlaceholder() {
    Operator<DTYPE> **list_of_placeholder = m_aStart->GetOutputOperator();
    int num_of_placeholder                = m_aStart->GetOutputDegree();

    for (int i = 0; i < num_of_placeholder; i++) {
        delete list_of_placeholder[i];
        list_of_placeholder[i] = NULL;
    }

    return 1;
}

// ===========================================================================================

template<typename DTYPE>
Operator<DTYPE> *NeuralNetwork<DTYPE>::AddPlaceholder(Tensor<DTYPE> *pTensor, std::string pName) {
    std::cout << "NeuralNetwork<DTYPE>::Placeholder(Tensor<DTYPE> *, std::string )" << '\n';

    // placeholder의 경우 trainable하지 않다.
    Operator<DTYPE> *temp = new Placeholder<DTYPE>(pTensor, pName);

    temp->AddEdgebetweenOperators(m_aStart);

    return temp;
}

// ===========================================================================================

// 주소에 따라 조절되는 알고리즘 추가 필요
// forwardPropagate Algorithm 수정 필요
template<typename DTYPE>
int NeuralNetwork<DTYPE>::ForwardPropagate(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd) {
    pEnd->ForwardPropagate();

    return 1;
}

template<typename DTYPE>
int NeuralNetwork<DTYPE>::BackPropagate(Operator<DTYPE> *pStart, Optimizer<DTYPE> *pOptimizer) {
    // ObjectOperator로부터 시작한다
    pOptimizer->GetObjectOperator()->BackPropagate();

    return 1;
}

// ===========================================================================================

template<typename DTYPE>
int NeuralNetwork<DTYPE>::Run(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd) {
    this->ForwardPropagate(pStart, pEnd);
    return 1;
}

template<typename DTYPE>
int NeuralNetwork<DTYPE>::Run(Operator<DTYPE> *pEnd) {
    this->ForwardPropagate(m_aStart, pEnd);
    return 1;
}

template<typename DTYPE>
int NeuralNetwork<DTYPE>::Run(Optimizer<DTYPE> *pOptimizer) {
    this->ForwardPropagate(m_aStart, pOptimizer->GetObjectOperator());

    this->BackPropagate(m_aStart, pOptimizer);

    this->UpdateVariable(pOptimizer);
    return 1;
}

// ===========================================================================================

// ===========================================================================================

template<typename DTYPE>
void NeuralNetwork<DTYPE>::PrintGraph(Operator<DTYPE> *pEnd) {
    std::cout << '\n';
    pEnd->PrintGraph();
    std::cout << '\n';
}

template<typename DTYPE>
void NeuralNetwork<DTYPE>::PrintGraph(Optimizer<DTYPE> *pOptimizer) {
    std::cout << '\n';
    pOptimizer->GetObjectOperator()->PrintGraph();
    std::cout << '\n';
}

template<typename DTYPE>
void NeuralNetwork<DTYPE>::PrintData(Optimizer<DTYPE> *pOptimizer, int forceprint) {
    pOptimizer->GetObjectOperator()->PrintData(forceprint);
}

template<typename DTYPE>
void NeuralNetwork<DTYPE>::PrintData(Operator<DTYPE> *pOperator, int forceprint) {
    std::cout << "\n\n" << pOperator->GetName() << ": PrintData()" << '\n';

    if (pOperator->GetOutput() != NULL) pOperator->PrintOutput(forceprint);

    if (pOperator->GetDelta() != NULL) pOperator->PrintDelta(forceprint);

    if (pOperator->GetGradient() != NULL) pOperator->PrintGradient(forceprint);

    std::cout << pOperator->GetName() << ": ~PrintData() \n\n";
}

// ===========================================================================================

template<typename DTYPE>
int NeuralNetwork<DTYPE>::CreateGraph(Optimizer<DTYPE> *pOptimizer) {
    // Final Operator

    // SetEndOperator();

    // =====================================

    // SetOptimizer(pOptimizer);
    this->AllocOptimizer(pOptimizer);

    // =====================================

    return 1;
}

// int NeuralNetwork<DTYPE>::CreateGraph(){
//// 추후에 만들 Optimizer는 그 자체가 Trainable Operator주소를 가질 수 있도록 만들 것이다.
//// factory method 삭제예정
// }
