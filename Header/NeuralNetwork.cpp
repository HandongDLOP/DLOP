#include "NeuralNetwork.h"

template class NeuralNetwork<int>;
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;

template<typename DTYPE> NeuralNetwork<DTYPE>::NeuralNetwork() {
    std::cout << "NeuralNetwork<DTYPE>::NeuralNetwork()" << '\n';
    this->Alloc();
}

template<typename DTYPE> NeuralNetwork<DTYPE>::~NeuralNetwork() {
    std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';
    this->Delete();
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Alloc() {
    std::cout << "NeuralNetwork<DTYPE>::Alloc()" << '\n';
    m_aaPlaceholder  = NULL;
    m_aaOperator     = NULL;
    m_aaTensorholder = NULL;
    m_aOptimizer     = NULL;

    numOfPlaceholder  = 0;
    numOfOperator     = 0;
    numOfTensorholder = 0;

    return TRUE;
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::Delete() {
    std::cout << "NeuralNetwork<DTYPE>::Delete()" << '\n';

    if (m_aaPlaceholder) {
        for (int i = 0; i < numOfPlaceholder; i++) {
            delete m_aaPlaceholder[i];
        }
        delete[] m_aaPlaceholder;
    }

    if (m_aaOperator) {
        for (int i = 0; i < numOfOperator; i++) {
            delete m_aaOperator[i];
        }
        delete[] m_aaOperator;
    }

    if (m_aaTensorholder) {
        for (int i = 0; i < numOfTensorholder; i++) {
            delete m_aaTensorholder[i];
        }
        delete[] m_aaTensorholder;
    }

    delete m_aOptimizer;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AddPlaceholder(Placeholder<DTYPE> *pPlaceholder) {
    return NULL;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AddOperator(Operator<DTYPE> *pOperator) {
    return NULL;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AddTensorholder(Tensorholder<DTYPE> *pTensorholder) {
    return NULL;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AddOptimizer(Optimizer<DTYPE> *pOptimizer) {
    return NULL;
}

// ===========================================================================================

// template<typename DTYPE>
// int NeuralNetwork<DTYPE>::AllocOptimizer(Optimizer<DTYPE> *pOptimizer) {
// pOptimizer->GetObjectOperator()->AllocOptimizer(pOptimizer);
//// pOptimizer->SetBatch(pOptimizer->GetObjectOperator()->GetOutput()->GetBatch());
//
//// Object Operator는 거의 100% Optimizer가 필요 없다.
//// m_aEnd->SetOptimizer(pOptimizer);
// return TRUE;
// }

// int NeuralNetwork<DTYPE>::DeleteOperator() {
// m_aEnd->DeleteInputOperator();
// return TRUE;
// }

// template<typename DTYPE>
// int NeuralNetwork<DTYPE>::DeletePlaceholder() {
// Operator<DTYPE> **list_of_placeholder = m_aStart->GetOutputOperator();
// int num_of_placeholder                = m_aStart->GetOutputDegree();
//
// for (int i = 0; i < num_of_placeholder; i++) {
// delete list_of_placeholder[i];
// list_of_placeholder[i] = NULL;
// }
//
// return TRUE;
// }

// ===========================================================================================

// template<typename DTYPE>
// Operator<DTYPE> *NeuralNetwork<DTYPE>::AddPlaceholder(Tensor<DTYPE> *pTensor, std::string pName) {
// std::cout << "NeuralNetwork<DTYPE>::Placeholder(Tensor<DTYPE> *, std::string )" << '\n';
//
//// placeholder의 경우 trainable하지 않다.
// Operator<DTYPE> *temp = new Placeholder<DTYPE>(pTensor, pName);
//
// temp->AddEdgebetweenOperators(m_aStart);
//
// return temp;
// }

// ===========================================================================================

// 주소에 따라 조절되는 알고리즘 추가 필요
// forwardPropagate Algorithm 수정 필요
// template<typename DTYPE>
// int NeuralNetwork<DTYPE>::ForwardPropagate(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd) {
// pEnd->ForwardPropagate();
//
// return TRUE;
// }
//
// template<typename DTYPE>
// int NeuralNetwork<DTYPE>::BackPropagate(Operator<DTYPE> *pStart, Optimizer<DTYPE> *pOptimizer) {
//// ObjectOperator로부터 시작한다
// pOptimizer->GetObjectOperator()->BackPropagate();
//
// return TRUE;
// }

// ===========================================================================================

template<typename DTYPE> int NeuralNetwork<DTYPE>::Run(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd) {
    pEnd->ForwardPropagate();
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Run(Operator<DTYPE> *pEnd) {
    pEnd->ForwardPropagate();
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Run(Optimizer<DTYPE> *pOptimizer) {
    pOptimizer->GetObjectOperator()->ForwardPropagate();
    pOptimizer->GetObjectOperator()->BackPropagate();
    pOptimizer->UpdateVariable();
    return TRUE;
}

// ===========================================================================================

template<typename DTYPE> int NeuralNetwork<DTYPE>::CreateGraph() {

    return TRUE;
}

// int NeuralNetwork<DTYPE>::CreateGraph(){
//// 추후에 만들 Optimizer는 그 자체가 Trainable Operator주소를 가질 수 있도록 만들 것이다.
//// factory method 삭제예정
// }
