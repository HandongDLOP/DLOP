#include "Objective.h"

template class Objective<int>;
template class Objective<float>;
template class Objective<double>;

template<typename DTYPE> Objective<DTYPE>::Objective(std::string pName) {
    std::cout << "Objective<DTYPE>::Objective()" << '\n';
    m_aResult = NULL;
    m_aGradient = NULL;
    m_pInputNeuralNetwork = NULL;
    m_pInputOperator = NULL;
    m_pInputTensor = NULL;
    m_name = pName;
}

template<typename DTYPE> Objective<DTYPE>::Objective(NeuralNetwork<DTYPE> *pNeuralNetwork, std::string pName) {
    std::cout << "Objective<DTYPE>::Objective()" << '\n';
    m_aResult = NULL;
    m_aGradient = NULL;
    m_pInputNeuralNetwork = NULL;
    m_pInputOperator = NULL;
    m_pInputTensor = NULL;
    m_name = pName;
    Alloc(pNeuralNetwork);
}

template<typename DTYPE> Objective<DTYPE>::~Objective() {
    std::cout << "Objective<DTYPE>::~Objective()" << '\n';
    this->Delete();
}

template<typename DTYPE> int Objective<DTYPE>::Alloc(NeuralNetwork<DTYPE> *pNeuralNetwork) {
    std::cout << "Objective<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';

    m_pInputNeuralNetwork = pNeuralNetwork;
    m_pInputOperator = m_pInputNeuralNetwork->GetResultOperator();
    m_pInputTensor = m_pInputOperator->GetResult();

    return TRUE;
}

template<typename DTYPE> void Objective<DTYPE>::Delete() {
    if (m_aResult) {
        delete m_aResult;
        m_aResult = NULL;
    }

    if (m_aGradient) {
        delete[] m_aGradient;
        m_aGradient = NULL;
    }
}

template<typename DTYPE> void Objective<DTYPE>::SetResult(Tensor<DTYPE> *pTensor) {
    m_aResult = pTensor;
}

template<typename DTYPE> void Objective<DTYPE>::SetGradient(Tensor<DTYPE> *pTensor) {
    m_aGradient = pTensor;
}

template<typename DTYPE> Tensor<DTYPE> *Objective<DTYPE>::GetResult() const {
    return m_aResult;
}

template<typename DTYPE> Tensor<DTYPE> *Objective<DTYPE>::GetGradient() const {
    return m_aGradient;
}

template<typename DTYPE> NeuralNetwork<DTYPE> *Objective<DTYPE>::GetNeuralNetwork() const {
    return m_pInputNeuralNetwork;
}

template<typename DTYPE> Operator<DTYPE> *Objective<DTYPE>::GetOperator() const {
    return m_pInputOperator;
}

template<typename DTYPE> Tensor<DTYPE> *Objective<DTYPE>::GetTensor() const {
    return m_pInputTensor;
}

template<typename DTYPE> std::string Objective<DTYPE>::GetName() const {
    return m_name;
}

template<typename DTYPE> Tensor<DTYPE> *Objective<DTYPE>::ForwardPropagate(Operator<DTYPE> *pLabel) {
    std::cout << this->GetName() << '\n';
    return NULL;
}

template<typename DTYPE> Tensor<DTYPE> *Objective<DTYPE>::BackPropagate() {
    std::cout << this->GetName() << '\n';
    return NULL;
}

// int main(int argc, char const *argv[]) {
// Objective<int> *temp1 = new Objective<int>("temp1");
// Objective<int> *temp2 = new Objective<int>(temp1, "temp2");
// Objective<int> *temp3 = new Objective<int>(temp1, temp2, "temp3");
//
// std::cout << temp3->GetInput()[0]->GetName() << '\n';
// std::cout << temp3->GetInput()[1]->GetName() << '\n';
// std::cout << temp1->GetOutput()[0]->GetName() << '\n';
//
// delete temp1;
// delete temp2;
// delete temp3;
//
// return 0;
// }
