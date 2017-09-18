#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const int p_noOperator) : MaxOperator(p_noOperator) {
    std::cout << "NeuralNetwork::NeuralNetwork(int p_noOperator)" << '\n';
    Alloc();
}

NeuralNetwork::~NeuralNetwork() {
    Delete();
    std::cout << "NeuralNetwork::~NeuralNetwork()" << '\n';
}

bool NeuralNetwork::Alloc() {
    std::cout << "NeuralNetwork::Alloc()" << '\n';
    *m_aOperator = new Operator[MaxOperator];
    return true;
}

void NeuralNetwork::Delete() {
    std::cout << "NeuralNetwork::Delete()" << '\n';

    for (int count = 0; count < m_noOperator; count++) {
        std::cout << count << '\n';
        delete m_aOperator[count];
    }
    // delete [] m_aOperator;

    // delete m_aOperator[m_noOperator];
}

bool NeuralNetwork::AddOperator(Operator *Type) {
    std::cout << "NeuralNetwork::CreateOperator(Operator * Type)" << '\n';

    if (m_noOperator >= MaxOperator) {
        std::cout << "Error!" << '\n';
        return false;
    }

    // Create Operator!
    m_aOperator[m_noOperator] = Type;

    m_noOperator++;

    return true;
}

bool NeuralNetwork::AddObjective(Objective * Type){
    std::cout << "NeuralNetwork::CreateOperator(Operator * Type)" << '\n';

    if (m_noOperator >= MaxOperator) {
        std::cout << "Error!" << '\n';
        return false;
    }

    // Create Operator!
    // m_aOperator[m_noOperator] = Type;

    // m_noOperator++;

    return true;
}
