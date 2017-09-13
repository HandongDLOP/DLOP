#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const int p_noLayer) : m_noLayer(p_noLayer) {
    std::cout << "NeuralNetwork::NeuralNetwork(int p_noLayer)" << '\n';
    // Alloc();
}

NeuralNetwork::~NeuralNetwork(){
    Delete();
    std::cout << "NeuralNetwork::~NeuralNetwork()" << '\n';
}

bool NeuralNetwork::Alloc() {
    std::cout << "NeuralNetwork::Alloc()" << '\n';
    // *m_aLayer = new Layer[m_noLayer];
    return true;
}

void NeuralNetwork::Delete(){
    std::cout << "NeuralNetwork::Delete()" << '\n';

    for (int count = countofLayer - 1; count > 0 ; count--) delete m_aLayer[count];
    // delete [] *m_aLayer;

    // delete m_aLayer[countofLayer];

    return;
}

bool NeuralNetwork::CreateLayer(Layer * Type){
    std::cout << "NeuralNetwork::CreateLayer(Layer * Type)" << '\n';

    if (countofLayer >= m_noLayer) {
        std::cout << "Error!" << '\n';
        return false;
    }

    // Create Layer!
    m_aLayer[countofLayer] = Type;

    countofLayer++;

    return true;
}
