#include <iostream>
#include <string>

class Layer {
private:
    /* data */

public:
    Layer() {}

    virtual ~Layer() {}

    virtual void Do() {}
};

// class Activation {
// private:
///* data */
//
// public:
// Activation ();
// virtual ~Activation ();
//
// virtual void Do() {}
// };

class Conv2D : public Layer {
private:
    /* data */

public:
    Conv2D() {
        std::cout << "Conv2d()" << '\n';
    }

    virtual ~Conv2D() {}

    void Do() {
        std::cout << "Success Conv!" << '\n';
    }
};

class Maxpool : public Layer {
private:
    /* data */

public:
    Maxpool() {
        std::cout << "MaxPool()" << '\n';
    }

    virtual ~Maxpool() {}

    void Do() {
        std::cout << "Success Max!" << '\n';
    }
};

class Factory {
private:
    /* data */

public:
    Factory() {}

    virtual ~Factory() {}

    static Layer* typefactory(const std::string& type) {
        if ((type == "Conv2D") || (type == "defalt")) return new Conv2D();

        if (type == "Maxpool") return new Maxpool();

        return NULL;
    }
};

class NeuralNetwork {
private:
    int countofLayer = 0;
    int m_noLayer;
    Layer *m_aLayer[];

public:
    NeuralNetwork(int p_noLayer) : m_noLayer(p_noLayer) {
        Alloc();
    }

    virtual ~NeuralNetwork() {}

    bool Alloc() {
        *m_aLayer = new Layer[m_noLayer];

        return true;
    }

    bool CreateLayer(Layer *Type) {
        if (countofLayer >= m_noLayer) {
            std::cout << "already full" << '\n';
            return false;
        }
        // m_aLayer[countofLayer] = Factory::typefactory(type);

        m_aLayer[countofLayer] = Type;

        countofLayer++;

        return true;
    }

    bool Propagate() {
        if (countofLayer < m_noLayer) {
            std::cout << "some value is empty" << '\n';
            return false;
        }

        for (int i = 0; i < m_noLayer; i++) {
            m_aLayer[i]->Do();
        }
        return true;
    }
};


int main(int argc, char const *argv[]) {
    std::cout << "---------------Start---------------" << '\n';

    NeuralNetwork *HGU = new NeuralNetwork(3);

    HGU->CreateLayer(new Conv2D());

    HGU->CreateLayer(new Maxpool());

    HGU->CreateLayer(new Maxpool());

    // HGU->CreateLayer("Maxpool") = new Maxpool(stride,input tensor,c);

    HGU->Propagate();


    std::cout << "---------------Done---------------" << '\n';

    return 0;
}
