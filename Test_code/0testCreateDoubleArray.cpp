#include <iostream>

class Tensor {
private:
    double **m_aData = NULL;
    int m_Shape      = 0;

public:
    Tensor(double *pData, int pShape) {
        std::cout << "Tensor::Tensor(double *)" << '\n';

        m_aData    = new double *[1];
        m_aData[1] = pData;
        m_Shape    = pShape;
    }

    virtual ~Tensor() {
        std::cout << "Tensor::~Tensor()" << '\n';

        delete[] m_aData[1];

        delete[] m_aData;
    }

    void PrintData() {
        for (int i = 0; i < m_Shape; i++) {
            std::cout << m_aData[1][i] << ", ";
        }
        std::cout << '\n';
    }
};

Tensor* CreateTensor() {
    // double temp_[] = { 2, 4, 5, 6 };
    double *temp = new double[4]{ 2, 4, 5, 6 };
    //
    // temp[0] = 2;
    // temp[1] = 3;
    // temp[2] = 4;
    // temp[3] = 5;

    return new Tensor(temp, 4);
}

int main(int argc, char const *argv[]) {
    Tensor *tensor = CreateTensor();

    tensor->PrintData();

    delete tensor;

    return 0;
}
