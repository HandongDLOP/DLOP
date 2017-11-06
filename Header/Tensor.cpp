#include "Tensor.h"

bool Tensor::Alloc(int pRank, std::initializer_list<int> pShape) {
    m_ashape = new TensorShape(pRank, pShape);

    for (int i = 0; i < m_ashape->Getrank(); i++) {
        m_flat_dim *= m_ashape->Getshape()[i];
    }

    return true;
}

bool Tensor::Alloc(int pRank, std::initializer_list<int> pShape, float *pData){
    m_ashape = new TensorShape(pRank, pShape);

    m_adata = pData;

    for (int i = 0; i < m_ashape->Getrank(); i++) {
        m_flat_dim *= m_ashape->Getshape()[i];
    }

    return true;
}

bool Tensor::Delete() {
    std::cout << "Tensor::Delete()" << '\n';
    delete m_ashape;
    delete m_adata;
    return true;
}

//===========================================================================================

Tensor * Tensor::Truncated_normal(int pRank, std::initializer_list<int> pShape){
    std::cout << "Tensor::Truncated_normal()" << '\n';

    // 추후 교수님이 주신 코드를 참고해서 바꿀 것
    std::random_device rd;
    std::mt19937 gen(rd());

    Tensor * temp_Tensor = new Tensor(pRank, pShape);
    int flat_dim = temp_Tensor->GetFlatDim();
    float *temp_data = new float[flat_dim];

    for (int i = 0; i < flat_dim; i++) {
        std::normal_distribution<float> rand(0, 0.6);
        temp_data[i] = rand(gen);
        // std::cout << temp_data[i] << ' ';
    }

    temp_Tensor->SetData(temp_data);

    return temp_Tensor; // 임시
}
