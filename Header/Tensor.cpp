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
    }

    temp_Tensor->SetData(temp_data);

    return temp_Tensor;
}

Tensor * Tensor::Zero(int pRank, std::initializer_list<int> pShape){
    std::cout << "Tensor::Zero()" << '\n';

    Tensor * temp_Tensor = new Tensor(pRank, pShape);
    int flat_dim = temp_Tensor->GetFlatDim();
    float *temp_data = new float[flat_dim];

    temp_Tensor->SetData(temp_data);

    return temp_Tensor;
}

Tensor * Tensor::Constant(int pRank, std::initializer_list<int> pShape, float constant){
    std::cout << "Tensor::Constant()" << '\n';

    Tensor * temp_Tensor = new Tensor(pRank, pShape);
    int flat_dim = temp_Tensor->GetFlatDim();
    float *temp_data = new float[flat_dim];

    for (int i = 0; i < flat_dim; i++) {
        temp_data[i] = constant;
    }

    temp_Tensor->SetData(temp_data);

    return temp_Tensor;

}

void Tensor::PrintData(){
    if (m_adata == NULL) {
        std::cout << "data is empty!" << '\n';
        exit(0);
    }

    /*
     * 알고리즘 아이디어 1 :
     * rank[1] = rankmul1
     * rank[1] * rank[2] = rankmul2
     * ...
     * rank[1] * rank[2] * rank[3] * rank[4] = rankmul4
     * 각자 변수 지정 후 사용
     *
     * 알고리즘 아이디어 1_2 :
     * 위에서 구한 rankmulN이 0인 경우를 체크해서 사용
     */

    int *rank = m_ashape->Getshape();

    if (rank[0] != 0) {
        std::cout << "[ ";

        for (int r0 = 0; r0 < rank[0]; r0++) {
            if (rank[1] != 0) {
                std::cout << "[ ";

                for (int r1 = 0; r1 < rank[1]; r1++) {
                    if (rank[2] != 0) {
                        std::cout << "[ ";

                        for (int r2 = 0; r2 < rank[2]; r2++) {
                            if (rank[3] != 0) {
                                std::cout << "[ ";

                                for (int r3 = 0; r3 < rank[3]; r3++) {
                                    if (rank[4] != 0) {
                                        std::cout << "[ ";

                                        for (int r4 = 0; r4 < rank[4]; r4++) {
                                            std::cout << m_adata[rank[1] * rank[2] * rank[3] * rank[4] * r0 + rank[2] * rank[3] * rank[4] * r1 + rank[3] * rank[4] * r2 + rank[4] * r3 + r4] << " ";
                                        }

                                        std::cout << "]";
                                    } else std::cout << m_adata[rank[1] * rank[2] * rank[3] * r0 + rank[2] * rank[3] * r1 + rank[3] * r2 + r3] << " ";
                                }

                                std::cout << "]";
                            } else std::cout << m_adata[rank[1] * rank[2] * r0 + rank[2] * r1 + r2] << " ";
                        }

                        std::cout << "]";
                    } else std::cout << m_adata[rank[1] * r0 + r1] << " ";
                }

                std::cout << "]";
            } else std::cout << m_adata[r0] << " ";
        }

        std::cout << "]";
    } else std::cout << m_adata[0] << '\n';

    std::cout << '\n';
}
