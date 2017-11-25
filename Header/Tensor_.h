#ifndef TENSOR_H_
#define TENSOR_H_

#include <iostream>
#include <random>
#include "Tensorshape.h"

// template <class T>

class Tensor {
private:
    // 현재는 scala 값은 따로 존재하지 않고 rnak0 dimension 1로 취급한다.
    TensorShape *m_ashape = NULL;
    float *m_adata        = NULL; // 추후 템플릿으로 수정 예정
    int m_flat_dim        = 1;

public:
    Tensor() {
        std::cout << "Tensor::Tensor()" << '\n';
    }

    Tensor(TensorShape *pshape) {
        std::cout << "Tensor::Tensor(TensorSahpe *)" << '\n';
        m_ashape = new TensorShape(pshape);

        int *temp_dim = m_ashape->Getdim();

        Alloc(temp_dim[0], temp_dim[1], temp_dim[2], temp_dim[3], temp_dim[4]);
    }

    Tensor(int pDim0, int pDim1, int pDim2, int pDim3, int pDim4) {
        std::cout << "Tensor::Tensor(int, int, int, int, int)" << '\n';
        Alloc(pDim0, pDim1, pDim2, pDim3, pDim4);
    }

    Tensor(int pDim0, int pDim1, int pDim2, int pDim3, int pDim4, float *pData) {
        std::cout << "Tensor::Tensor(int, int, int, int, int)" << '\n';
        Alloc(pDim0, pDim1, pDim2, pDim3, pDim4);
        SetData(pData);
    }

    virtual ~Tensor() {
        std::cout << "Tensor::~Tensor()" << '\n';

        // delete를 제대로 하기 위해서는 계속해서 새로운 Tensor를 만들어낼 필요가 있다.
        // 추후 Delete를 고려해서 리펙토링 할 것
        Delete();
    }

    bool Alloc(int pDim0, int pDim1, int pDim2, int pDim3, int pDim4);

    // bool Alloc(int pRank, std::initializer_list<int> pDim, INITIAL_MODE mode);

    bool Delete();

    // ===========================================================================================

    static Tensor* Truncated_normal(int pDim0, int pDim1, int pDim2, int pDim3, int pDim4, float mean, float stddev);


    static Tensor* Zeros(int pDim0, int pDim1, int pDim2, int pDim3, int pDim4);


    static Tensor* Constants(int pDim0, int pDim1, int pDim2, int pDim3, int pDim4, float constant);


    // ===========================================================================================

    void Reshape(int pDim0, int pDim1, int pDim2, int pDim3, int pDim4) {
        TensorShape *temp = new TensorShape(pDim0, pDim1, pDim2, pDim3, pDim4);

        int temp_flat_dim = 1;

        for (int i = 0; i < temp->Getrank(); i++) {
            temp_flat_dim *= temp->Getdim()[i];
        }

        if (temp_flat_dim != m_flat_dim) {
            std::cout << "invalid shape transformation" << '\n';
            delete temp;
            exit(0);
        }

        if (m_ashape != NULL) delete m_ashape;

        m_ashape = temp;
    }

    void Reshape(TensorShape *pshape) {
        // 추후 주소를 절대 바꾸지 않도록 바꾸어야 한다.
        if (m_ashape != NULL) delete m_ashape;
        m_ashape = pshape;
    }

    void SetData(float *pData) {
        // pData 의 크기와 dimension 크기가 일치하는지 확인

        // for (int i = 0; i < m_flat_dim; i++) {
        //     m_adata[i] = pData[i];
        // }

        // delete pData;

        if(m_adata != NULL) delete m_adata;
        m_adata = pData;
    }

    void SetFlatDim(int pflat_dim) {
        m_flat_dim = pflat_dim;
    }

    void SetTensor(Tensor *pTensor) {
        int *temp_dim = pTensor->Getshape()->Getdim();

        Alloc(temp_dim[0], temp_dim[1], temp_dim[2], temp_dim[3], temp_dim[4]);

        if (m_adata != NULL) delete m_adata;
        m_adata = new float[m_flat_dim];

        for (int i = 0; i < m_flat_dim; i++) {
            m_adata[i] = pTensor->GetData()[i];
        }
    }

    void SetTensor(TensorShape *pshape) {
        int *temp_dim = pshape->Getdim();

        Alloc(temp_dim[0], temp_dim[1], temp_dim[2], temp_dim[3], temp_dim[4]);

        if (m_adata != NULL) delete m_adata;
        m_adata = new float[m_flat_dim];
    }

    // ===========================================================================================

    TensorShape* Getshape() const {
        return m_ashape;
    }

    int Getrank() const {
        return m_ashape->Getrank();
    }

    int* Getdim() const {
        return m_ashape->Getdim();
    }

    float* GetData() const {
        return m_adata;
    }

    int GetFlatDim() const {
        return m_flat_dim;
    }

    // ===========================================================================================


    void Flat() {
        Reshape(m_flat_dim, 0, 0, 0, 0);
    }

    // ===========================================================================================

    Tensor * Copyto(Tensor *pTensor) {
        std::cout << "Tensor::Copyto(Tensor *)" << '\n';

        Tensor *temp_Tensor = new Tensor(pTensor->Getshape());

        int *temp_dim = pTensor->Getshape()->Getdim();

        Alloc(temp_dim[0], temp_dim[1], temp_dim[2], temp_dim[3], temp_dim[4]);

        for (int i = 0; i < m_flat_dim; i++) {
            temp_Tensor->GetData()[i] = pTensor->GetData()[i];
        }

        return temp_Tensor;
    }


    void PrintData();

    // Initialization(const std::string &type = "default");
};

#endif  // TENSOR_H_
