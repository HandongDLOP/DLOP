#ifndef TENSOR_H_
#define TENSOR_H_

#include <iostream>
#include <random>
#include <array>
#include "Tensorshape.h"

// template <class T>

class Tensor {
private:
    TensorShape *m_ashape = NULL;
    float *m_adata = NULL;  // 추후 템플릿으로 수정 예정
    int m_flat_dim = 1;

public:
    Tensor() {}

    Tensor(int pRank, std::initializer_list<int> pShape) {
        std::cout << "Tensor::Tensor(int, std::initializer_list<int)" << '\n';
        Alloc(pRank, pShape);
    }

    virtual ~Tensor() {
        std::cout << "Tensor::~Tensor()" << '\n';

        // delete를 제대로 하기 위해서는 계속해서 새로운 Tensor를 만들어낼 필요가 있다.
        // 위의 부분이 구현이 되기 전까지는 이 부분은 잠시 free하지 않도록 한다(Testing 단계)
        // Delete();
    }

    bool Alloc(int pRank, std::initializer_list<int> pShape);

    bool Alloc(int pRank, std::initializer_list<int> pShape, float *pData);

    // bool Alloc(int pRank, std::initializer_list<int> pShape, INITIAL_MODE mode);

    bool Delete();

    //===========================================================================================

    static Tensor * Truncated_normal(int pRank, std::initializer_list<int> pShape);

    //===========================================================================================

    void Setshape(TensorShape * pshape){
        m_ashape = pshape;
    }

    void SetData(float * pData){
        m_adata = pData;
    }

    void SetFlatDim(int pflat_dim){
        m_flat_dim = pflat_dim;
    }

    //===========================================================================================

    TensorShape * Getshape() {
        return m_ashape;
    }

    float* GetData() const {
        return m_adata;
    }

    int GetFlatDim(){
        return m_flat_dim;
    }

    //===========================================================================================

    void PrintData(){
        if (m_adata == NULL){
            std::cout << "data is empty!" << '\n';
            exit(0);
        }

        for (int i = 0; i < m_flat_dim; i++) {
            std::cout << m_adata[i] << ' ';
        }
        std::cout << '\n';
    }

    // Initialization(const std::string &type = "default");
};

#endif  // TENSOR_H_
