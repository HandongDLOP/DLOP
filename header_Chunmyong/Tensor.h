#ifndef TENSOR_H_
#define TENSOR_H_

#include <iostream>
#include <random>
#include <array>
#include "Tensorshape.h"

// 추후 Function pointer로 전환 필요
enum INITIAL_MODE {
    ONE,
    RANDOM,
    TRUNCATED_NORMAL
};

// template <class T>

class Tensor {
private:
    TensorShape *m_shape;
    float *m_adata;  // 추후 템플릿으로 수정 예정
    int m_flat_dim = 1;

public:
    Tensor() {}

    Tensor(int pRank, std::initializer_list<int> pShape, INITIAL_MODE mode) {
        std::cout << "Tensor(int, std::initializer_list<int>, INITIAL_MODE)" << '\n';
        Alloc(pRank, pShape, mode);
    }

    virtual ~Tensor() {}

    bool Alloc(int pRank, std::initializer_list<int> pShape) {
        m_shape = new TensorShape(pRank, pShape);

        for (int i = 0; i < m_shape->Getrank(); i++) {
            m_flat_dim *= m_shape->Getshape()[i];
        }

        return true;
    }

    bool Alloc(int pRank, std::initializer_list<int> pShape, INITIAL_MODE mode) {
        std::cout << "Tensor::Alloc(int, std::initializer_list<int>, INITIAL_MODE)" << '\n';
        m_shape = new TensorShape(pRank, pShape);

        for (int i = 0; i < m_shape->Getrank(); i++) {
            m_flat_dim *= m_shape->Getshape()[i];
        }

        if (mode == TRUNCATED_NORMAL) {
            m_adata = Truncated_normal();
        }

        return true;
    }

    float* Truncated_normal() {
        std::cout << "Tensor::Truncated_normal()" << '\n';

        std::random_device rd;
        std::mt19937 gen(rd());

        float *temp = new float[m_flat_dim];

        for (int i = 0; i < m_flat_dim; i++) {
            std::normal_distribution<float> rand(0, 0.6);
            temp[i] = rand(gen);
            // std::cout << temp[i] << ' ';
        }

        return temp;
    }

    void        Delete() {}

    void Setshape(TensorShape * pshape){
        m_shape = pshape;
    }

    void SetData(float * pData){
        m_adata = pData;
    }

    void SetFlatDim(int pflat_dim){
        m_flat_dim = pflat_dim;
    }

    TensorShape * Getshape() {
        return m_shape;
    }

    float* GetData() const {
        return m_adata;
    }

    int GetFlatDim(){
        return m_flat_dim;
    }

    void PrintData(){
        for (int i = 0; i < m_flat_dim; i++) {
            std::cout << m_adata[i] << ' ';
        }
        std::cout << '\n';
    }

    // Initialization(const std::string &type = "default");
};

#endif  // TENSOR_H_
