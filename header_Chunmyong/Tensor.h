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
    TensorShape m_shape;
    float *m_adata;  // 추후 템플릿으로 수정 예정

public:
    Tensor() {}

    Tensor(int pRank, int *pShape) {}

    virtual ~Tensor() {}

    bool Alloc(int pRank, std::initializer_list<int> pShape) {
        m_shape.Alloc(pRank, pShape);


        return true;
    }

    bool Alloc(int pRank, std::initializer_list<int> pShape, INITIAL_MODE mode) {
        m_shape.Alloc(pRank, pShape);

        if (mode == TRUNCATED_NORMAL) {
            m_adata = Truncated_normal();
        }

        return true;
    }

    float* Truncated_normal() {
        int  flat_dim = 1;
        int  rank     = m_shape.Getrank();
        int *shape    = m_shape.Getshape();

        std::random_device rd;
        std::mt19937 gen(rd());

        for (int i = 0; i < rank; i++) {
            flat_dim *= shape[i];
        }

        float *temp = new float[flat_dim];

        for (int i = 0; i < flat_dim; i++) {
            std::normal_distribution<float> rand(0, 0.6);
            temp[i] = rand(gen);
        }

        return temp;
    }

    void Delete();

    // Initialization(const std::string &type = "default");
};

#endif  // TENSOR_H_
