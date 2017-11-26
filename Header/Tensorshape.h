#ifndef TENSORSHAPE_H_
#define TENSORSHAPE_H_    value

#include <iostream>
#include <array>

class TensorShape {
private:
    // 확인하기
    // int m_rank  = 0;
    int *m_aDim = new int[5];

public:
    TensorShape() {}

    TensorShape(int pDim0, int pDim1, int pDim2, int pDim3, int pDim4) {
        std::cout << "TensorShape::TensorShape(int, int, int, int, int)" << '\n';
        Alloc(pDim0, pDim1, pDim2, pDim3, pDim4);
    }

    virtual ~TensorShape() {}

    bool Alloc(int pDim0, int pDim1, int pDim2, int pDim3, int pDim4) {
        m_aDim = new int[5];

        m_aDim[0] = pDim0;
        m_aDim[1] = pDim1;
        m_aDim[2] = pDim2;
        m_aDim[3] = pDim3;
        m_aDim[4] = pDim4;

        // dimension 확인은 나중에 진행한다.

        // for (m_rank = 0; m_rank < 5; m_rank++) {
        //     if (m_aDim[m_rank] == 0) {
        //         std::cout << "invalid dimension!" << '\n';
        //         exit(0);
        //     }
        //     break;
        // }

        return true;
    }

    int* GetDim() {
        return m_aDim;
    }

    bool Delete() {
        return true;
    }
};

#endif  // TENSORSHAPE_H_
