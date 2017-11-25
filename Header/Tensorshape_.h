#ifndef TENSORSHAPE_H_
#define TENSORSHAPE_H_    value

#include <iostream>
#include <array>

class TensorShape {
private:
    // 확인하기
    int m_rank  = 0;
    int *m_aDim = new int[5];

public:
    TensorShape() {}

    TensorShape(TensorShape *pshape) {
        std::cout << "TensorShape::TensorShape(TensorShape *)" << '\n';
        int *temp_dim = pshape->Getdim();

        Alloc(temp_dim[0], temp_dim[1], temp_dim[2], temp_dim[3], temp_dim[4]);
    }

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

        for (m_rank = 0; m_rank < 5; m_rank++) {
            if (m_aDim[m_rank] != 0) {
                continue;
            } else {
                for (int j = m_rank + 1; j < 5; j++) {
                    if (m_aDim[j] != 0) {
                        std::cout << "invalid shape!" << '\n';
                        exit(0);
                    }
                }
            }
            break;
        }

        return true;
    }

    int Getrank() {
        return m_rank;
    }

    int* Getdim() {
        return m_aDim;
    }

    bool Delete() {
        return true;
    }
};

#endif  // TENSORSHAPE_H_
