#ifndef TENSORSHAPE_H_
#define TENSORSHAPE_H_    value

#include <iostream>
#include <array>

class TensorShape {
private:
    // 확인하기
    int m_rank;
    int *m_ashape;

public:
    TensorShape() {}

    // 추후 형태 바꿀 예정 (파라미터를 printf 처럼 받을 수 있게)
    TensorShape(int pRank, std::initializer_list<int> pShape) {
        Alloc(pRank, pShape);
    }

    virtual ~TensorShape() {}

    bool Alloc(int pRank, std::initializer_list<int> pShape) {

        if((unsigned int)pRank != pShape.size()){
            std::cout << "there is size abort!" << '\n';
            exit(0);
        }

        m_rank  = pRank;
        m_ashape = new int[pRank];

        int j = 0;

        for (auto i = pShape.begin(); i != pShape.end(); i++) {
            m_ashape[j] = *i;
        }

        return true;
    }

    int Getrank(){
        return m_rank;
    }

    int *Getshape(){
        int * temp = new int[m_rank];

        // deep copy
        for (int i = 0; i < m_rank; i++){
            temp[i] = m_ashape[i];
        }

        return temp;
    }


};

#endif  // TENSORSHAPE_H_
