#ifndef TENSORSHAPE_H_
#define TENSORSHAPE_H_    value

#include <iostream>
#include <array>

class TensorShape {
private:
    // 확인하기
    int m_rank;
    int *m_ashape = new int[5];

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

        List_to_Shape(pShape);

        return true;
    }

    // 추후 파라미터 변화에 대해 유연하게 반응하기 위해서
    void List_to_Shape(std::initializer_list<int> pShape){
        int j = 0;

        for (auto i = pShape.begin(); i != pShape.end(); i++) {
            m_ashape[j] = *i;
            j++;
        }
    }

    int Getrank(){
        return m_rank;
    }

    int *Getshape(){
        return m_ashape;
    }

    bool Delete(){

        return true;
    }

};

#endif  // TENSORSHAPE_H_
