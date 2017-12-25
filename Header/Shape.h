#ifndef __SHAPE__
#define __SHAPE__    value

#include <iostream>
#include <stdexcept>
#include <exception>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#ifndef TRUE
    # define TRUE     1
    # define FALSE    0
#endif  // !TRUE

class Shape {
private:
    int m_Rank;
    int *m_aDim;

public:
    Shape() {
        m_Rank = 0;
        m_aDim = NULL;
    }

    Shape(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) ;
    Shape(int pBatchSize, int pChannelSize, int pRowSize, int pColSize);
    Shape(int pChannelSize, int pRowSize, int pColSize);
    Shape(int pRowSize, int pColSize);
    Shape(int pColSize);

    /*
     * Shape(int pRank, ...) {
     *  std::cout << "Shape::Shape(int, ...)" << '\n';
     *  m_Rank = 0;
     *  m_aDim = NULL;
     *
     *  // for Variable Argument
     *  va_list ap;
     *  va_start(ap, pRank);
     *
     *  Alloc(pRank, ap);
     *
     *  va_end(ap);
     * }
     */

    Shape(Shape *pShape) {
        std::cout << "Shape::Shape(Shape *)" << '\n';
        Alloc(pShape);
    }

    virtual ~Shape() {
        std::cout << "Shape::~Shape()" << '\n';
        Delete();
    }

    int  Alloc();
    int  Alloc(int pRank, int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize);
    /*int  Alloc(int pRank, va_list ap);*/
    int  Alloc(Shape *pShape);
    int  Delete();

    void SetRank(int pRank);
    int  GetRank();

    int& operator[](int pRanknum) {
        try {
            if (pRanknum >= 0) return m_aDim[pRanknum];
            else throw;
        }
        catch (...) {
            printf("Receive invalid pRanknum value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            exit(0);
            // return FALSE;
        }
    }
};


#endif  // __SHAPE__
