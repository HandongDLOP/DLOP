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
    Shape();
    Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4);
    Shape(int pSize0, int pSize1, int pSize2, int pSize3);
    Shape(int pSize0, int pSize1, int pSize2);
    Shape(int pSize0, int pSize1);
    Shape(int pSize0);
    Shape(Shape *pShape);
    virtual ~Shape();

    int  Alloc();
    int  Alloc(int pRank, ...);
    int  Alloc(Shape *pShape);
    void Delete();

    void SetRank(int pRank);
    int  GetRank();

    int& operator[](int pRanknum);
};


#endif  // __SHAPE__
