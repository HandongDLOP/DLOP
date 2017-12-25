#ifndef __DATA__
#define __DATA__      value

#include "Shape.h"

#define SIZEOFCOLS    1024

template<typename DTYPE> class Data {
private:
    int m_Size;
    int m_Cols;  // max column size
    int m_Rows;
    DTYPE **m_aData;

public:
    Data() {
        m_Size  = 0;
        m_Cols  = 0;
        m_Rows  = 0;
        m_aData = NULL;
    }

    Data(unsigned int pSize) {
        std::cout << "Data<DTYPE>::Data(Shape *)" << '\n';
        m_Size  = 0;
        m_Cols  = 0;
        m_Rows  = 0;
        m_aData = NULL;
        Alloc(pSize);
    }

    Data(Data *pData) {
        std::cout << "Data<DTYPE>::Data(Data *)" << '\n';
        m_Size  = 0;
        m_Cols  = 0;
        m_Rows  = 0;
        m_aData = NULL;
        Alloc(pData);
    }

    virtual ~Data() {
        std::cout << "Data<DTYPE>::~Data()" << '\n';
        Delete();
    }

    int    Alloc(unsigned int pSize);
    int    Alloc(Data *pData);
    int    Delete();

    int    GetSize();
    int    GetCols();
    int    GetRows();

    DTYPE& operator[](unsigned int index) {
        return m_aData[index / SIZEOFCOLS][index % SIZEOFCOLS];
    }
};


#endif  // __DATA__
