#include "Shape.h"

Shape::Shape(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize){
    m_Rank = 0;
    m_aDim = NULL;

    Alloc(5, pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);
}

Shape::Shape(int pBatchSize, int pChannelSize, int pRowSize, int pColSize){
    m_Rank = 0;
    m_aDim = NULL;

    Alloc(4, 1, pBatchSize, pChannelSize, pRowSize, pColSize);
}

Shape::Shape(int pChannelSize, int pRowSize, int pColSize){
    m_Rank = 0;
    m_aDim = NULL;

    Alloc(3, 1, 1, pChannelSize, pRowSize, pColSize);
}

Shape::Shape(int pRowSize, int pColSize){
    m_Rank = 0;
    m_aDim = NULL;

    Alloc(2, 1, 1, 1, pRowSize, pColSize);
}

Shape::Shape(int pColSize){
    m_Rank = 0;
    m_aDim = NULL;

    Alloc(1, 1, 1, 1, 1, pColSize);
}

int Shape::Alloc() {
    m_Rank = 0;
    m_aDim = NULL;
    return TRUE;
}
/*
 * int Shape::Alloc(int pRank, va_list ap) {
 *  try {
 *      if (pRank == 0) m_Rank = 1;
 *      else if (pRank > 0) m_Rank = pRank;
 *      else throw pRank;
 *  } catch (int e) {
 *      printf("Receive invalid rank value %d in %s (%s %d)\n", e, __FUNCTION__, __FILE__, __LINE__);
 *      return FALSE;
 *  }
 *
 *  try {
 *      m_aDim = new int[m_Rank];
 *  } catch (...) {
 *      printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
 *      return FALSE;
 *  }
 *
 *  if (pRank == 0) {
 *      m_aDim[pRank] = 1;
 *  } else {
 *      // need to check compare between pRank value and number of another parameter
 *      for (int i = 0; i < pRank; i++) {
 *          // need to check whether int or not
 *          m_aDim[i] = va_arg(ap, int);
 *      }
 *  }
 *
 *  return TRUE;
 * }
 */
int Shape::Alloc(int pRank, int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
    try {
        if (pRank > 0) m_Rank = pRank;
        else throw pRank;
    } catch (int e) {
        printf("Receive invalid rank value %d in %s (%s %d)\n", e, __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    try {
        m_aDim = new int[5];
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    m_aDim[0] = pColSize;
    m_aDim[1] = pRowSize;
    m_aDim[2] = pChannelSize;
    m_aDim[3] = pBatchSize;
    m_aDim[4] = pTimeSize;

    return TRUE;
}

int Shape::Alloc(Shape *pShape) {
    int pRank = pShape->GetRank();

    try {
        m_aDim = new int[5];
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    for (int i = 0; i < pRank; i++) {
        // need to check whether int or not
        m_aDim[i] = (*pShape)[i];
    }

    return TRUE;
}

int Shape::Delete() {
    try {
        delete[] m_aDim;
    } catch (...) {
        printf("Failed to deallocate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }
    return TRUE;
}

void Shape::SetRank(int pRank) {
    m_Rank = pRank;
}

int Shape::GetRank() {
    return m_Rank;
}

// for Print Shape Information
std::ostream& operator<<(std::ostream& pOS, Shape& pShape) {
    int rank = pShape.GetRank();

    pOS << "Rank is " << rank << ", Dimension is [";

    for (int i = 0; i < rank; i++) pOS << pShape[i] << ", ";

    pOS << "]";

    return pOS;
}

// example code
// int main(int argc, char const *argv[]) {
// Shape *temp = new Shape(3, 1, 3, 4);
//
//// (*temp)[2] = 4;
//
// std::cout << *temp << '\n';
//
// delete temp;
//
// return 0;
// }
