#include "Shape.h"

Shape::Shape() {
    m_Rank = 0;
    m_aDim = NULL;
}

Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    m_Rank = 0;
    m_aDim = NULL;

    Alloc(5, pSize0, pSize1, pSize2, pSize3, pSize4);
}

Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3) {
    m_Rank = 0;
    m_aDim = NULL;

    Alloc(4, pSize0, pSize1, pSize2, pSize3);
}

Shape::Shape(int pSize0, int pSize1, int pSize2) {
    m_Rank = 0;
    m_aDim = NULL;

    Alloc(3, pSize0, pSize1, pSize2);
}

Shape::Shape(int pSize0, int pSize1) {
    m_Rank = 0;
    m_aDim = NULL;

    Alloc(2, pSize0, pSize1);
}

Shape::Shape(int pSize0) {
    m_Rank = 0;
    m_aDim = NULL;

    Alloc(1, pSize0);
}

Shape::Shape(Shape *pShape) {
    std::cout << "Shape::Shape(Shape *)" << '\n';
    Alloc(pShape);
}

Shape::~Shape() {
    std::cout << "Shape::~Shape()" << '\n';
    Delete();
}

int Shape::Alloc() {
    m_Rank = 0;
    m_aDim = NULL;
    return TRUE;
}

int Shape::Alloc(int pRank, ...) {
    try {
        if (pRank > 0) m_Rank = pRank;
        else throw pRank;
    } catch (int e) {
        printf("Receive invalid rank value %d in %s (%s %d)\n", e, __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    try {
        m_aDim = new int[m_Rank];
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    va_list ap;
    va_start(ap, pRank);

    // need to check compare between pRank value and number of another parameter
    for (int i = 0; i < pRank; i++) {
        // need to check whether int or not
        m_aDim[i] = va_arg(ap, int);
    }
    va_end(ap);

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

void Shape::Delete() {
    if (m_aDim) {
        delete[] m_aDim;
        m_aDim = NULL;
    }
}

void Shape::SetRank(int pRank) {
    m_Rank = pRank;
}

int Shape::GetRank() {
    return m_Rank;
}

int& Shape::operator[](int pRanknum) {
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

/////////////////////////////////////////////////// for Print Shape Information
std::ostream& operator<<(std::ostream& pOS, Shape& pShape) {
    int rank = pShape.GetRank();

    pOS << "Rank is " << rank << ", Dimension is [";

    for (int i = 0; i < rank; i++) pOS << pShape[i] << ", ";
    pOS << "]";
    return pOS;
}

// // example code
// int main(int argc, char const *argv[]) {
//     Shape *temp = new Shape(1, 1, 1, 4, 2);
//
//     std::cout << *temp << '\n';
//
//     delete temp;
//
//     return 0;
// }
