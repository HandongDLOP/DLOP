#include <iostream>

class SuperClass {
private:
    int A = 0;
    int B = 1;
    int C = 2;

public:
    SuperClass(int pA, int pB, int pC = 0) {
        A = pA;
        B = pB;
        C = pC;

        Test();
    }

    virtual ~SuperClass() {}

    void SetA(int pA) {}

    void SetB(int pB) {}

    void SetC(int pC) {}

    void Get() {
        std::cout << A << '\n';
        std::cout << B << '\n';
        std::cout << C << '\n';
    }

    virtual void Print() {
        std::cout << "나중 정의" << '\n';
    }

    void Test(){
        std::cout << "Test" << '\n';
        Print();
    }
};


class subClass : public SuperClass {
private:
    /* data */
    int D = 4;

public:
    subClass(int pA, int pB, int pD, int pC = 0) : SuperClass(pA,  pB, pC) {
        D = pD;
        Get();
        std::cout << D << '\n';

        Test();
    }

    virtual ~subClass(){}

    virtual void Print(){
        std::cout << "새정의" << '\n';
    }

};

int main(int argc, char const *argv[]) {
    SuperClass * test = new subClass(1,2,3,3);

    test->Test();

    return 0;
}
