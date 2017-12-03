#include <iostream>

template<typename T>
class Test {
private:
    int data;

public:
    Test() {}

    virtual ~Test() {}
};


template<typename T>
class Super {
private:
    T data = 0;

public:
    Super(T param) {
        std::cout << "Super" << '\n';
        data = param;
    }

    virtual ~Super() {}

    T GetData() {
        return data;
    }
};


template<typename T>
class Sub : public Super<T>{
private:
    /* data */

public:
    Sub(T param) : Super<T>(param) {
        std::cout << "Sub" << '\n';
    }

    virtual ~Sub() {}
};

int main(int argc, char const *argv[]) {
    int data = 5;

    Super<int *> *temp = new Sub<int *>(&data);

    int testdata = *temp->GetData();

    std::cout << testdata << '\n';

    delete temp;

    return 0;
}
