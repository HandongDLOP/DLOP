#include <iostream>

class Test {
private:
    int test = 0;

public:
    Test() {}

    virtual ~Test(){}

    int operator[](unsigned int idx1){
        return test;
    }
};

int main(int argc, char const *argv[]) {
    return 0;
}
