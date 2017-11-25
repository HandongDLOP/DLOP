#include <iostream>

class X {
private:

    /* data */

public:

    X() {
        std::cout << "X::X()" << '\n';
    }

    X(int arg1) : X(){
        std::cout << "X::X(int)" << '\n';
    }

    virtual ~X(){}
};

int main(int argc, char const *argv[]) {
    X *temp = new X(1);

    delete temp;
    return 0;
}
