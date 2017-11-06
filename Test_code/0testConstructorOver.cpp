#include <iostream>

class Operator {
private:
    /* data */

    void inner(){
        std::cout << "inner" << '\n';
    }

public:
    Operator() {
        std::cout << "Operator" << '\n';
    }

    virtual ~Operator() {}

    virtual void Run() = 0;

    void Run2() {
        inner();
    }
};

class Relu : public Operator {
private:
    /* data */

public:
    Relu() : Operator() {
        std::cout << "Relu" << '\n';
    }

    virtual ~Relu() {}

    void Run() {
        std::cout << "Run from Relu" << '\n';
        Run2();
    }
};

int main(int argc, char const *argv[]) {
    Operator *HGUOP = new Relu();

    HGUOP->Run();

    delete HGUOP;

    return 0;
}
