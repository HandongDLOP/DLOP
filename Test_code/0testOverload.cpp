#include <iostream>

class Layer {
private:
    /* data */

public:
    Layer (){}
    virtual ~Layer (){}

    void Print(int a){
        std::cout << a << '\n';
    }

    void Print(double a) {
        std::cout << a << '\n';
    }

};

int main(int argc, char const *argv[]) {

    Layer alpha;

    alpha.Print(1);

    alpha.Print(.1);

    return 0;
}
