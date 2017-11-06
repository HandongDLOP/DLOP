#include <iostream>
#include <typeinfo>
#define quote(x) #x

class Superclass {
private:
    int cheak = 0;
    const char* name;
public:
    Superclass (int pcheck){
        // std::cout << "Superclass" << '\n';

        cheak = pcheck;

        name = __FUNCTION__;

    }
    virtual ~Superclass (){

    }

    void Print(/* arguments */) {
        std::cout << name << '\n';

    }
};

int main(int argc, char const *argv[]) {

    Superclass *A = new Superclass(1);

    A->Print();

    std::cout << typeid(quote()).name() << '\n';

    return 0;
}
