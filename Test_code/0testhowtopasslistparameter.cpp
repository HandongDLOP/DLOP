#include <iostream>

// void Phenotype(std::map<int, int> c) {
//     for (int i = 0; i < 2; i++) std::cout << c[i] << '\n';
// }

class X {
public:
    X(std::initializer_list<int> list) {
        for (auto i = list.begin(); i != list.end(); i++) {
            std::cout << *i << std::endl;
        }
    }

    ~X() {}
};


int main(int argc, char const *argv[]) {
    // Operator * HGU = new Operator(conv);

    // Phenotype({ 1, 2 });

    X * x = new X({1,2,3,4,5});

    delete(x);

    return 0;
}
