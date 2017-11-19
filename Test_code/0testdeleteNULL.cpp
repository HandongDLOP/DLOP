#include <iostream>

int main(int argc, char const *argv[]) {
    int * a = NULL;

    std::cout << a << '\n';

    delete a;

    return 0;
}
