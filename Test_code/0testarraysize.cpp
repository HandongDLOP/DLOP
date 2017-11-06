#include <iostream>

int main(int argc, char const *argv[]) {

    int * a = new int[4];

    int size = sizeof(a) / sizeof(a[0]);

    std::cout << size << '\n';

    return 0;
}
