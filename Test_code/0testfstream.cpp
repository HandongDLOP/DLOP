#include <iostream>
#include <fstream>

int main(int argc, char const *argv[]) {
    std::ifstream ifs("test.txt", std::ifstream::binary);
    char c = ifs.get();

    std::cout << ifs.good() << '\n';

    while (ifs.good()) {
        std::cout << c;
        c = ifs.get();
    }

    ifs.close();

    return 0;
}
