#include <iostream>
#include <algorithm>

class Test {
private:
    /* data */

public:
    Test (){}
    virtual ~Test (){}
};

int main(int argc, char const *argv[]) {

    Test ** Array = new Test*[4];

    Test * node_1 = new Test();

    Test * node_2 = new Test();

    Test * node_3 = new Test();

    Test * node_4 = new Test();

    Array[0] = node_1;
    Array[1] = node_2;

    return 0;
}
