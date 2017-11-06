#include <iostream>

class Test {
private:
    int currentIndex = 0;
    int maxIndex     = 0;
    int **Address;

public:
    Test(int index) {
        maxIndex = index;

        Address =  new int*[maxIndex];
    }

    virtual ~Test() {}

    bool PutAddress(int datum) {
        if(Address[currentIndex] != NULL){
            std::cout << "Already there" << '\n';
            return false;
        }

        Address[currentIndex] = new int(datum);
        currentIndex++;

        return true;
    }

    void Print() {
        for (int i = 0; i < currentIndex; i++) {
            std::cout << *Address[i] << '\n';
        }
    }
};


int main(int argc, char const *argv[]) {
    Test *HGU = new Test(3);

    HGU->PutAddress(1);

    HGU->PutAddress(2);

    HGU->PutAddress(3);

    HGU->PutAddress(4);

    HGU->Print();

    return 0;
}
