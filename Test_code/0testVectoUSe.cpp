#include <iostream>
#include <vector>

using namespace std;

class Temp {
private:
    vector<int> data;

public:
    Temp (int n){
        data.resize(n);
    }
    virtual ~Temp (){

    }

    vector<int> GetData(){
        return data;
    }
};


int main(int argc, char const *argv[]) {
    Temp test(12);

    vector<int> data = test.GetData();

    std::cout << data[1] << '\n';
    return 0;
}
