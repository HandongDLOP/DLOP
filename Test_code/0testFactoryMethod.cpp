#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <map>

// enum 사용법
enum TYPE { relu, conv, maxpooling };

class MetaParameter {
private:

public:
    MetaParameter() {}

    virtual ~MetaParameter() {}

    static void Test() {}

    virtual bool Alloc() = 0;
};

class Relu : public MetaParameter {
private:
public:
    Relu() {
        Alloc();
    }
    virtual ~Relu() {}

    bool Alloc(){
        std::cout << "Relu metaparameter" << '\n';
        return true;
    }
};


class Operator {
private:
    MetaParameter *m_Parameter;

public:
    Operator(TYPE op, int a = 0) {
        if (op == relu) {
            m_Parameter = new Relu();
        }

        // for(unsigned int i = 0; i < 4; i++){
        //     std::cout << int_list[i] << '\n';
        // }
    }

    bool SetMetaPrameter(){

        return true;
    }

    virtual ~Operator() {}
};



int main(int argc, char const *argv[]) {


    return 0;
}
