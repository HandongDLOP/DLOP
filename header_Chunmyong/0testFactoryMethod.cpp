#include <iostream>
#include <string>


// enum 사용법
enum TYPE { relu, conv, maxpooling };

class MetaParameter {
private:

public:
    MetaParameter() {}

    virtual ~MetaParameter() {}

    virtual bool Alloc() = 0;
};

class Relu : public MetaParameter {
private:
public:
    Relu() {}
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
    Operator(TYPE op) {
        if (op == relu) {
            m_Parameter = new Relu();
        }
    }

    bool SetMetaPrameter(){

        return true;
    }

    virtual ~Operator() {}
};


int main(int argc, char const *argv[]) {


    return 0;
}
