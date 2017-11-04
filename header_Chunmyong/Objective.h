#ifndef OBJECTIVE_FUNCTION_H_
#define OBJECTIVE_FUNCTION_H_    value

#include "Tensor.h"

class Objective {
private:
/* data */

public:
Objective() {}

virtual ~Objective() {}

void ComputeDeltaBar(Tensor * pDesiredOutput);
};

#endif  // OBJECTIVE_FUNCTION_H_
