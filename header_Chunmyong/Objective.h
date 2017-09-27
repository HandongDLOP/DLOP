#ifndef OBJECTIVE_FUNCTION_H_
#define OBJECTIVE_FUNCTION_H_    value

#include "Manna.h"

class Objective {
private:
/* data */

public:
Objective() {}

virtual ~Objective() {}

void ComputeDeltaBar(Manna * pDesiredOutput);
};

#endif  // OBJECTIVE_FUNCTION_H_
