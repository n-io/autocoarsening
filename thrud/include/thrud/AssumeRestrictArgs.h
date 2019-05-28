#ifndef ASSUME_RESTRICT_ARGS_H
#define ASSUME_RESTRICT_ARGS_H

#include "thrud/Utils.h"

#include "llvm/Pass.h"

using namespace llvm;

namespace llvm {
class Function;
}

class AssumeRestrictArgs : public FunctionPass {

public:
  static char ID;
  AssumeRestrictArgs();

  virtual bool runOnFunction(Function &function);
  virtual void getAnalysisUsage(AnalysisUsage &au) const;
};

#endif
