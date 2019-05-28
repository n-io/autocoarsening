#ifndef REPLACE_GIDS_H
#define REPLACE_GIDS_H

#include "thrud/Utils.h"

#include "llvm/Pass.h"

using namespace llvm;

namespace llvm {
class Function;
}

class ReplaceGlobalIds : public FunctionPass {

public:
  static char ID;
  ReplaceGlobalIds();

  virtual bool runOnFunction(Function &function);
  virtual void getAnalysisUsage(AnalysisUsage &au) const;
};

#endif
