#ifndef OCCUPANCY_REDUCTION_H
#define OCCUPANCY_REDUCTION_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstIterator.h"

#include "thrud/Utils.h"

using namespace llvm;

class OccupancyReduction : public FunctionPass {

  public:
    static char ID;
    const static unsigned int LOCAL_ADDRESS_SPACE = 3;
    
    OccupancyReduction() : FunctionPass(ID) {}
    virtual void getAnalysisUsage(AnalysisUsage &au) const;
    virtual bool runOnFunction(Function &F);
    //virtual bool doFinalization(Module &M);

  };

  #endif
