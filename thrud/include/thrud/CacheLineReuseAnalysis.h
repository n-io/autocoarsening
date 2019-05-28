#ifndef CACHE_LINE_REUSE_ANALYSIS_H
#define CACHE_LINE_REUSE_ANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstIterator.h"

#include <set>
#include <vector>

#include "thrud/MemAccessDescriptor.h"
#include "thrud/NDRange.h"
#include "thrud/Utils.h"

using namespace llvm;

class CacheLineReuseAnalysis : public FunctionPass {

  public:
    static char ID;
    const static unsigned int GLOBAL_ADDRESS_SPACE = 1;
    
    CacheLineReuseAnalysis() : FunctionPass(ID) {}
    //~CacheLineReuseAnalysis();
    virtual void getAnalysisUsage(AnalysisUsage &au) const;
    virtual bool runOnFunction(Function &F);
    //virtual bool doFinalization(Module &M);

  private:
    NDRange *ndr;
    LoopInfo *loopInfo;
    int dimensions;
    Instruction* lastInstruction;
    //std::vector<BasicBlock::Iterator> loopStack;
    std::set<Instruction*> memops;
    std::set<Instruction*> relevantInstructions;
    std::set<Loop *> relevantLoops;
    std::vector<std::map<Instruction*, std::vector<MemAccessDescriptor>>> accessDescriptorStack;
    std::map<StringRef, std::set<int>> accessedCacheLines;
    std::string diagnosis;

    int getDimensionality();
    inst_iterator simulate(inst_iterator inst, Instruction* fwdDef, Loop* innermostLoop);
    PHINode *getInductionVariable(Loop* loop) const;
    void preprocess(Function *function, std::set<Instruction*>& memops, std::set<Instruction*>& relevantInstructions);
    Value * getAccessedSymbolPtr(Value * v);
    StringRef getAccessedSymbolName(Value * v);
    bool isCachedAddressSpace(Instruction * inst);
    std::vector<MemAccessDescriptor> findInStack(Instruction* inst);
    void mergeIntoStack(std::map<Instruction*, std::vector<MemAccessDescriptor>> &defs);
    bool isFwdDef(Instruction* inst);
    std::vector<MemAccessDescriptor> getOperand(Value * v);
    void applyBinaryOp(function<int(int, int)> f, Instruction * inst);

    inline void addToStack(Instruction* inst, std::vector<MemAccessDescriptor> mad) {
      accessDescriptorStack.back().insert(std::pair<Instruction*, std::vector<MemAccessDescriptor>>(inst, mad));
    }
    inline void addToStack(Instruction* inst, MemAccessDescriptor mad) {
      addToStack(inst, std::vector<MemAccessDescriptor>{mad});
    }
    

};

#endif
