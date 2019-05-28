#include "thrud/ReplaceGlobalIds.h"

#include "thrud/NDRange.h"

#include "llvm/Support/CommandLine.h"

using namespace llvm;

extern cl::opt<std::string> KernelNameCL;
extern cl::opt<unsigned int> CoarseningDirectionCL;

ReplaceGlobalIds::ReplaceGlobalIds() : FunctionPass(ID) {}

void ReplaceGlobalIds::getAnalysisUsage(AnalysisUsage &au) const {
  // todo
  au.addRequired<NDRange>();
  au.setPreservesCFG();
}

bool ReplaceGlobalIds::runOnFunction(Function &function) {
  Function *functionPtr = (Function *)&function;
  //errs() << "Running ReplaceGlobalIds on function " << functionPtr->getName() << "\n";
  //return false;
  
  if (!isKernel(functionPtr))
    return false;

  // Apply the pass to the selected kernel only.
  std::string FunctionName = functionPtr->getName();
  if (KernelNameCL != "" && FunctionName != KernelNameCL)
    return false;

  NDRange *ndr = &getAnalysis<NDRange>();
  unsigned int direction = CoarseningDirectionCL;
  InstVector gids = ndr->getGids(direction);

  std::vector<Instruction *> unusedInsts;
  for (InstVector::iterator instIter = gids.begin(), instEnd = gids.end();
       instIter != instEnd; ++instIter) {
    Instruction *inst = *instIter;

    LLVMContext & context = inst->getContext();
    IntegerType * intType = IntegerType::getInt32Ty(context);

    CallInst *cinst = (CallInst *) inst;

    ConstantInt *cint = dyn_cast<ConstantInt>(cinst->getArgOperand(0));

    CallInst *groupId = (CallInst* ) inst->clone();

    groupId->setCalledFunction(ndr->getOclFunctionPtr(NDRange::GET_GROUP_ID));
    groupId->insertAfter(inst);
    //ndr->registerOclInst(direction, NDRange::GET_GROUP_ID, groupId);

    CallInst *localSize = (CallInst* ) inst->clone();
    localSize->setCalledFunction(ndr->getOclFunctionPtr(NDRange::GET_LOCAL_SIZE));
    localSize->insertAfter(groupId);
    //ndr->registerOclInst(direction, NDRange::GET_LOCAL_SIZE, localSize);

    CallInst *localId = (CallInst* ) inst->clone();
    localId->setCalledFunction(ndr->getOclFunctionPtr(NDRange::GET_LOCAL_ID));
    localId->insertAfter(localSize);
    //ndr->registerOclInst(direction, NDRange::GET_LOCAL_ID, localId);

    Instruction *mul = getMulInst(groupId, localSize);
    mul->insertAfter(localId);
    Instruction *add = getAddInst(mul, localId);
    add->insertAfter(mul);

    replaceUses(inst, add);
    
    //ndr->unregisterOclInst(direction, NDRange::GET_GLOBAL_ID, inst);
    //errs () << "Trying to remove inst from parent >> " << *inst << "\n";
    unusedInsts.push_back(inst); 
    //inst->removeFromParent();
  }
  for (auto it = unusedInsts.begin(); it != unusedInsts.end(); ++it) {
    Instruction *inst = *it;
    inst->eraseFromParent();
  }

  return unusedInsts.size() > 0;
  //return false;
}

char ReplaceGlobalIds::ID = 0;
static RegisterPass<ReplaceGlobalIds> X("replaceGIDs", "Replaces use of get_global_id with get_group_id * get_local_size + get_local_id"); // todo
