#include "thrud/AssumeRestrictArgs.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/IR/Attributes.h"

using namespace llvm;

extern cl::opt<std::string> KernelNameCL;
extern cl::opt<unsigned int> CoarseningDirectionCL;

AssumeRestrictArgs::AssumeRestrictArgs() : FunctionPass(ID) {}

void AssumeRestrictArgs::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesCFG();
}

bool AssumeRestrictArgs::runOnFunction(Function &function) {
  Function *functionPtr = (Function *)&function;
  //errs() << "Running AssumeRestrictArgs on function " << functionPtr->getName() << "\n";
  
  if (!isKernel(functionPtr))
    return false;

  // Apply the pass to the selected kernel only.
  std::string FunctionName = functionPtr->getName();
  if (KernelNameCL != "" && FunctionName != KernelNameCL)
    return false;

  bool isModified = false;
  for (Function::arg_iterator it = functionPtr->arg_begin(); it != functionPtr->arg_end(); ++it) {
    Argument& arg = *it;
    //errs() << "Arg is : " << arg << " " << arg.hasNoAliasAttr() << " ";
    if (arg.getType()->isPointerTy() && !arg.hasNoAliasAttr()) {
      arg.addAttr(AttributeSet::get(arg.getContext(), AttributeSet::FunctionIndex, Attribute::NoAlias));
      isModified = true;
    }
    //errs() << " now: " << arg.hasNoAliasAttr() << "\n";
  }

  return isModified;
}

char AssumeRestrictArgs::ID = 0;
static RegisterPass<AssumeRestrictArgs> X("assumeRestrictArgs", "Assume Restrict Args Pass - adds NoAlias attr to pointer type function args");
