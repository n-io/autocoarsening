#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <map>
#include <list>
#include <functional>
#include <algorithm>

#include "thrud/OccupancyReduction.h"
#include "thrud/NDRange.h"
#include "thrud/Utils.h"

using namespace llvm;

extern cl::opt<std::string> KernelNameCL;
cl::opt<unsigned int> SharedMemBytes("shmem", cl::init(0), cl::Hidden, cl::desc("The amount of redundant shared memory reserved in bytes"));

void OccupancyReduction::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<NDRange>();
}

bool OccupancyReduction::runOnFunction(Function &F) {
  Function *function = (Function *)&F;
  if (!isKernel(function))
    return false;

  // Apply the pass to the selected kernel only.
  std::string FunctionName = F.getName();
  if (KernelNameCL != "" && FunctionName != KernelNameCL || SharedMemBytes == 0)
    return false;

  Module &module = *(function->getParent());
  //std::string gvId = FunctionName + "..ored";
  //module.getOrInsertGlobal(gvId, ArrayType::get(Type::getInt8Ty(F.getContext()), SharedMemBytes));
  //GlobalVariable *gv = module.getNamedGlobal(gvId);
  
  ArrayType *arrayTy = ArrayType::get(IntegerType::get(module.getContext(), 8), SharedMemBytes);

  GlobalVariable *gv = new GlobalVariable(/*Module=*/module,
                                          /*Type=*/arrayTy,
					  /*isConstant=*/false,
					  /*Linkage=*/GlobalValue::LinkageTypes::InternalLinkage,
					  /*Initializer=*/ConstantAggregateZero::get(arrayTy),
					  /*Name=*/FunctionName + "..ored",
					  /*InsertBefore=*/nullptr,
					  /*ThreadLocalMode=*/GlobalValue::ThreadLocalMode::NotThreadLocal,
					  /*AddressSpace=*/LOCAL_ADDRESS_SPACE);

  gv->setAlignment(1);

  //BasicBlock *bb = new BasicBlock("oredblock", nullptr);
  BasicBlock *entry = (BasicBlock *)&function->front();
  BasicBlock *oredMemAccess = BasicBlock::Create(F.getContext(), "ored..memaccess", function, &function->front());
  BasicBlock *oredEntry = BasicBlock::Create(F.getContext(), "ored.." + entry->getName(), function, &function->front());

  IRBuilder<> eBuilder(oredEntry);
  IRBuilder<> maBuilder(oredMemAccess);

  // ---- ENTRY BLOCK
  // if (get_local_size(0) == MAX_INT) { jump to dummy mem access bb }

  NDRange *ndr = &getAnalysis<NDRange>();
  Function *oclF = ndr->getOclFunctionPtr(NDRange::GET_LOCAL_SIZE);
  CallInst *callInst = eBuilder.CreateCall(oclF, eBuilder.getInt32(0));
  //std::vector<Constant*> args;
  //args.pushBack(ConstantInt::get(F.getContext(), APInt::getNullValue(8)));
  //CallInst *callInst = eBuilder.CreateCall(oclF, args);

  Value *icmp = eBuilder.CreateICmpEQ(callInst, ConstantInt::get(F.getContext(), APInt::getSignedMaxValue(32)));

  eBuilder.CreateCondBr(icmp, oredMemAccess, entry);


  // ---- MEM ACCESS BLOCK
  // create store inst:
  // store volatile i8 0, i8 addrspace(3)* getelementptr inbounds ([6144 x i8] addrspace(3)* @mykernel..ored, i32 0, i32 0), align 1

  std::vector<Constant*> const_ptr_indices;
  const_ptr_indices.push_back(ConstantInt::get(F.getContext(), APInt::getNullValue(8)));
  const_ptr_indices.push_back(ConstantInt::get(F.getContext(), APInt::getNullValue(8)));
  Constant *gep = ConstantExpr::getInBoundsGetElementPtr(gv, const_ptr_indices);
  maBuilder.CreateStore(maBuilder.getInt8(0), gep, true)->setAlignment(1);

  maBuilder.CreateBr(entry);

  //const_ptr_indices.push_back(ConstantInt::get(F.getContext(), APInt(8, StringRef("0"), 10)));
  //Value *gep = builder.CreateConstGEP1_32(arrayTy, gv, 0);
  //Value *gep = builder.CreateInBoundsGEP(gv, builder.getInt8(0));

  //errs() << "gep is " << *gep << "\n";

  return true;
}

char OccupancyReduction::ID = 0;
static RegisterPass<OccupancyReduction> X("ored", "Occupancy Reduction Pass - Reserves redundant shared memory");
