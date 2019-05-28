#include "thrud/ThreadCoarsening.h"

#include "thrud/DataTypes.h"
#include "thrud/Utils.h"

//------------------------------------------------------------------------------
void ThreadCoarsening::scaleNDRange() {
  InstVector InstTids;
  scaleSizes();
  scaleIds();
}

//------------------------------------------------------------------------------
void ThreadCoarsening::scaleSizes() {
  InstVector sizeInsts = THREAD_LEVEL_COARSENING ? ndr->getSizes(direction) : ndr->getGlobalSizes(direction);

  //errs() << "Size instructions:\n";
  //dumpVector(sizeInsts);
  
  /*if (THREAD_LEVEL_COARSENING)
    errs() << "Applying thread-level coarsening\n";
  else 
    errs() << "Applying block-level coarsening\n";*/

  for (InstVector::iterator iter = sizeInsts.begin(), iterEnd = sizeInsts.end();
       iter != iterEnd; ++iter) {
    // Scale size.
    Instruction *inst = *iter;
    Instruction *mul = getMulInst(inst, factor);
    mul->insertAfter(inst);
    //errs() << "Inserting " << *mul << "\n";
    // Replace uses of the old size with the scaled one.
    replaceUses(inst, mul);
  }
}

//------------------------------------------------------------------------------

// Scaling function: origTid = [newTid / st] * cf * st + newTid % st + subid * st
void ThreadCoarsening::scaleIdsThreadLevelCoarsening() {
  unsigned int cfst = factor * stride;

  InstVector tids = ndr->getTids(direction);
  for (InstVector::iterator instIter = tids.begin(), instEnd = tids.end();
       instIter != instEnd; ++instIter) {
    Instruction *inst = *instIter;

    // Compute base of new tid.
    Instruction *div = getDivInst(inst, stride); 
    div->insertAfter(inst);
    Instruction *mul = getMulInst(div, cfst);
    mul->insertAfter(div);
    Instruction *modulo = getModuloInst(inst, stride);
    modulo->insertAfter(mul);
    Instruction *base = getAddInst(mul, modulo);
    base->insertAfter(modulo);

    // Replace uses of the threadId with the new base.
    replaceUses(inst, base);
    modulo->setOperand(0, inst);
    div->setOperand(0, inst);

    // Compute the remaining thread ids.
    cMap.insert(std::pair<Instruction *, InstVector>(inst, InstVector()));
    InstVector &current = cMap[base];
    current.reserve(factor - 1);

    Instruction *bookmark = base;
    for (unsigned int index = 2; index <= factor; ++index) {
      Instruction *add = getAddInst(base, (index - 1) * stride);
      add->insertAfter(bookmark);
      current.push_back(add);
      bookmark = add;
    }
  }
}

void ThreadCoarsening::scaleIds() {
  if (THREAD_LEVEL_COARSENING) {
    scaleIdsThreadLevelCoarsening();
    return;
  }

  // replace all globalIds with (groupId * localSize + localId)
  InstVector gids = ndr->getGids(direction);

  /*std::vector<Instruction *> unusedInsts;
  for (InstVector::iterator instIter = gids.begin(), instEnd = gids.end();
       instIter != instEnd; ++instIter) {
    Instruction *inst = *instIter;

    LLVMContext & context = inst->getContext();
    IntegerType * intType = IntegerType::getInt32Ty(context);

    CallInst *cinst = (CallInst *) inst;

    ConstantInt *cint = dyn_cast<ConstantInt>(cinst->getArgOperand(0));

    //IRBuilder<> builder(inst->getParent());
    //builder.createCall(ndr->getOclFunctionPtr(NDRange::GET_GROUP_ID), ArrayRef(direction));
    ////ConstantInt * const param = ConstantInt::get(intType, direction);
    //ArrayRef<ConstantInt> params(*param);
    
    //ArrayRef<Value *> params(ConstantInt::get(intType, direction));
    //CallInst *groupId = CallInst::Create(ndr->getOclFunctionPtr(NDRange::GET_GROUP_ID), ArrayRef<Value *>(ConstantInt::get(intType, direction)));
    //std::cout << "Finished\n";
    

    CallInst *groupId = (CallInst* ) inst->clone();

    groupId->setCalledFunction(ndr->getOclFunctionPtr(NDRange::GET_GROUP_ID));
    groupId->insertAfter(inst);
    ndr->registerOclInst(direction, NDRange::GET_GROUP_ID, groupId);

    CallInst *localSize = (CallInst* ) inst->clone();
    localSize->setCalledFunction(ndr->getOclFunctionPtr(NDRange::GET_LOCAL_SIZE));
    localSize->insertAfter(groupId);
    ndr->registerOclInst(direction, NDRange::GET_LOCAL_SIZE, localSize);

    CallInst *localId = (CallInst* ) inst->clone();
    localId->setCalledFunction(ndr->getOclFunctionPtr(NDRange::GET_LOCAL_ID));
    localId->insertAfter(localSize);
    ndr->registerOclInst(direction, NDRange::GET_LOCAL_ID, localId);

    Instruction *mul = getMulInst(groupId, localSize);
    mul->insertAfter(localId);
    Instruction *add = getAddInst(mul, localId);
    add->insertAfter(mul);

    replaceUses(inst, add);
    
    ndr->unregisterOclInst(direction, NDRange::GET_GLOBAL_ID, inst);
    errs () << "(NDRangeScaling) Trying to remove inst from parent >> " << *inst << "\n";
    unusedInsts.push_back(inst); 
    //inst->removeFromParent();
  }
  for (auto it = unusedInsts.begin(); it != unusedInsts.end(); ++it) {
    Instruction *inst = *it;
    inst->eraseFromParent();
  }*/
  //scaleIds2();
  // replace all groupIds with (cf * groupId + i)
  InstVector groupIds = ndr->getGroupIds(direction); 
  for (InstVector::iterator instIter = groupIds.begin(), instEnd = groupIds.end();
       instIter != instEnd; ++instIter) {
    Instruction *inst = *instIter;
    Instruction *base = getMulInst(inst, factor);
    base->insertAfter(inst);
    replaceUses(inst, base);
    base->setOperand(0, inst);
   
    cMap.insert(std::pair<Instruction *, InstVector>(inst, InstVector()));
    InstVector &current = cMap[base];
    current.reserve(factor - 1);

    Instruction *bookmark = base;
    for (unsigned int index = 2; index <= factor; ++index) {
      Instruction *add = getAddInst(base, (index - 1));
      //errs() << "Creating instruction for cf " << (index - 1) << " >> " << *add << "\n";
      add->insertAfter(bookmark);
      current.push_back(add);
      bookmark = add;
    }

  }
  //dumpCoarseningMap(cMap);
}


