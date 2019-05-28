#include "thrud/ThreadCoarsening.h"

#include "thrud/DataTypes.h"
#include "thrud/Utils.h"

#include "llvm/IR/Module.h"

#include "llvm/Transforms/Utils/Cloning.h"

//------------------------------------------------------------------------------
void ThreadCoarsening::coarsenFunction() {
  RegionVector &regions = sdda->getOutermostDivRegions();
  InstVector &insts = sdda->getOutermostDivInsts();

  // Replicate shMem held in global vars.
  if (!THREAD_LEVEL_COARSENING) {
    Function *f = !insts.empty() ? insts[0]->getParent()->getParent() : nullptr;
    if (f == nullptr) {
      errs() << "Could not obtain function ptr from div insts\n";
      return;
    }
    shMemGlobals = sdda->getShMemGlobalsUsedIn(f);

    std::for_each(shMemGlobals.begin(), shMemGlobals.end(),
		  [this](GlobalVariable *gv) { replicateGlobal(gv); });
  }

  // Replicate instructions.
  std::for_each(insts.begin(), insts.end(),
                [this](Instruction *inst) { replicateInst(inst); });

  // Replicate regions.
  std::for_each(regions.begin(), regions.end(),
                [this](DivergentRegion *region) { replicateRegion(region); });
}

//------------------------------------------------------------------------------
void ThreadCoarsening::replicateGlobal(GlobalVariable *gv) {
  for (unsigned int index = 0; index < factor - 1; ++index) {
    Module &module = *(gv->getParent());
    GlobalVariable *newGV = new GlobalVariable(module,
                                               gv->getType()->getPointerElementType(),
                                               gv->isConstant(),
                                               gv->getLinkage(),
                                               gv->getInitializer(),
                                               gv->getName() + "..cf" + Twine(index + 2), //TODO
                                               (GlobalVariable *) nullptr,
                                               gv->getThreadLocalMode(),
                                               gv->getType()->getAddressSpace());
    newGV->copyAttributesFrom(gv);
    shMemGlobalsCMap[gv].push_back(newGV);
  }
}

//------------------------------------------------------------------------------
void ThreadCoarsening::replicateInst(Instruction *inst) {
  InstVector current;
  current.reserve(factor - 1);
  Instruction *bookmark = inst;

  for (unsigned int index = 0; index < factor - 1; ++index) {
    // Clone.
    Instruction *newInst = inst->clone();
    renameValueWithFactor(newInst, inst->getName(), index);
    applyCoarseningMap(newInst, index);
    // Insert the new instruction.
    newInst->insertAfter(bookmark);
    bookmark = newInst;
    // Add the new instruction to the coarsening map.
    current.push_back(newInst);
  }
  cMap.insert(std::pair<Instruction *, InstVector>(inst, current));

  updatePlaceholderMap(inst, current);
}

//------------------------------------------------------------------------------
void ThreadCoarsening::updatePlaceholderMap(Instruction *inst,
                                            InstVector &coarsenedInsts) {
  // Update placeholder replacement map.
  auto phIter = phMap.find(inst);
  if (phIter != phMap.end()) {
    InstVector &coarsenedPhs = phIter->second;
    for (unsigned int index = 0; index < coarsenedPhs.size(); ++index) {
      phReplacementMap[coarsenedPhs[index]] = coarsenedInsts[index];
    }
  }
}

//------------------------------------------------------------------------------
void ThreadCoarsening::applyCoarseningMap(DivergentRegion &region,
                                          unsigned int index) {
  std::for_each(region.begin(), region.end(), [index, this](BasicBlock *block) {
    applyCoarseningMap(block, index);
  });
}

//------------------------------------------------------------------------------
void ThreadCoarsening::applyCoarseningMap(BasicBlock *block,
                                          unsigned int index) {
  for (auto iter = block->begin(), iterEnd = block->end();
       iter != iterEnd; ++iter) {
    applyCoarseningMap(iter, index);
  }
}

//------------------------------------------------------------------------------
void ThreadCoarsening::applyCoarseningMap(Instruction *inst,
                                          unsigned int index) {
  for (unsigned int opIndex = 0, opEnd = inst->getNumOperands();
       opIndex != opEnd; ++opIndex) {
    Instruction *operand = dyn_cast<Instruction>(inst->getOperand(opIndex));
    if (operand != nullptr) {
      Instruction *newOp = getCoarsenedInstruction(operand, index);
      if (newOp == nullptr)
	continue;
      inst->setOperand(opIndex, newOp);
    } else if (!THREAD_LEVEL_COARSENING) {
      if (GlobalVariable *gv = dyn_cast<GlobalVariable>(inst->getOperand(opIndex))) {
	if (shMemGlobals.find(gv) != shMemGlobals.end()) {
	  GlobalVariable *divGV = shMemGlobalsCMap[gv][index];
	  inst->setOperand(opIndex, divGV);
	}
      } else if (LoadInst *loadInst = dyn_cast<LoadInst>(inst)) {
	if (isSharedMemAddressSpace(loadInst->getPointerAddressSpace())) {
	  if (ConstantExpr *ptrOp = dyn_cast<ConstantExpr>(loadInst->getPointerOperand())) {
	    if (GlobalVariable *gv = dyn_cast<GlobalVariable>(ptrOp->getOperand(0))) {
	      if (shMemGlobals.find(gv) != shMemGlobals.end()) {
		GlobalVariable *divGV = shMemGlobalsCMap[gv][index];
		Constant *newOp = ptrOp->getWithOperandReplaced(0, divGV);
		loadInst->setOperand(0, newOp);
	      }
	    }
	  }
	}
      }
    }
  }
}

//------------------------------------------------------------------------------
Instruction *
ThreadCoarsening::getCoarsenedInstruction(Instruction *inst,
                                          unsigned int coarseningIndex) {
  CoarseningMap::iterator It = cMap.find(inst);
  // The instruction is in the map.
  if (It != cMap.end()) {
    InstVector &entry = It->second;
    Instruction *result = entry[coarseningIndex];
    return result;
  } else {
    // The instruction is divergent.
    if (sdda->isDivergent(inst)) {
      // Look in placeholder map.
      CoarseningMap::iterator phIt = phMap.find(inst);
      Instruction *result = nullptr;
      if (phIt != phMap.end()) {
        // The instruction is in the placeholder map.
        InstVector &entry = phIt->second;
        result = entry[coarseningIndex];
      }
      // The instruction is not in the placeholder map.
      else {
        // Make an entry in the placeholder map.
        InstVector newEntry;
        for (unsigned int counter = 0; counter < factor - 1; ++counter) {
          Instruction *ph = inst->clone();
          ph->insertAfter(inst);
          renameValueWithFactor(
              ph, (inst->getName() + Twine(".place.holder")).str(),
              coarseningIndex);
          newEntry.push_back(ph);
        }
        phMap.insert(std::pair<Instruction *, InstVector>(inst, newEntry));
        // Return the appropriate placeholder.
        result = newEntry[coarseningIndex];
      }
      return result;
    }
  }
  return nullptr;
}

//------------------------------------------------------------------------------
void ThreadCoarsening::replacePlaceholders() {
  // Iterate over placeholder map.
  for (auto &mapIter : phMap) {
    InstVector &phs = mapIter.second;
    // Iteate over placeholder vector.
    for (auto ph : phs) {
      Value *replacement = phReplacementMap[ph];
      if (replacement != nullptr && ph != replacement)
        ph->replaceAllUsesWith(replacement);
    }
  }
}
