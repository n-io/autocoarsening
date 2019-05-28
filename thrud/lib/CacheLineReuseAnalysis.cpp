#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <map>
#include <list>
#include <functional>
#include <algorithm>

#include "thrud/CacheLineReuseAnalysis.h"
#include "thrud/MemAccessDescriptor.h"
#include "thrud/NDRange.h"
#include "thrud/Utils.h"

#define REQUIRE_CANONICAL_LOOPxxx
#define DEBUG_PRINTxxx

using namespace llvm;

const int MAX_DIMENSIONS[] = {32,2,2};

extern cl::opt<std::string> KernelNameCL;
cl::opt<unsigned int> WarpSize("warp-size", cl::init(32), cl::Hidden, cl::desc("The size of one warp within which threads perform lock-step execution"));
cl::opt<unsigned int> CacheLineSize("cache-line-size", cl::init(32), cl::Hidden, cl::desc("The size of a cache line in bytes"));

void CacheLineReuseAnalysis::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<NDRange>();
  au.addRequired<LoopInfo>();
}

int CacheLineReuseAnalysis::getDimensionality() {
  int dimensions = 0;
  for (int dimension = 0; dimension < NDRange::DIRECTION_NUMBER; ++dimension) {
    if (ndr->getTids(dimension).size() > 0) {
      dimensions = dimensions + 1;
    }
  }
  return dimensions;
}

Value * CacheLineReuseAnalysis::getAccessedSymbolPtr(Value * v) {
  // returns what holds the pointer (e.g. GetElementPtr or pointer passed as function arg) to the accessed symbol
  if (Instruction * inst = dyn_cast<Instruction>(v)) {
    switch (inst->getOpcode()) {
      case Instruction::Load:
	return getAccessedSymbolPtr(inst->getOperand(0));
      case Instruction::Store:
	return getAccessedSymbolPtr(inst->getOperand(1));
      case Instruction::GetElementPtr:
	return inst;
      default:
	diagnosis = "Could not cast operand to any known type (opcode: " + std::to_string(inst->getOpcode()) + ")";
	return v;
    }
  } else if (isa<Argument>(v)) {
    return v;
  }
  diagnosis = "No case for retreiving symbol ptr for " + std::string(v->getName());
  return v;
}

StringRef CacheLineReuseAnalysis::getAccessedSymbolName(Value * v) {
  Value * ptr = getAccessedSymbolPtr(v);
  if (GetElementPtrInst * gep = dyn_cast<GetElementPtrInst>(ptr)) {
    return gep->getOperand(0)->getName();
  } else if (isa<Argument>(v)) {
    return v->getName();
  }
#ifdef DEBUG_PRINT
  errs() << " Error on retrieving symbol name of " << *v  << " (name: " << v->getName() << ")\n";
#endif
  diagnosis = "No case for retrieving symbol name of " + std::string(v->getName());
  return "";
}

bool CacheLineReuseAnalysis::isCachedAddressSpace(Instruction * inst) {
  // For convenience, this also takes load/store instructions directly
  switch(inst->getOpcode()) {
    case Instruction::Load:
      return dyn_cast<LoadInst>(inst)->getPointerAddressSpace() == GLOBAL_ADDRESS_SPACE;
    case Instruction::Store:
      return dyn_cast<StoreInst>(inst)->getPointerAddressSpace() == GLOBAL_ADDRESS_SPACE;
    case Instruction::GetElementPtr:
      return dyn_cast<GetElementPtrInst>(inst)->getPointerAddressSpace() == GLOBAL_ADDRESS_SPACE;
    default:
      diagnosis = "Could not obtain address space of unknown instruction type (opcode: " + std::to_string(inst->getOpcode()) + ")";
      return false;
  }
}

PHINode *CacheLineReuseAnalysis::getInductionVariable(Loop* loop) const {
  // This is a variation of Loop::getCanonicalInductionVariable with a few changes commented out
  BasicBlock *H = loop->getHeader();

  BasicBlock *Incoming = nullptr, *Backedge = nullptr;
  pred_iterator PI = pred_begin(H);
  assert(PI != pred_end(H) &&
         "Loop must have at least one backedge!");
  Backedge = *PI++;
  if (PI == pred_end(H)) return nullptr;  // dead loop
  Incoming = *PI++;
  if (PI != pred_end(H)) return nullptr;  // multiple backedges?

  if (loop->contains(Incoming)) {
    if (loop->contains(Backedge))
      return nullptr;
    std::swap(Incoming, Backedge);
  } else if (!loop->contains(Backedge))
    return nullptr;

  // Loop over all of the PHI nodes, looking for a canonical indvar.
  for (BasicBlock::iterator I = H->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
#ifdef DEBUG_PRINT
    errs() << "Looping over phi-node " << *PN << "\n";
#endif
#ifdef REQUIRE_CANONICAL_LOOP
    if (ConstantInt *CI =
        dyn_cast<ConstantInt>(PN->getIncomingValueForBlock(Incoming)))
      //if (CI->isNullValue()) 
        if (Instruction *Inc =
            dyn_cast<Instruction>(PN->getIncomingValueForBlock(Backedge)))
          if (Inc->getOpcode() == Instruction::Add &&
                Inc->getOperand(0) == PN)
            if (ConstantInt *CI = dyn_cast<ConstantInt>(Inc->getOperand(1)))
              //if (CI->equalsInt(1))
#endif
                return PN;
  }
  return nullptr;
}

void CacheLineReuseAnalysis::preprocess(Function *function, std::set<Instruction*>& memops, std::set<Instruction*>& relevantInstructions) {
  std::set<Instruction*> defs;
  std::set<BasicBlock*> relevantBlocks;
  // find all cached memory accesses
  for (auto iter = function->begin(), end = function->end(); iter != end; iter++) {
#ifdef DEBUG_PRINT
    errs() << "Basic block " << iter->getName() << " has " << iter->size() << " instructions "
           << "(part of loop: " << loopInfo->getLoopFor(iter)
           << " is loop header: " << loopInfo->isLoopHeader(iter)
           << " loop depth: " << loopInfo->getLoopDepth(iter) << ")\n";
#endif
    for (BasicBlock::iterator inst = iter->begin(), e = iter->end(); inst != e; ++inst) {
      // only process loads and stores to global memory
      errs() << "  " << *inst << "\n";
      lastInstruction = inst;
      if ((inst->getOpcode() == Instruction::Load || inst->getOpcode() == Instruction::Store) && isCachedAddressSpace(inst)) {
	memops.insert(inst);
	const int paramIdx = inst->getOpcode() == Instruction::Load ? 0 : 1;
	if (Instruction *operand = dyn_cast<Instruction>(inst->getOperand(paramIdx))) {
	  while (operand->getOpcode() == Instruction::BitCast) {
            operand = dyn_cast<Instruction>(operand->getOperand(0));
	  }
#ifdef DEBUG_PRINT
          errs () << "preprocessing " << *operand << "\n";
#endif
	  defs.insert(operand);
          StringRef symbolName = getAccessedSymbolName(operand);
          if (!symbolName.empty()) {
            accessedCacheLines.insert(std::pair<StringRef, std::set<int>>(symbolName, std::set<int>()));
            relevantBlocks.insert(iter);
          }
	}
      }
    }
  }
  // traverse the use-def chain to find all relevant definitionAs
  while (defs.size() > 0) {
    Instruction *inst = *defs.begin();
    defs.erase(defs.begin());
    // avoid looping infinitely
    if (relevantInstructions.count(inst) == 0) {
      relevantInstructions.insert(inst);
      relevantBlocks.insert(inst->getParent());
      for (Use &u : inst->operands()) {
	if (Instruction *operand = dyn_cast<Instruction>(u.get())) {
	  defs.insert(operand);
	}
      }
    }
  }

  // iterate over loops
#ifdef DEBUG_PRINT
  errs() << "Iterating over loops\n";
#endif
  for (auto it = loopInfo->begin(); it != loopInfo->end(); it++) {
    Loop * loop = *it;
    for (auto blkIt = loop->block_begin(); blkIt != loop->block_end(); blkIt++) {
      if (relevantBlocks.find(*blkIt) != relevantBlocks.end()) {
        relevantLoops.insert(*it);
      }
    }
    std::vector<Loop *> subLoops = loop->getSubLoopsVector();
    for (auto subLoopIt = subLoops.begin(); subLoopIt != subLoops.end(); subLoopIt++) {
      Loop * subLoop = *subLoopIt;
      for (auto blkIt = subLoop->block_begin(); blkIt != subLoop->block_end(); blkIt++) {
	if (relevantBlocks.find(*blkIt) != relevantBlocks.end()) {
	  relevantLoops.insert(*subLoopIt);
	}
      }
      
    }
  }
  for (auto it = relevantLoops.begin(); it != relevantLoops.end(); it++) {
    PHINode * pn = getInductionVariable(*it);
    //errs() << "IsLoopSimplifyForm? " << (*it)->isLoopSimplifyForm() << "\n";
    if (pn == NULL) {
      diagnosis = "Loop structure too complicated";
      break;
    }
#ifdef DEBUG_PRINT
    errs() << "Canonical induction variable is: " << *pn << "\n";
    for (unsigned int i = 0; i < pn->getNumOperands(); i++) {
      errs() << "  operand: " << *pn->getOperand(i) << " types: " << isa<Instruction>(pn->getOperand(i)) << isa<ConstantInt>(pn->getOperand(i)) << "   " << (isa<ConstantInt>(pn->getOperand(i)) ? dyn_cast<ConstantInt>(pn->getOperand(i))->getValue().getSExtValue() : 1234) << "\n";
    }
    errs() << "Loop has the following blocks:\n";
    for (auto bi = (*it)->block_begin(); bi != (*it)->block_end(); bi++) {
      errs() << "  " << (*bi)->getName() << "\n";
    }
#endif
  }
}

bool CacheLineReuseAnalysis::runOnFunction(Function &F) {
  Function *function = (Function *)&F;
  if (!isKernel(function))
    return false;

  // Apply the pass to the selected kernel only.
  std::string FunctionName = F.getName();
  if (KernelNameCL != "" && FunctionName != KernelNameCL)
    return false;

  ndr = &getAnalysis<NDRange>();
  loopInfo = &getAnalysis<LoopInfo>();


  dimensions = getDimensionality();
#ifdef DEBUG_PRINT
  errs() << "Kernel ";
  errs().write_escaped(F.getName()) << " is " << dimensions << "-dimensional\n";
#endif

  //std::set<Instruction*> relevantInstructions;
  preprocess(function, memops, relevantInstructions);

#ifdef DEBUG_PRINT
  errs() << "Printing memops:\n";
  for (Instruction * memop : memops) {
    errs() << *memop << "\n";
  }
  errs() << "There are " << relevantInstructions.size() << " relevant instructions\n";

  // do a bit of printing 
  for (auto iter = function->begin(), end = function->end(); iter != end; iter++) {
    errs() << "Basic block " << iter->getName() << " has " << iter->size() << " instructions\n";
    for (BasicBlock::iterator inst = iter->begin(), e = iter->end(); inst != e; ++inst) {
      errs() << (relevantInstructions.count(inst) ? ">>" : "  ") << *inst << "\n";
    }
  }
#endif

  accessDescriptorStack.push_back(std::map<Instruction*, vector<MemAccessDescriptor>>());
  simulate(inst_begin(F), lastInstruction, NULL);

#ifdef DEBUG_PRINT
  errs() << F.getName() << " used " << MemAccessDescriptor::SIZE << " bytes for MADs and " << MemAccessDescriptor::CACHE_SIZE << " for cache: "
         << (MemAccessDescriptor::SIZE + MemAccessDescriptor::CACHE_SIZE) << " bytes\n";
#endif
  if (diagnosis.empty()) {
    errs() << "No cache line re-use detected, OK to coarsen\n";
  } else {
    errs() << diagnosis << "\n";
  }
  return false;
}

inst_iterator
CacheLineReuseAnalysis::simulate(inst_iterator it, Instruction* fwdDef, Loop *innermostLoop) {
  bool done = false;
  while (!done) {
    // Loop up to a later def or until diagnosis is set
    done = (&*it == fwdDef) || !diagnosis.empty();
    Instruction* inst = &*it++;
#ifdef DEBUG_PRINT
    errs() << (relevantInstructions.count(inst) ? ">>" : "  ") << *inst << (ndr->isTid(inst) ? " TID[" + std::to_string(ndr->getDirection(inst))+"]" : "") << "\n";
#endif
    if (relevantInstructions.count(inst) > 0 && diagnosis.empty()) {
      if (ndr->isLocal(inst) || ndr->isGlobal(inst)) {
	int dimension = ndr->getDirection(inst);
	MemAccessDescriptor v(dimension, MAX_DIMENSIONS[dimension]);
        addToStack(inst, v);
      } else if (ndr->isGlobalSize(inst) || ndr->isLocalSize(inst)) {
	int n = 1;
	for (int i = 0; i < dimensions; i++) {
	  n *= MAX_DIMENSIONS[i];
	}
        addToStack(inst, MemAccessDescriptor(n));
      } else if (ndr->isGroupId(inst)) {
        addToStack(inst, MemAccessDescriptor(0));
      } else if (ndr->isGroupsNum(inst)) {
        addToStack(inst, MemAccessDescriptor(1));
      } else if (memops.count(inst) > 0) {
        StringRef accessedSymbolName = getAccessedSymbolName(inst);
        diagnosis = "Program is data dependent in access to [" + std::string(accessedSymbolName) + "]";
      } else if (inst->getOpcode() == Instruction::Add) {
	applyBinaryOp(std::plus<int>(), inst);
      } else if (inst->getOpcode() == Instruction::Sub) {
        applyBinaryOp(std::minus<int>(), inst);
      } else if (inst->getOpcode() == Instruction::Mul) {
	applyBinaryOp(std::multiplies<int>(), inst);
      } else if (inst->getOpcode() == Instruction::UDiv) {
	applyBinaryOp(std::divides<int>(), inst);
      } else if (inst->getOpcode() == Instruction::URem) {
	applyBinaryOp(std::modulus<int>(), inst);
      } else if (inst->getOpcode() == Instruction::Shl) {
        applyBinaryOp([](int a, int b) {return a << b;}, inst);
      } else if (inst->getOpcode() == Instruction::Or) {
        applyBinaryOp(std::bit_or<int>(), inst);
      } else if (inst->getOpcode() == Instruction::And) {
        applyBinaryOp(std::bit_and<int>(), inst);
      } else if (inst->getOpcode() == Instruction::Xor) {
        applyBinaryOp(std::bit_xor<int>(), inst);
      } else if (inst->getOpcode() == Instruction::Select) {
        std::vector<MemAccessDescriptor> preds = getOperand(inst->getOperand(0));
	std::vector<MemAccessDescriptor> ops1 = getOperand(inst->getOperand(1));
	std::vector<MemAccessDescriptor> ops2 = getOperand(inst->getOperand(2));
	std::vector<MemAccessDescriptor> result;
	for (MemAccessDescriptor pred : preds) {
	  for (MemAccessDescriptor op1 : ops1) {
	    for (MemAccessDescriptor op2 : ops2) {
	      result.push_back(pred.select(op1, op2));
	    }
	  }
	}
	addToStack(inst, result);

      } else if (inst->getOpcode() == Instruction::SExt) {
        addToStack(inst, getOperand(inst->getOperand(0))); 
      } else if (inst->getOpcode() == Instruction::Trunc) {
        addToStack(inst, getOperand(inst->getOperand(0))); 
      } else if (inst->getOpcode() == Instruction::ICmp) {
        ICmpInst *iCmpInst = dyn_cast<ICmpInst>(inst);
        switch (iCmpInst->getPredicate()) {
          case ICmpInst::Predicate::ICMP_EQ:  applyBinaryOp(std::equal_to<int>(), inst); break;
          case ICmpInst::Predicate::ICMP_UGT: applyBinaryOp(std::greater<int>(), inst); break;
          case ICmpInst::Predicate::ICMP_UGE: applyBinaryOp(std::greater_equal<int>(), inst); break;
          case ICmpInst::Predicate::ICMP_ULT: applyBinaryOp(std::less<int>(), inst); break;
          case ICmpInst::Predicate::ICMP_ULE: applyBinaryOp(std::less_equal<int>(), inst); break;
          case ICmpInst::Predicate::ICMP_SGT: applyBinaryOp(std::greater<int>(), inst); break;
          case ICmpInst::Predicate::ICMP_SGE: applyBinaryOp(std::greater_equal<int>(), inst); break;
          case ICmpInst::Predicate::ICMP_SLT: applyBinaryOp(std::less<int>(), inst); break;
          case ICmpInst::Predicate::ICMP_SLE: applyBinaryOp(std::less_equal<int>(), inst); break;
          default: diagnosis = "Unknown icmp predicate";
        }
#ifdef DEBUG_PRINT
        errs() << "Inst is an icmp, predicate is: " << iCmpInst->getPredicate() << " eq is " << ICmpInst::Predicate::ICMP_EQ << "\n";
#endif
      } else if (inst->getOpcode() == Instruction::BitCast) {
        // NoOp typecast...
#ifdef DEBUG_PRINT
	errs() << "Encountered bitcast, op0 is " << inst->getOperand(0)->getName() << "\n";
#endif
	addToStack(inst, getOperand(inst->getOperand(0)));
      } else if (inst->getOpcode() == Instruction::GetElementPtr) {
	// this does not do anything interesting itself - forward the last MemAccessDescriptor
        addToStack(inst, getOperand(inst->getOperand(1)));
      } else if (inst->getOpcode() == Instruction::PHI) {
        Loop* loop = loopInfo->getLoopFor(inst->getParent());
#ifdef DEBUG_PRINT
        errs () << "Instruction is \n"
                << "   -> " << *inst << "\n";
        errs() << "Processing phi instruction... Loop is " << loop << " and innermost loop is " << innermostLoop << "\n";
#endif
        if (loop != NULL && loop != innermostLoop /*&& inst == loop->getCanonicalInductionVariable()*/) {
          std::map<Instruction*, std::vector<MemAccessDescriptor>> baseIt;
          std::map<Instruction*, std::vector<MemAccessDescriptor>> stepIt;
          std::vector<MemAccessDescriptor> baseItVector;
          Instruction * nextFwdDef = NULL;
          for (unsigned int i = 0; i < inst->getNumOperands(); i++) {
            Value * v = inst->getOperand(i);
#ifdef DEBUG_PRINT
            errs() << "  ==> operand" << i << " is: " << *v << "\n";
#endif
            if (isa<Instruction>(v) && isFwdDef(dyn_cast<Instruction>(v))) {
              if (nextFwdDef != NULL) {
                diagnosis = "More than one fwd def found for loop";
              }
              nextFwdDef = dyn_cast<Instruction>(inst->getOperand(i));
            } else {
              std::vector<MemAccessDescriptor> mads = getOperand(inst->getOperand(i));
              baseItVector.insert(baseItVector.end(), mads.begin(), mads.end());
            }
          }
#ifdef DEBUG_PRINT
	  if (nextFwdDef != 0) {
            errs() << "Executing until: " << *nextFwdDef << "\n";
	  } else {
            errs() << "Executing until: null\n";
	  }
#endif
          baseIt.insert(std::pair<Instruction*, std::vector<MemAccessDescriptor>>(inst, baseItVector));
          accessDescriptorStack.push_back(baseIt);
          simulate(it, nextFwdDef, loop);
          if (!diagnosis.empty()) return it;
          baseIt = accessDescriptorStack.back();
          accessDescriptorStack.pop_back();
          std::vector<MemAccessDescriptor> nextFwdDefVec = baseIt.find(nextFwdDef)->second;
          // supposedly that's all we need for the next iteration
          stepIt.insert(std::pair<Instruction*, std::vector<MemAccessDescriptor>>(inst, nextFwdDefVec));
          accessDescriptorStack.push_back(stepIt);
          it = simulate(it, nextFwdDef, loop);
          stepIt = accessDescriptorStack.back();
          accessDescriptorStack.pop_back();
          // merge baseIt and stepIt into current scope
          mergeIntoStack(baseIt);
          mergeIntoStack(stepIt);
        } else {
	  std::vector<MemAccessDescriptor> mads;
#ifdef DEBUG_PRINT
          errs() << "processing an if for inst " << *inst << "\n";
#endif
	  for (unsigned int i = 0; i < inst->getNumOperands(); i++) {
	    std::vector<MemAccessDescriptor> phiBranch = getOperand(inst->getOperand(i));
	    mads.insert(mads.end(), phiBranch.begin(), phiBranch.end());
	  }
	  addToStack(inst, mads);
        }
      } else {
	diagnosis = "Unknown opcode - " + std::to_string(inst->getOpcode());
      }
    } else if (memops.count(inst) > 0 && diagnosis.empty()) {
#ifdef DEBUG_PRINT
      errs() << "Simulating mem access for " << *inst << "\n";
#endif
      int alignment = -1;
      bool isStore = false;
      if (inst->getOpcode() == Instruction::Load) {
	alignment = dyn_cast<LoadInst>(inst)->getAlignment();
      } else if (inst->getOpcode() == Instruction::Store) {
	alignment = dyn_cast<StoreInst>(inst)->getAlignment();
	isStore = true;
      }
      Value * ptr = getAccessedSymbolPtr(inst);
      StringRef accessedSymbolName = getAccessedSymbolName(ptr);
      std::vector<MemAccessDescriptor> mads = getOperand(ptr);
      std::set<int> * prevAccesses = &accessedCacheLines[accessedSymbolName];
      std::set<int> accessesToAdd; 
      for (MemAccessDescriptor & mad : mads) {
	mad.print();
	bool fullCoalescing = true;
	list<int> accesses = mad.getMemAccesses(WarpSize, alignment, CacheLineSize, &fullCoalescing);
	int accessesNum = accesses.size();
	accesses.sort();
	accesses.unique();
	int uniqueAccessesNum = accesses.size();
	int duplicates = accessesNum - uniqueAccessesNum;

	if (duplicates == 0 && uniqueAccessesNum > 1) {
	  std::vector<int> intersection(prevAccesses->size() + accesses.size());
	  duplicates = set_intersection(prevAccesses->begin(), prevAccesses->end(), accesses.begin(), accesses.end(), intersection.begin()) - intersection.begin();
	}

	accessesToAdd.insert(accesses.begin(), accesses.end());

#ifdef DEBUG_PRINT
	errs() << "Returned " << accessesNum << " accesses to " << uniqueAccessesNum << " unique cache lines with " << duplicates << " duplicates\n";
#endif
	if (isStore && fullCoalescing) {
	  errs() << "Ignoring mem accesses of fully coalesced store instruction";
	} else if (duplicates > 0 && uniqueAccessesNum > 1) {
	  diagnosis = "Cache line re-use in access to [" + std::string(accessedSymbolName) + "]";
	}
      }
      prevAccesses->insert(accessesToAdd.begin(), accessesToAdd.end());
    }
  }
  return it;
}

bool CacheLineReuseAnalysis::isFwdDef(Instruction* inst) {
  // An instruction that has not yet been encountered in the control flow
  // Must be a relevant inst
  for (auto it = accessDescriptorStack.rbegin(); it != accessDescriptorStack.rend(); it++) {
    if (it->count(inst) > 0) return false;
  }
  return true;
}

void CacheLineReuseAnalysis::mergeIntoStack(std::map<Instruction*, std::vector<MemAccessDescriptor>> &defs) {
  for (auto const &p : defs) {
    if (accessDescriptorStack.back().count(p.first) == 0) {
      accessDescriptorStack.back().insert(p);
    } else {
      std::vector<MemAccessDescriptor> * vec = &accessDescriptorStack.back().find(p.first)->second;
      vec->insert(vec->end(), p.second.begin(), p.second.end());
    }
  }
}

void CacheLineReuseAnalysis::applyBinaryOp(function<int(int, int)> f, Instruction * inst) {
  std::vector<MemAccessDescriptor> ops1 = getOperand(inst->getOperand(0));
  std::vector<MemAccessDescriptor> ops2 = getOperand(inst->getOperand(1));
  std::vector<MemAccessDescriptor> result;
  for (MemAccessDescriptor op1 : ops1) {
    for (MemAccessDescriptor op2 : ops2) {
      result.push_back(op1.compute(f, op2));
    }
  }
  addToStack(inst, result);
}

std::vector<MemAccessDescriptor> CacheLineReuseAnalysis::findInStack(Instruction* inst) {
  for (auto it = accessDescriptorStack.rbegin(); it != accessDescriptorStack.rend(); it++) {
    if (it->count(inst) > 0) return it->find(inst)->second;
  }

#ifdef DEBUG_PRINT
  errs() << "No prev def found, inst is " << inst;
  if (inst != NULL)
    errs () << " (deref: " << *inst << ")";
  errs() << "\n";
#endif
  return std::vector<MemAccessDescriptor>();
}

std::vector<MemAccessDescriptor> CacheLineReuseAnalysis::getOperand(Value * v) {
  if (ConstantInt * intval = dyn_cast<ConstantInt>(v)) {
    return std::vector<MemAccessDescriptor>{MemAccessDescriptor(intval->getValue().getSExtValue())};
  } else if (Instruction * inst = dyn_cast<Instruction>(v)) {
    return findInStack(inst);
  } else if (isa<Argument>(v)) {
    return std::vector<MemAccessDescriptor>{MemAccessDescriptor(10000)};
  } else if (isa<UndefValue>(v)) {
    return std::vector<MemAccessDescriptor>();
  } else {
    diagnosis = "Unknown operand type";
    return std::vector<MemAccessDescriptor>();
  }
}

char CacheLineReuseAnalysis::ID = 0;
static RegisterPass<CacheLineReuseAnalysis> X("clr", "Cache Line Re-Use Analysis Pass");
