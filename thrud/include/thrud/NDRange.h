#ifndef NDRANGE_H
#define NDRANGE_H

#include "thrud/Utils.h"

#include "llvm/Pass.h"

using namespace llvm;

namespace llvm {
class Function;
}

class NDRange : public FunctionPass {
  void operator=(const NDRange &);
  NDRange(const NDRange &);

public:
  static char ID;
  NDRange();

  virtual bool runOnFunction(Function &function);
  virtual void getAnalysisUsage(AnalysisUsage &au) const;

public:
  InstVector getTids();
  InstVector getGids();
  InstVector getGroupIds();
  InstVector getSizes();
  InstVector getGlobalSizes();
  InstVector getDivergentIds();
  InstVector getTids(int direction);
  InstVector getGids(int direction);
  InstVector getGroupIds(int direction);
  InstVector getSizes(int direction);
  InstVector getGlobalSizes(int direction);
  InstVector getDivergentIds(int direction);
  InstVector getSMemAllocs();

  bool isTid(Instruction *inst);
  bool isTidInDirection(Instruction *inst, int direction);
  std::string getType(Instruction *inst) const;
  bool isCoordinate(Instruction *inst) const;
  bool isSize(Instruction *inst) const;

  int getDirection(Instruction *inst) const;

  bool isGlobal(Instruction *inst) const;
  bool isLocal(Instruction *inst) const;
  bool isGlobalSize(Instruction *inst) const;
  bool isLocalSize(Instruction *inst) const;
  bool isGroupId(Instruction *inst) const;
  bool isGroupsNum(Instruction *inst) const;

  bool isGlobal(Instruction *inst, int direction) const;
  bool isLocal(Instruction *inst, int direction) const;
  bool isGlobalSize(Instruction *inst, int direction) const;
  bool isLocalSize(Instruction *inst, int direction) const;
  bool isGroupId(Instruction *inst, int direction) const;
  bool isGroupsNum(Instruction *inst, int dimension) const;

  Function *getOclFunctionPtr(std::string name) const;
  void registerOclInst(int direction, std::string name, Instruction *inst);
  void unregisterOclInst(int direction, std::string name, Instruction *inst);

  void dump();

public:
  static std::string GET_GLOBAL_ID;
  static std::string GET_LOCAL_ID;
  static std::string GET_GLOBAL_SIZE;
  static std::string GET_LOCAL_SIZE;
  static std::string GET_GROUP_ID;
  static std::string GET_GROUPS_NUMBER;
  static int DIRECTION_NUMBER;

private:
  void init();
  bool isPresentInDirection(Instruction *inst, const std::string &functionName,
                            int direction) const;
  void findOpenCLFunctionCallsByNameAllDirs(std::string calleeName,
                                            Function *caller);

  void getAllOpenCLFunctionPtrs(Function *caller);
  void getExistingOpenCLFunctionPtr(std::string calleeName, Function *caller, std::set<std::string> &unlinked);

private:
  std::map<std::string, Function *> oclFunctionPointers;
  std::vector<std::map<std::string, InstVector>> oclInsts;
};

// Non-member functions.
void findOpenCLFunctionCallsByName(std::string calleeName, Function *caller,
                                   int dimension, InstVector &target);
void findOpenCLFunctionCalls(Function *callee, Function *caller, int dimension,
                             InstVector &target);

#endif
