#! /usr/bin/python

import itertools;
import os;
import subprocess;
import time;
import sys;

originalTests = {
#"memset/memset" : ["memset1", "memset2"], 
"memset/memset" : ["memset2D"],
"memcpy/memcpy" : ["rmrrmw", "cmrcmw"],
"mm/mm" : ["mm"], 
"mt/mt" : ["mt"],
"mv/mv" : ["MatVecMulUncoalesced0", "MatVecMulUncoalesced1", "MatVecMulCoalesced0"],
"divRegion/divRegion" : ["divRegion"], 
"polybench/OpenCL/2DCONV/2DCONV" : ["Convolution2D_kernel"],
"polybench/OpenCL/2MM/2MM" : ["mm2_kernel1"],
"polybench/OpenCL/3DCONV/3DCONV" : ["Convolution3D_kernel"],
"polybench/OpenCL/3MM/3MM" : ["mm3_kernel1"],
"polybench/OpenCL/ATAX/ATAX" : ["atax_kernel1", "atax_kernel2"],
#"polybench/OpenCL/ATAX/ATAX" : ["atax_kernel1"],
"polybench/OpenCL/BICG/BICG" : ["bicgKernel1"],
"polybench/OpenCL/CORR/CORR" : ["mean_kernel", "std_kernel", "reduce_kernel"],
"polybench/OpenCL/COVAR/COVAR" : ["mean_kernel", "reduce_kernel", "covar_kernel"],
"polybench/OpenCL/FDTD-2D/FDTD-2D" : ["fdtd_kernel1", "fdtd_kernel2", "fdtd_kernel3"],
"polybench/OpenCL/GEMM/GEMM" : ["gemm"],
"polybench/OpenCL/GESUMMV/GESUMMV" : ["gesummv_kernel"],
"polybench/OpenCL/GRAMSCHM/GRAMSCHM" : ["gramschmidt_kernel1", "gramschmidt_kernel2", "gramschmidt_kernel3"],
"polybench/OpenCL/MVT/MVT" : ["mvt_kernel1"],
"polybench/OpenCL/SYR2K/SYR2K" : ["syr2k_kernel"],
"polybench/OpenCL/SYRK/SYRK" : ["syrk_kernel"]
};

### Add configs as you need

kepler = {
"computeUnits" : "15",
"maxActiveThreadsPerCU" : "2048",
"maxGroupsPerCU" : "16",
"maxRegsPerCU" : "65536",
"maxSMemPerCU" : "49152",
"maxSMemPerBlock" : "49152"
};

maxwell = {
"computeUnits" : "24",
"maxActiveThreadsPerCU" : "2048",
"maxGroupsPerCU" : "32",
"maxRegsPerCU" : "65536",
"maxSMemPerCU" : "98304",
"maxSMemPerBlock" : "49152"
};

pascal = {
"computeUnits" : "20",
"maxActiveThreadsPerCU" : "2048",
"maxGroupsPerCU" : "32",
"maxRegsPerCU" : "65536",
"maxSMemPerCU" : "98304",
"maxSMemPerBlock" : "49152"
};

###  Check that the following paths are set correctly

CLANG = "clang";
OPT = "opt";
LIB_THRUD = "/data/build/autocoarsening/thrud/lib/libThrud.so";
OCL_HEADER = "/data/code/autocoarsening/thrud/include/opencl_spir.h";
OPTIMIZATION = "-O3";
LD_PRELOAD = "/data/build/autocoarsening/opencl_tools/function_overload/libaxtorwrapper.so";
PREFIX = "/data/build/autocoarsening/tests";

### The following should not need changing

TC_COMPILE_LINE = "-mem2reg -load " + LIB_THRUD + " -structurizecfg -instnamer -be -tc -coarsening-factor %s -coarsening-direction %s -coarsening-stride %s -div-region-mgt classic -kernel-name %s -simplifycfg -loop-instsimplify -early-cse -load-combine -licm " + OPTIMIZATION;
CLR_OPTIONS = "-load " + LIB_THRUD + " -assumeRestrictArgs -domtree -gvn -basicaa -loop-simplify -indvars -load-combine -early-cse -clr -kernel-name %s -warp-size %s -cache-line-size %s";
ORED_OPTIONS = "-load " + LIB_THRUD + " -ored -kernel-name %s -shmem %s";
#COMPUTE_CACHE = "~/.nv/ComputeCache";
ORED_TMP_FILE = "/tmp/%s.txt";

### Modify these to control execution behaviour

THREAD_LEVEL_COARSENING = False;         # False for block-level coarsening, True for thread-level coarsening
OCCUPANCY_REDUCTION = False;             # experimental
tests = originalTests;
arch = pascal;                           # use this if you have multiple GPUs and only want to run on one
device = "1" if arch == kepler else "0";

applyModel = len(sys.argv) > 1 and sys.argv[1] == "APPLY_COARSENING_MODEL"  # pass this arg to this script to run with coarsening model


#-------------------------------------------------------------------------------
#def printRed(message):
#  print "\033[1;31m%s\033[1;m" % message

#-------------------------------------------------------------------------------
#def printGreen(message):
#  print "\033[1;32m%s\033[1;m" % message;

#-------------------------------------------------------------------------------
def runCommand(arguments):
  WAITING_TIME = 3000;
  #delProcess = subprocess.call(["rm", "-rf", COMPUTE_CACHE]);
  workingDir = None

  # use this for debugging
  # arguments = ["gdb"] + arguments;

  runProcess = subprocess.Popen(arguments, cwd=workingDir,
                                           #stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT);
                                           #stderr=subprocess.PIPE);
  runPid = runProcess.pid;
  counter = 0;
  returnCode = None;

  #outs, errs = runProcess.communicate();
  #returnCode = runProcess.poll();

  # Manage the case in which the run hangs.
  while(counter < WAITING_TIME and returnCode == None):
    counter += 1;
    time.sleep(1);
    returnCode = runProcess.poll();

  if(returnCode == None):
    runProcess.kill();
    return (-1, "", "Time expired!");

  commandOutput = runProcess.communicate();

  #print(commandOutput[0]);
  #print(commandOutput[1]);

  return (returnCode, commandOutput[0], commandOutput[1]);

#-------------------------------------------------------------------------------
def runTest(command, kernelName, warpSize, cacheLineSize, ored, oredStr, cd, cf, st):
  kernelConfig = str.split(kernelName, ':');
  kernelName = kernelConfig[0];
  cd = "0" if (len(kernelConfig) <= 1) else kernelConfig[1];

  # TODO this needs to change depending on test suite used
  command = [os.path.join(PREFIX, command), kernelName];
    

  os.environ["OCL_HEADER"] = OCL_HEADER;
  os.environ["TC_KERNEL_NAME"] = kernelName;
  os.environ["LD_PRELOAD"] = LD_PRELOAD;
  os.environ["OCL_COMPILER_OPTIONS"] = TC_COMPILE_LINE % \
    (cf, cd, st, kernelName);
  os.environ["CLR_OPTIONS"] = CLR_OPTIONS % (kernelName, warpSize, cacheLineSize);

  if (OCCUPANCY_REDUCTION):
    os.environ["OCCUPANCY_REDUCTION"] = ORED_OPTIONS % (kernelName, ored);
  if ("OCCUPANCY_REDUCTION_SETUP" in os.environ):
    del(os.environ["OCCUPANCY_REDUCTION"]);
    # this also deletes definition from previous iteration

  if (THREAD_LEVEL_COARSENING):
    os.environ["THREAD_LEVEL_COARSENING"] = "true";

  # set architectural parameters for model
  os.environ["ARCH_COMPUTE_UNITS"] = arch["computeUnits"];
  os.environ["ARCH_ACTIVE_THREADS_PER_CU"] = arch["maxActiveThreadsPerCU"];
  os.environ["ARCH_GROUPS_PER_CU"] = arch["maxGroupsPerCU"];
  os.environ["ARCH_REGS_PER_CU"] = arch["maxRegsPerCU"];
  os.environ["ARCH_SMEM_PER_CU"] = arch["maxSMemPerCU"];
  os.environ["CUDA_VISIBLE_DEVICES"] = device;
  os.environ["CUDA_CACHE_DISABLE"] = "1";

  print(command);
  sys.stdout.flush();
  result = runCommand(command);
  if (not oredStr):
    # print "oredStr not found, attempting to read"
    fname = ORED_TMP_FILE % kernelName;
    oredStr = "0/0@0"
    if (os.path.isfile(fname)):
      with open(fname, 'r') as infile:
	ignore, ignore_existingSMem = map(str.strip, next(infile).split(' ', 1))
	ignore, threadsPerBlock = map(str.strip, next(infile).split(' ', 1))
	ignore, occupancy = map(str.strip, next(infile).split(' ', 1))
      print "Parsed ", threadsPerBlock, ", and ", occupancy
      blocksPerSM = int(float(occupancy) * int(arch["maxActiveThreadsPerCU"]) / int(threadsPerBlock))
      oredStr = str(blocksPerSM) + "/" + str(blocksPerSM) + "@" + threadsPerBlock


  if(result[0] == 0):
    print(" ".join([kernelName, cd, cf, st, oredStr, "Ok!"]));
    sys.stdout.flush();
    return 0;
  else:
    print(result[2]);
    print(" ".join([kernelName, cd, cf, st, oredStr, "Failure"]));
    sys.stdout.flush();
    return 1;

def main():
  directions = ["0"];
  if (applyModel) :
    factors = ["1"];
    os.environ["MAX_COARSENING_FACTOR"] = "32";
  else:
    factors = ["1", "2", "4", "8", "16", "32"];
  strides = ["32"];#, "2", "32"];

  warpSize = "32";
  cacheLineSize = "32";

  configs = itertools.product(directions, factors, strides);
  configs = [x for x in configs];

  counter = 0;
  failures = 0;

  for test in tests:
    kernels = tests[test];
    for kernel in kernels:
      for config in configs:
        if (OCCUPANCY_REDUCTION):
	  oredTmpFile = ORED_TMP_FILE % kernel;
	  os.environ["OCCUPANCY_REDUCTION_SETUP"] = oredTmpFile;
        failure = runTest(test, kernel, warpSize, cacheLineSize, "0", None, *config);
        failures += failure;
        counter += 1;
	if (OCCUPANCY_REDUCTION):
	  del(os.environ["OCCUPANCY_REDUCTION_SETUP"])

	  print "attempting to parse ", oredTmpFile;
	  existingSMem = "0"
	  threadsPerBlock = "512"
	  occupancy = "100"
          if (os.path.isfile(oredTmpFile)):
	    with open(oredTmpFile, 'r') as infile:
	      ignore, existingSMem = map(str.strip, next(infile).split(' ', 1))
	      ignore, threadsPerBlock = map(str.strip, next(infile).split(' ', 1))
	      ignore, occupancy = map(str.strip, next(infile).split(' ', 1))
	    print "Parsed ", int(existingSMem), ", ", threadsPerBlock, ", and ", occupancy
	    blocksPerSM = int(float(occupancy) * int(arch["maxActiveThreadsPerCU"]) / int(threadsPerBlock))
	    print "blocksPerSM ", blocksPerSM
	    minBlocksPerSM = (int(arch["maxSMemPerCU"]) // int(arch["maxSMemPerBlock"])) - 1
	    if (minBlocksPerSM >= blocksPerSM):
	      minBlocksPerSM = 0

	    for blocks in range(blocksPerSM - 1, minBlocksPerSM, -1):
	      additionalSMem = ( int(arch["maxSMemPerCU"]) - (blocks + 1) * int(existingSMem) ) // (blocks + 1) + 1
	      print "Additional smem:", additionalSMem
	      print "blocks: ", blocks
	      print "existing smem", existingSMem
	      failure = runTest(test, kernel, warpSize, cacheLineSize, str(additionalSMem), str(blocks) + "/" + str(blocksPerSM) + "@" + threadsPerBlock, *config);
	      failures += failure;
	      counter += 1;

  print("#################################");
  print(str(failures) + " failures out of " + str(counter));

main();
