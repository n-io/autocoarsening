#include <CL/cl.h>

#include "Utils.h"

#include <stdlib.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdexcept>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

#include <pthread.h>
void junk() {
  int i;
  i=pthread_getconcurrency();
};

#define __AXTOR_DEBUG_PRINTXX 1
#define PERFORM_AXTOR_COMPILE 1

std::string compile(std::string &inputFile, const char *options, std::string &optOptions,
                    std::string &outputFile, int seed, bool cacheDependenceAnalysis);
cl_program compileAllCF(std::string &inputFile,
                        const char *options,
			std::string &outputFile,
			int seed,
			unsigned int maxCoarseningFactor,
			cl_context context,
			clCreateProgramWithSourceFunction originalCreateProgramWithSource,
			clBuildProgramFunction originalBuildProgram,
			cl_uint num_devices,
                        const cl_device_id *device_list,
                        void (*pfn_notify)(cl_program, void *),
                        void *user_data);
cl_program compileSingleCF(std::string &inputFile,
                           const char *options,
			   std::string &optOptions,
			   std::string &outputFile,
			   int seed,
			   cl_context context,
			   clCreateProgramWithSourceFunction originalCreateProgramWithSource,
			   clBuildProgramFunction originalBuildProgram,
			   cl_uint num_devices,
                           const cl_device_id *device_list,
                           void (*pfn_notify)(cl_program, void *),
                           void *user_data,
                           bool cacheDependenceAnalysis);

//------------------------------------------------------------------------------
// OpenCL Runtime state data structures.
struct ProgramDesc {
  cl_context context;
  std::string sourceStr;
  cl_program handle;

  ProgramDesc(cl_context context, std::string sourceStr)
      : context(context), sourceStr(sourceStr), handle(0) {}

  ProgramDesc(cl_context context, cl_program handle)
      : context(context), sourceStr(""), handle(handle) {}

  bool isValid() const { return handle != 0; }
  bool isFromBinary() const { return sourceStr.empty(); }
};

typedef std::vector<ProgramDesc *> ProgramDescVec;

struct KernelResources {
  int regs;
  int smem;
  int cmem;
  int cf;
  int direction;
  bool isCacheDependent;
  std::string cdaLog;
  float occupancy;
  int activeThreadsByNumBlocks;
  int activeThreadsBySMem;
  int activeThreadsByRegs;
  int achievableActiveThreads;

  KernelResources(int regs, int smem, int cmem, int cf, int direction, bool isCacheDependent, std::string cdaLog)
      : regs(regs), smem(smem), cmem(cmem), cf(cf), direction(direction), isCacheDependent(isCacheDependent), cdaLog(cdaLog),
        occupancy(0.0), activeThreadsByNumBlocks(0), activeThreadsBySMem(0), activeThreadsByRegs(0), achievableActiveThreads(0) {}
};

struct KernelLaunchConfig {
  int numBlocks;
  int numThreadsPerBlock;
  int gridDim[3];

  KernelLaunchConfig() {}

  KernelLaunchConfig(int numBlocks, int numThreadsPerBlock)
      : numBlocks(numBlocks), numThreadsPerBlock(numThreadsPerBlock) {}
};

//typedef std::map<string, std::vector<KernelResources>> ResourceMap;

// Runtime state state
static ProgramDescVec programs;
static std::map<std::string, std::vector<KernelResources *>> kernelResources;
static std::map<std::string, KernelLaunchConfig *> kernelLaunchConfig;
static std::map<std::string, int> chosenCFs;
//static std::map<std::string, int> kernelRequestedBlocksMap;

// OpenCL functions.
//------------------------------------------------------------------------------

extern "C" cl_kernel clCreateKernel(cl_program program, const char *kernel_name,
                                    cl_int *errcode_ret) {
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);
  clCreateKernelFunction originalCreateKernel;
  *(void **)(&originalCreateKernel) = dlsym(RTLD_NEXT, CL_CREATE_KERNEL_NAME);

  cl_int errorCode;
  cl_kernel kernel =
      originalCreateKernel(desc->handle, kernel_name, &errorCode);
  dumpError(errorCode);
  if (errcode_ret)
    *errcode_ret = errorCode;

  return kernel;
}

//------------------------------------------------------------------------------
extern "C" cl_int clReleaseProgram(cl_program program) {
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);

  clReleaseProgramFunction originalReleaseProgram;
  *(void **)(&originalReleaseProgram) =
      dlsym(RTLD_NEXT, CL_RELEASE_PROGRAM_NAME);

  return dumpError(originalReleaseProgram(desc->handle));
}

//------------------------------------------------------------------------------
extern "C" cl_int clRetainProgram(cl_program program) {
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);

  clRetainProgramFunction originalRetainProgram;
  *(void **)(&originalRetainProgram) = dlsym(RTLD_NEXT, CL_RETAIN_PROGRAM_NAME);

  return dumpError(originalRetainProgram(desc->handle));
}

//------------------------------------------------------------------------------
extern "C" cl_int clGetProgramInfoWithNoTypeCastHack(cl_program program,
                                                     cl_program_info param_name,
                                                     size_t param_value_size, void *param_value,
                                                     size_t *param_value_size_ret) {
  clGetProgramInfoFunction originalGetProgramInfo;
  *(void **)(&originalGetProgramInfo) =
      dlsym(RTLD_NEXT, CL_GET_PROGRAM_INFO_NAME);
  return originalGetProgramInfo(program, param_name, param_value_size,
                                param_value, param_value_size_ret);
}

extern "C" cl_int clGetProgramInfo(cl_program program,
                                   cl_program_info param_name,
                                   size_t param_value_size, void *param_value,
                                   size_t *param_value_size_ret) {
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);

  clGetProgramInfoFunction originalGetProgramInfo;
  *(void **)(&originalGetProgramInfo) =
      dlsym(RTLD_NEXT, CL_GET_PROGRAM_INFO_NAME);
  return originalGetProgramInfo(desc->handle, param_name, param_value_size,
                                param_value, param_value_size_ret);
}

//------------------------------------------------------------------------------
cl_int clGetProgramBuildInfoWithNoTypeCastHack(cl_program program, cl_device_id device,
                                               cl_program_build_info param_name,
                                               size_t param_value_size,
                                               void *param_value,
                                               size_t *param_value_size_ret) {

  clGetProgramBuildInfoFunction originalGetProgramBuildInfo;
  *(void **)(&originalGetProgramBuildInfo) =
      dlsym(RTLD_NEXT, CL_GET_PROGRAM_BUILD_INFO_NAME);
  return dumpError(originalGetProgramBuildInfo(program, device, param_name,
                                               param_value_size, param_value,
                                               param_value_size_ret));
}

extern "C" cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device,
                                        cl_program_build_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);
  
  return clGetProgramBuildInfoWithNoTypeCastHack(desc->handle, device, param_name, param_value_size, param_value, param_value_size_ret); 

}

//------------------------------------------------------------------------------
extern "C" cl_program
clCreateProgramWithBinary(cl_context context, cl_uint num_devices,
                          const cl_device_id *device_list,
                          const size_t *lengths, const unsigned char **binaries,
                          cl_int *binary_status, cl_int *errcode_ret) {
  clCreateProgramWithBinaryFunction originalCreateProgramWithBinary;
  *(void **)(&originalCreateProgramWithBinary) =
      dlsym(RTLD_NEXT, CL_CREATE_PROGRAM_WITH_BINARY_NAME);
  cl_program realHandle = originalCreateProgramWithBinary(
      context, num_devices, device_list, lengths, binaries, binary_status,
      errcode_ret);

  ProgramDesc *desc = new ProgramDesc(context, realHandle);
  programs.push_back(desc);
  return reinterpret_cast<cl_program>(desc);
}

//------------------------------------------------------------------------------
extern "C" cl_program clCreateProgramWithSource(cl_context context,
                                                cl_uint count,
                                                const char **strings,
                                                const size_t *lengths,
                                                cl_int *errcode_ret) {
  std::stringstream buffer;
  for (uint i = 0; i < count; ++i) {
    if (lengths != NULL && lengths[i] > 0) {
      buffer.write(strings[i], lengths[i]);
      buffer << "\n";
    } else {
      buffer << strings[i] << "\n";
    }
  }

  ProgramDesc *desc = new ProgramDesc(context, buffer.str());
  programs.push_back(desc);
  cl_program fakeHandle = reinterpret_cast<cl_program>(desc);

  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return fakeHandle;
}

//------------------------------------------------------------------------------
extern "C" cl_int clBuildProgram(cl_program program, cl_uint num_devices_param,
                                 const cl_device_id *device_list_param,
                                 const char *options,
                                 void (*pfn_notify)(cl_program, void *),
                                 void *user_data) {
  cl_uint num_devices;
  cl_device_id * device_list_temp;
  bool recoverDeviceList = (device_list_param == NULL || num_devices_param == 0);

  if (recoverDeviceList) {
#ifdef __AXTOR_DEBUG_PRINT
    std::cout << "Build program was called with 0 devices or with device_list = null\n";
#endif
    cl_platform_id platform_id[10];
    cl_uint num_platforms = 0;
    clGetPlatformIDs(10, platform_id, &num_platforms);
#ifdef __AXTOR_DEBUG_PRINT
    if (num_platforms > 1) {
      std::cout << "Found " << num_platforms << " platforms\n";
    }
#endif

    device_list_temp = (cl_device_id*)malloc(sizeof(cl_device_id));
    clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 1, device_list_temp, &num_devices);
    if (num_devices > 1 || num_devices == 0) {
      char buffer[50];
#ifdef __AXTOR_DEBUG_PRINT
      sprintf(buffer, "Found %d devices", num_devices);
#endif
      verifyOutputCode(CL_INVALID_DEVICE, buffer);
      exit(CL_INVALID_DEVICE);
    }
  } else {
    num_devices = num_devices_param;
  }

  const cl_device_id * device_list = recoverDeviceList ? device_list_temp : device_list_param;


  srand((unsigned)time(0));
  int seed = rand() % 100000;

  // Get pointer to original function call.
  clBuildProgramFunction originalBuildProgram;
  *(void **)(&originalBuildProgram) = dlsym(RTLD_NEXT, CL_BUILD_PROGRAM_NAME);

  clCreateProgramWithSourceFunction originalCreateProgramWithSource;
  *(void **)(&originalCreateProgramWithSource) =
      dlsym(RTLD_NEXT, CL_CREATE_PROGRAM_WITH_SOURCE_NAME);

  // Get the source file name.
  std::string inputFile = getMangledFileName(OCL_INPUT_FILE, seed);
  std::string outputFile = getMangledFileName(OCL_OUTPUT_FILE, seed);

  // Get the program handle.
  ProgramDesc *desc = reinterpret_cast<ProgramDesc *>(program);
  if (desc->isFromBinary()) {
#ifdef __AXTOR_DEBUG_PRINT
    std::cout << "Running from binary\n";
#endif
    cl_int errorCode;
    errorCode = originalBuildProgram(desc->handle, num_devices, device_list,
                                     "-cl-nv-verbose -cl-nv-opt-level 3", pfn_notify, user_data);
#ifdef __AXTOR_DEBUG_PRINT
    std::cout << "Finished originalBuildProgram" << std::endl;
#endif
    //desc->handle = program;
    verifyOutputCode(errorCode, "Error building the new program");

    return CL_SUCCESS;
  }

  // Dump the program.
  writeFile(inputFile, desc->sourceStr);

  // Compile the program.
  std::string testMaxCoarseningFactor = getEnvString("MAX_COARSENING_FACTOR");
#ifdef __AXTOR_DEBUG_PRINT
  std::cout << "Options: " << (options != NULL ? options : "") << std::endl;
  std::cout << "MaxCoarseningFactor is: " << testMaxCoarseningFactor << std::endl;
#endif
  int maxCoarseningFactor = 0;
  if (!testMaxCoarseningFactor.empty()) {
    maxCoarseningFactor = std::stoi(testMaxCoarseningFactor);
  }
  desc->handle = compileAllCF(inputFile, options, outputFile, seed, maxCoarseningFactor, desc->context, originalCreateProgramWithSource, originalBuildProgram, num_devices, device_list, pfn_notify, user_data);

  cl_bool param_val;
  clGetDeviceInfo(*device_list, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &param_val, NULL);
  cl_ulong buff;
  clGetDeviceInfo(*device_list, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &buff, NULL);
  cl_uint uintparam;
  clGetDeviceInfo(*device_list, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &uintparam, NULL);
#ifdef __AXTOR_DEBUG_PRINT
  std::cout << "Device has unified memory: " << param_val << std::endl;
  std::cout << "Device local mem size: " << buff << std::endl;
  std::cout << "Device global mem cacheline size: " << uintparam << std::endl;
#endif

  return CL_SUCCESS;
}

void parseBuildLog(std::string buildLog, std::string kernelName, int& regs, int& smem, int& cmem) {
  size_t lineStart = buildLog.find("ptxas info", buildLog.find("Function properties for " + kernelName + "\n"));
  size_t lineLen = buildLog.find("\n", lineStart) - lineStart;
  std::string logLine = buildLog.substr(lineStart, lineLen);
  
  size_t regNumEnd = logLine.find("register");
  size_t regNumStart = logLine.find_last_of(" ", regNumEnd - 2) + 1;

  size_t cmemNumEnd = logLine.find("bytes cmem");
  size_t cmemNumStart = logLine.find_last_of(" ", cmemNumEnd - 2) + 1;

  size_t smemNumEnd = logLine.find("bytes smem");
  size_t smemNumStart = logLine.find_last_of(" ", smemNumEnd - 2) + 1;

  std::string regStr = logLine.substr(regNumStart, regNumEnd - 2 - regNumStart + 1);
  std::string smemStr = logLine.substr(smemNumStart, smemNumEnd - 2 - smemNumStart + 1);
  std::string cmemStr = logLine.substr(cmemNumStart, cmemNumEnd - 3 - cmemNumStart + 1);
  
  if (regNumEnd != std::string::npos && !regStr.empty()) {
    regs = std::stoi(regStr);
  }
  if (smemNumEnd != std::string::npos && !smemStr.empty()) {
    smem = std::stoi(smemStr);
  }
  if (cmemNumEnd != std::string::npos && !cmemStr.empty()) {
    cmem = std::stoi(cmemStr);
  }
}

bool parseCacheDependence(const int seed, std::string & cdaLog) {
  std::string inputFile = getMangledFileName(CLR_FILE, seed);
  std::string outputFile = inputFile + ".grepResult";
  std::string searchStr = "No cache line re-use detected, OK to coarsen";
  //std::string cmd = "grep \"" + searchStr + "\" " + cdaFileName + " > " + cdaGrepFileName;
  std::string cmd = "tail -1 " + inputFile + " > " + outputFile;
  system(cmd.c_str());
  size_t resultSize;
  char* resultPtr = readFile(outputFile.c_str(), &resultSize);
  cdaLog.assign(resultPtr, resultSize > 1 && resultPtr[resultSize-2] == '\n' ? resultSize - 2 : resultSize);
#ifndef __AXTOR_DEBUG_PRINT
  std::string removeString = "rm " + inputFile + " && rm " + outputFile;
  system(removeString.c_str()); //retain intermediate files in debug mode
#endif
  return cdaLog.find(searchStr) == std::string::npos;
}

cl_program compileAllCF(std::string &inputFile,
                        const char *options,
			std::string &outputFile,
			int seed,
			unsigned int maxCoarseningFactor,
			cl_context context,
			clCreateProgramWithSourceFunction originalCreateProgramWithSource,
			clBuildProgramFunction originalBuildProgram,
			cl_uint num_devices,
                        const cl_device_id *device_list,
                        void (*pfn_notify)(cl_program, void *),
                        void *user_data)
{
  std::string optOptionsOriginal = getEnvString(OCL_COMPILER_OPTIONS);
  // std::cout << "Entering compileAllCF with OCL_COMPILER_OPTIONS=" << optOptionsOriginal << std::endl;
  std::string kernelName = getEnvString(TC_KERNEL_NAME);
  
  if (maxCoarseningFactor > 0) {
    
    const std::string cfFlag = " -coarsening-factor ";
    const std::string cdFlag = " -coarsening-direction ";
    size_t cfStart = optOptionsOriginal.find(cfFlag);
    size_t cdStart = optOptionsOriginal.find(cdFlag);
    size_t cdArgStartPos = optOptionsOriginal.find_first_not_of(" \t\n\r\\", cdStart + cdFlag.length());
    size_t cdArgEndPos = optOptionsOriginal.find(" ", cdArgStartPos);
    int coarseningDirection = std::stoi(optOptionsOriginal.substr(cdArgStartPos, cdArgEndPos-cdArgStartPos));
    
    std::string verboseOptions(options != NULL ? options : "");
    if (verboseOptions.find("-cl-nv-verbose") == std::string::npos) {
      verboseOptions.append(" -cl-nv-verbose");
    }
    std::string optOptions = optOptionsOriginal;
    size_t buildLogSize;
#ifdef __AXTOR_DEBUG_PRINT
    std::cout << "Entering compile function for coarsening kernel " << kernelName << " with coarsening direction " << coarseningDirection << " and build string: " << optOptionsOriginal << std::endl;
#endif
    for (unsigned int coarseningFactor = 1; coarseningFactor <= maxCoarseningFactor; coarseningFactor <<= 1) {
      bool cacheDependenceAnalysis = coarseningFactor == 1;
      // setup
      size_t argPos = optOptions.find_first_not_of(" \t\n\r\\", cfStart + cfFlag.length());
      size_t argEndPos = optOptions.find(" ", argPos);
      optOptions.replace(argPos, argEndPos-argPos, std::to_string(coarseningFactor));

#ifdef __AXTOR_DEBUG_PRINT
      std::cout << "For CF = " << coarseningFactor << " new build string is: " << optOptions << std::endl;
#endif
      // set up buildString (optOptions)
      // call compile
      cl_program program;
      try {
        program = compileSingleCF(inputFile, verboseOptions.c_str(), optOptions, outputFile, seed, context, originalCreateProgramWithSource, originalBuildProgram,
                                             num_devices, device_list, pfn_notify, user_data, cacheDependenceAnalysis);
      } catch (int e) {
        std::cout << "Caught exception when compiling for cf " << coarseningFactor << "\n";
        break;
      }
      cl_int errorCode = clGetProgramBuildInfoWithNoTypeCastHack(program, *device_list, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
      //cl_build_status sts;
      //cl_int errorCode = clGetProgramBuildInfoWithNoTypeCastHack(program, NULL, CL_PROGRAM_BUILD_STATUS, 0, NULL, &buildLogSize);
      verifyOutputCode(errorCode, "Error querying the build log size");
      char* buildLogData = new char[buildLogSize+1];
      errorCode = clGetProgramBuildInfoWithNoTypeCastHack(program, *device_list, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLogData, NULL);

      verifyOutputCode(errorCode, "Error querying the build log");
      std::string buildLog(buildLogData);
      delete [] buildLogData;
#ifdef __AXTOR_DEBUG_PRINT
      std::cout << "Build log size is " << buildLogSize << std::endl;
      std::cout << "Build log:\n" << buildLog << std::endl;
#endif

      // test whether kernel to be tested is in this file (program might keep kernels in separate .cl files)
      size_t buildLogKernelName = buildLog.find("Function properties for " + kernelName);
      if (buildLogKernelName != std::string::npos) {
	int regs = 0;
	int smem = 0;
	int cmem = 0;
	parseBuildLog(buildLog, kernelName, regs, smem, cmem);
	std::string cdaLog;
	bool isCacheDependent = cacheDependenceAnalysis ? parseCacheDependence(seed, cdaLog) : false;

#ifdef __AXTOR_DEBUG_PRINT
	std::cout << "Kernel " << kernelName << " with cf " << coarseningFactor << ": " << regs << " regs " << smem << " smem " << cmem << " cmem" << std::endl;
	std::cout << "     --------------------------------    \n";
#endif
	kernelResources[kernelName].push_back(new KernelResources(regs, smem, cmem, coarseningFactor, coarseningDirection, isCacheDependent, cdaLog));
      }
    }
  }

  // re-set original build string
  // call compile and return its output
#ifdef __AXTOR_DEBUG_PRINT
  std::cout << "Now running default compile with build string: " << optOptionsOriginal << std::endl;
#endif

  std::string verboseOptions(options != NULL ? options : "");
  if (verboseOptions.find("-cl-nv-verbose") == std::string::npos) {
    verboseOptions.append(" -cl-nv-verbose");
  }
  size_t buildLogSize;
  cl_program result = compileSingleCF(inputFile, verboseOptions.c_str()/*options*/, optOptionsOriginal, outputFile, seed, context, originalCreateProgramWithSource, originalBuildProgram,
                                      num_devices, device_list, pfn_notify, user_data, false);

  cl_int errorCode = clGetProgramBuildInfoWithNoTypeCastHack(result, *device_list, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
  verifyOutputCode(errorCode, "Error querying the build log size");
  char* buildLogData = new char[buildLogSize+1];
  errorCode = clGetProgramBuildInfoWithNoTypeCastHack(result, *device_list, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLogData, NULL);

  verifyOutputCode(errorCode, "Error querying the build log");
  std::string buildLog(buildLogData);
#ifdef __AXTOR_DEBUG_PRINT
  std::cout << "Build log: " << buildLog << std::endl;
#endif

  size_t buildLogKernelName = buildLog.find("Function properties for " + kernelName);
  if (buildLogKernelName != std::string::npos) {
    int regs = 0;
    int smem = 0;
    int cmem = 0;
    parseBuildLog(buildLog, kernelName, regs, smem, cmem);
    if (kernelResources[kernelName].empty()) {
      // else, it already exists in the map
      // coarsening factor and direction would have to be parsed,
      // but the maths in calculateOccupancies() will work if cf is set to 1
      kernelResources[kernelName].push_back(new KernelResources(regs, smem, cmem, 1, 1, false, ""));
    }
  }

  size_t bin_sz;
  errorCode = clGetProgramInfoWithNoTypeCastHack(result, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_sz, NULL);
  // Read binary (PTX file) to memory buffer
  char *bin = (char *)malloc(bin_sz);
  errorCode = clGetProgramInfoWithNoTypeCastHack(result, CL_PROGRAM_BINARIES, sizeof(char *), &bin, NULL);
  //unsigned char* bin = new unsigned char[bin_sz+1];
  //errorCode = clGetProgramInfoWithNoTypeCastHack(result, CL_PROGRAM_BINARIES, bin_sz, bin, NULL);

#ifdef __AXTOR_DEBUG_PRINT
  FILE* fp;
  fp = fopen("/tmp/dump.ptx", "wb");
  fwrite(bin, sizeof(char), bin_sz, fp);
  fclose(fp);
#endif

  delete [] buildLogData;
  //delete [] bin;
  free(bin);

  return result;
}

cl_program compileSingleCF(std::string &inputFile,
                           const char *options,
			   std::string &optOptions,
			   std::string &outputFile,
			   int seed,
			   cl_context context,
			   clCreateProgramWithSourceFunction originalCreateProgramWithSource,
			   clBuildProgramFunction originalBuildProgram,
			   cl_uint num_devices,
                           const cl_device_id *device_list,
                           void (*pfn_notify)(cl_program, void *),
                           void *user_data,
                           bool cacheDependenceAnalysis)
{
  std::string oclOptions = compile(inputFile, options, optOptions, outputFile, seed, cacheDependenceAnalysis);

  // Create the new program.
  size_t outputSize;
  const char *outputProgram;
  cl_int errorCode;

#ifdef PERFORM_AXTOR_COMPILE
  outputProgram = readFile(outputFile.c_str(), &outputSize);
#else
  outputProgram = readFile(inputFile.c_str(), &outputSize); // read from file not processed by axtor
#endif
 
  cl_program program = originalCreateProgramWithSource(
      context, 1, (const char **)&outputProgram, &outputSize, &errorCode);
  verifyOutputCode(errorCode, "Error creating the new program");

  // Build the new program.
  errorCode = originalBuildProgram(program, num_devices, device_list,
                                   oclOptions.c_str(), pfn_notify, user_data);
  verifyOutputCode(errorCode, "Error building the new program");
  delete[] outputProgram;
  return program;
}

//------------------------------------------------------------------------------
std::string compile(std::string &inputFile, const char *options, std::string &optOptions,
                    std::string &outputFile, int seed, bool cacheDependenceAnalysis) {
  // Compile the program.

  if (options == NULL)
    options = "";

#ifdef __AXTOR_DEBUG_PRINT
  std::cout << "Options in compile: " << options << std::endl;
#endif

  std::string clangOptions(options);
  std::string oclOptions;

  splitCompilerOptions(clangOptions, oclOptions);

#ifdef __AXTOR_DEBUG_PRINT
  std::cout << "clangOptions: " << clangOptions << std::endl << "optOptions: " << optOptions << std::endl << "oclOptions: " << oclOptions << std::endl;
#endif
#ifdef PERFORM_AXTOR_COMPILE
  if (compileWithAxtor(inputFile, clangOptions, optOptions, outputFile, seed, cacheDependenceAnalysis)) { //TODO: comment out
    std::cout << "Error compiling with axtor\n";
    exit(1);
  }
#endif

  return oclOptions;
}

//------------------------------------------------------------------------------
inline std::string getPotentialLimitingFactor(int activeThreadsByRegs, int activeThreadsBySMem, int activeThreadsByNumBlocks) {
  if (activeThreadsByNumBlocks < activeThreadsByRegs || activeThreadsByNumBlocks < activeThreadsBySMem) return "blocks per SMX";
  if (activeThreadsByRegs == activeThreadsBySMem) return "regs + smem";
  if (activeThreadsByRegs > activeThreadsBySMem) return "smem";
  return "regs";
}

//------------------------------------------------------------------------------
void calculateOccupancies(cl_uint work_dim, const size_t *global_work_size, const size_t *local_work_size, std::string kernelName) {
  std::vector<KernelResources*> coarsenings = kernelResources[kernelName];
  if (coarsenings.empty()) {
    std::cout << "No coarsenings found for kernel " << kernelName << std::endl;
    return;
  }

  KernelLaunchConfig * klc = new KernelLaunchConfig();
  //int gridDim[work_dim];
  int originalThreadsPerBlock = 1;
  int numBlocks = 1;
  // calculate block sizes and print what we know about input dimension
  for (int i = 0; i < work_dim; i++) {
    klc->gridDim[i] = global_work_size[i] / local_work_size[i];
    numBlocks *= klc->gridDim[i];
  }
  for (int i = 0; i < work_dim; i++) {
    originalThreadsPerBlock *= local_work_size[i];
  }
  klc->numBlocks = numBlocks;
  klc->numThreadsPerBlock = originalThreadsPerBlock;
  kernelLaunchConfig.emplace(kernelName, klc);

  std::string applyThreadLevelCoarseningStr = getEnvString("THREAD_LEVEL_COARSENING");
  bool isThreadLevelCoarsening = !applyThreadLevelCoarseningStr.empty();


  // set up device
  //const int computeUnits = stoi(getEnvString("ARCH_COMPUTE_UNITS", "15")); // not needed here
  const int maxActiveThreadsPerSMX = stoi(getEnvString("ARCH_ACTIVE_THREADS_PER_CU", "2048"));
  const int maxBlocksPerSMX = stoi(getEnvString("ARCH_GROUPS_PER_CU", "16"));
  const int maxRegsPerSMX = stoi(getEnvString("ARCH_REGS_PER_CU", "65536"));
  const int maxSMemPerSMX = stoi(getEnvString("ARCH_SMEM_PER_CU", "49152"));

  for (std::vector<KernelResources*>::reverse_iterator coarsening = coarsenings.rbegin(); coarsening != coarsenings.rend(); coarsening++) {
    int threadsPerBlock = isThreadLevelCoarsening ? (originalThreadsPerBlock / (*coarsening)->cf) : originalThreadsPerBlock;
    int theoreticBlocksPerSMX = std::min(maxBlocksPerSMX, maxActiveThreadsPerSMX / threadsPerBlock);
    (*coarsening)->activeThreadsByNumBlocks = theoreticBlocksPerSMX * threadsPerBlock;
    (*coarsening)->activeThreadsBySMem = std::min(maxSMemPerSMX / ((*coarsening)->smem > 0 ? (*coarsening)->smem : 1), theoreticBlocksPerSMX) * threadsPerBlock;
    (*coarsening)->activeThreadsByRegs = std::min(maxActiveThreadsPerSMX, ((maxRegsPerSMX / (*coarsening)->regs) / threadsPerBlock) * threadsPerBlock);
    (*coarsening)->achievableActiveThreads = std::min(std::min((*coarsening)->activeThreadsByRegs, (*coarsening)->activeThreadsByNumBlocks), (*coarsening)->activeThreadsBySMem);
    (*coarsening)->occupancy = ((*coarsening)->achievableActiveThreads * 100.0 / maxActiveThreadsPerSMX);

  }
  
  KernelResources *kr = *coarsenings.begin();
#ifdef __AXTOR_DEBUG_PRINT
  std::cout << "Kernel " << kernelName << " occupancy is " << kr->occupancy << "\n";
#endif

  std::string oredSetup = getEnvString("OCCUPANCY_REDUCTION_SETUP");
  if (!oredSetup.empty()) {
    std::cout << "Dumping occupancy reduction setup info to " << oredSetup << "\n";
    std::stringstream oredSetupInfo;
    oredSetupInfo << "smem " << kr->smem << "\nblocksize " << originalThreadsPerBlock << "\noccupancy " << (kr->occupancy / 100) << "\n";
    writeFile(oredSetup, oredSetupInfo.str());
  }

}

//------------------------------------------------------------------------------
void applyCoarseningModel(std::string kernelName) {
  std::vector<KernelResources*> coarsenings = kernelResources[kernelName];
  if (coarsenings.empty()) {
    std::cout << "No coarsenings found for kernel " << kernelName << std::endl;
    return;
  }

  if (kernelLaunchConfig.count(kernelName) == 0) {
    std::cout << "Did not store the number of requested blocks for kernel " << kernelName << std::endl;
  }

  // set up device
  const int computeUnits = stoi(getEnvString("ARCH_COMPUTE_UNITS", "15"));
  const int maxActiveThreadsPerSMX = stoi(getEnvString("ARCH_ACTIVE_THREADS_PER_CU", "2048"));
  const int maxBlocksPerSMX = stoi(getEnvString("ARCH_GROUPS_PER_CU", "16"));
  const int maxRegsPerSMX = stoi(getEnvString("ARCH_REGS_PER_CU", "65536"));
  const int maxSMemPerSMX = stoi(getEnvString("ARCH_SMEM_PER_CU", "49152"));

  int numBlocks = kernelLaunchConfig[kernelName]->numBlocks;
  int originalThreadsPerBlock = kernelLaunchConfig[kernelName]->numThreadsPerBlock;
  int maxExecutedBlocksPerRound = std::min(maxBlocksPerSMX, maxActiveThreadsPerSMX / originalThreadsPerBlock) * computeUnits;
  int maxCFByInputSize = numBlocks < maxExecutedBlocksPerRound ? 1 : numBlocks / maxExecutedBlocksPerRound;
  unsigned int maxCFByInputDivisibility = 1;
  const unsigned int coarseningDirection = coarsenings.front()->direction; // TODO: this assumes constant direction among all coarsened kernels
  while (kernelLaunchConfig[kernelName]->gridDim[coarseningDirection] > maxCFByInputDivisibility && kernelLaunchConfig[kernelName]->gridDim[coarseningDirection] % maxCFByInputDivisibility == 0) {
    maxCFByInputDivisibility <<= 1;
  }

  int chosenCF = 0;
  int chosenCFMaxActiveThreads = 0;
  std::string limitingFactor;
  std::string prevLimitingFactor;
  int maxCFByInputSize_maxActiveThreads = 0;
  int maxCFByInputDivisibility_maxActiveThreads = 0;

  std::cout << "Found the following coarsenings for kernel " << kernelName << ": " << std::endl;
  for (std::vector<KernelResources*>::reverse_iterator coarsening = coarsenings.rbegin(); coarsening != coarsenings.rend(); coarsening++) {
    int achievableActiveThreads = (*coarsening)->achievableActiveThreads;
    
    std::cout << (*coarsening)->cf << ": " << (*coarsening)->regs << " regs " << (*coarsening)->smem << " smem " << (*coarsening)->cmem << " cmem";// << std::endl;
    std::cout << "\tactive threads by regs: " << (*coarsening)->activeThreadsByRegs << ", block limit: " << (*coarsening)->activeThreadsByNumBlocks
              << ", smem: " << (*coarsening)->activeThreadsBySMem
              << ", occupancy => " << ((*coarsening)->occupancy) << "%" << std::endl;
    
    if ((*coarsening)->cf == maxCFByInputSize) {
      maxCFByInputSize_maxActiveThreads = achievableActiveThreads;
    }
    if ((*coarsening)->cf == maxCFByInputDivisibility) {
      maxCFByInputDivisibility_maxActiveThreads = achievableActiveThreads;
    }
    
    if (achievableActiveThreads > chosenCFMaxActiveThreads) {
      chosenCF = (*coarsening)->cf;
      chosenCFMaxActiveThreads = achievableActiveThreads;
      limitingFactor = prevLimitingFactor.empty() ? getPotentialLimitingFactor((*coarsening)->activeThreadsByRegs, (*coarsening)->activeThreadsBySMem, (*coarsening)->activeThreadsByNumBlocks) : prevLimitingFactor;
    }
    prevLimitingFactor = getPotentialLimitingFactor((*coarsening)->activeThreadsByRegs, (*coarsening)->activeThreadsBySMem, (*coarsening)->activeThreadsByNumBlocks);
  }
  if (chosenCF > maxCFByInputSize) {
    chosenCF = maxCFByInputSize;
    chosenCFMaxActiveThreads = maxCFByInputSize_maxActiveThreads;
    limitingFactor = "input size (" + std::to_string(numBlocks) + " blocks)";
  } else if (chosenCF > maxCFByInputDivisibility) {
    chosenCF = maxCFByInputDivisibility;
    chosenCFMaxActiveThreads = maxCFByInputDivisibility_maxActiveThreads;
    limitingFactor = "input divisibility";
  }
  int theoreticalCF = chosenCF;
  if (coarsenings.front()->cf == 1 && coarsenings.front()->isCacheDependent) {
    limitingFactor = "cache line re-use (" + coarsenings.front()->cdaLog + ")";
    chosenCF = 1;
  }
  std::cout << "Program has " << (coarsenings.front()->isCacheDependent ? "" : "no ") << "cache line re-use" << std::endl;
  std::cout << "Model prediction for kernel//blocks//theoretical cf//chosen cf//dir//limiting factor: " << kernelName << "\t" << numBlocks << "\t" << theoreticalCF << "\t" << chosenCF << "\t" << coarseningDirection << "\t" << limitingFactor << std::endl;
  chosenCFs[kernelName] = chosenCF;
}

cl_int clEnqueueNDRangeKernel(
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {

  // Setup the event to measure the kernel execution time.
  bool isEventNull = false;
  if (event == NULL) {
    isEventNull = true;
    event = new cl_event();
  }

  std::string kernelName = getKernelName(kernel);
  std::string envKernelName = getEnvString("TC_KERNEL_NAME");

  size_t *newGlobalSize = new size_t[work_dim];
  size_t *newLocalSize = new size_t[work_dim];
  size_t *real_local_work_size = new size_t[work_dim];

  // handle the case that local_work_size is NULL - take a best guess
  if (local_work_size == NULL) {
    switch (work_dim) {
      case 1: real_local_work_size[0] = 256;
	      break;
      case 2: real_local_work_size[0] = 16;
	      real_local_work_size[1] = 16;
	      break;
      case 3: real_local_work_size[0] = 16;
	      real_local_work_size[1] = 8;
	      real_local_work_size[2] = 8;
    }
  } else {
    memcpy(real_local_work_size, local_work_size, work_dim * sizeof(size_t));
  }

  if (kernelName != envKernelName) {
#ifdef __AXTOR_DEBUG_PRINT
    std::cout << "No coarsening for: " << kernelName << "\n";
    std::cout << "gws " << work_dim << " " << global_work_size[0] << "\n";
#endif
    memcpy(newGlobalSize, global_work_size, work_dim * sizeof(size_t));
    memcpy(newLocalSize, real_local_work_size, work_dim * sizeof(size_t));
  } else {
    calculateOccupancies(work_dim, global_work_size, real_local_work_size, kernelName);
    std::string testMaxCoarseningFactor = getEnvString("MAX_COARSENING_FACTOR");
    int maxCoarseningFactor = 0;
    if (!testMaxCoarseningFactor.empty()) {
      maxCoarseningFactor = std::stoi(testMaxCoarseningFactor);
    }
    if (maxCoarseningFactor > 0) {
      applyCoarseningModel(kernelName);
      if (chosenCFs.count(kernelName) > 0) {
        // select coarsening factor chosen by model prediction
        setenv("CF_OVERRIDE", std::to_string(chosenCFs[kernelName]).c_str(), 1);
      }
    }
    bool NDRangeResult =
        computeNDRangeDim(work_dim, global_work_size, real_local_work_size,
                          newGlobalSize, newLocalSize);

    if (NDRangeResult == false) {
      if (memcmp(global_work_size, newGlobalSize, work_dim * sizeof(size_t)) ==
          0) {
        newLocalSize = NULL;
      } else {
        std::cout << "Cannot apply coarsening when local work size is null\n";
        return 1;
      }
    }
  }

  std::string repetitionsString = getEnvString(OCL_REPETITIONS);
  unsigned int repetitions;
  if (repetitionsString != "")
    std::istringstream(repetitionsString) >> repetitions;
  else
    repetitions = 1;

  enqueueKernel(command_queue, kernel, work_dim, global_work_offset,
                newGlobalSize, newLocalSize, num_events_in_wait_list,
                event_wait_list, event, repetitions, kernelName);

  if (isEventNull) {
    clReleaseEvent(*event);
    delete event;
    event = NULL;
  }

  delete[] newGlobalSize;
  delete[] newLocalSize;
  delete[] real_local_work_size;

  return CL_SUCCESS;
}

//------------------------------------------------------------------------------
cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device,
                                      cl_command_queue_properties properties,
                                      cl_int *errcode_ret) {
  // Get pointer to original function calls.
  clCreateCommandQueueFunction originalclCreateCommandQueue;
  *(void **)(&originalclCreateCommandQueue) =
      dlsym(RTLD_NEXT, CL_CREATE_COMMAND_QUEUE_NAME);

  properties = properties | CL_QUEUE_PROFILING_ENABLE;

  return originalclCreateCommandQueue(context, device, properties, errcode_ret);
}
