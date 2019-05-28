LLVM Thread Coarsening Pass for OpenCL
======================================


Disclaimer
----------

This repository is an extension of [Alberto Magni's][email/alberto] [thread coarsening pass][www/originalCoarsening].
The extension adds safe, fully automated coarsening factor selection for NVIDIA GPUs.

All rights of the original coarsening pass, as retained in the initial commit, remain with the author. All further commits are our own work.

Content
-------

* opencl\_tools: Collection of support libraries for opencl
                 1. bench_support: library for modifying OpenCL program parameters from command line. 
                 2. function_overload: library to overload standard OpenCL calls to measure kernel execution time and change the compilation pipeline.
                 3. opencl_wrapper: C++ wrapper library for OpenCL.

* thurd: [LLVM Project][www/llvmProject] implementing the OpenCL coarsening pass. 

* tests: Set of OpenCL programs to test the functionality of the coarsening pass. 
          These programs use opencl_tools and require an OpenCL device to run. 

Prerequisited
-------------

Current version of thrud has been tested on the following version of LLVM and clang.
* LLVM fa840e7dfb9115a3ac9891d898e7fe2543c65948 
* clang 3e24ceaa26f9e1cbd67fdc8625f07bfcc9977053
* CMake 2.8.8 or above.

Installation
------------

* To get the right version of LLVM and clang follow the instructions to install from [git][www/llvmGit]
and then rebase both to the specified versions.

* Change LLVM\_DIR variable in CMakeLists to point to the LLVM directory containing
  LLVMConfig.cmake.

* Out-of-Source compilation is suggested.

* To run the testing programs use tests/runTests.py  
  Make sure to update the paths in LIB\_THRUD, OCL\_HEADER, LD\_PRELOAD and PREFIX
  to point to the correct locations depending on your installation.

Publications
------------

The autocoarsening extension is described in:

* [TACO] Predictable Thread Coarsening
Nicolai Stawinoga, Tony Field (15(2):23, June 2018)

The original coarsening pass has been used for the following publications:

* [SC13] A Large-Scale Cross-Architecture Evaluation of Thread-Coarsening
Alberto Magni, Christophe Dubach, Michael O'Boyle 
* [GPGPU7] Exploiting GPU Hardware Saturation for Fast Compiler Optimization
Alberto Magni, Christophe Dubach, Michael O'Boyle 
* [PACT2014] Automatic Optimization of Thread-Coarsening for Graphics Processors
Alberto Magni, Christophe Dubach, Michael O'Boyle 

These papers used an older version of the pass for LLVM 3.4.

For any question please contact [Nicolai Stawinoga][email/nicolai].

[email/nicolai]: n.stawinoga13@imperial.ac.uk
[email/alberto]: a.magni@sms.ed.ac.uk
[www/originalCoarsening]: https://github.com/HariSeldon/coarsening_pass
[www/llvmProject]: http://llvm.org/docs/Projects.html
[www/llvmGit]: http://llvm.org/docs/GettingStarted.html#git-mirror 
