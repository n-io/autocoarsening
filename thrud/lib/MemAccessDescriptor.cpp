#include "thrud/MemAccessDescriptor.h"
#include <vector>
#include <set>
#include <list>
#include <algorithm>
#include <functional>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_PRINT

using namespace std;

int MemAccessDescriptor::SIZE = 0;
int MemAccessDescriptor::CACHE_SIZE = 0;

void MemAccessDescriptor::init(const int x, const int y, const int z) {
  isValue = false;
  //hasDim[0] = x > 1;
  //hasDim[1] = y > 1;
  //hasDim[2] = z > 1;
  sizes[0] = x;
  sizes[1] = y;
  sizes[2] = z;
  mad.resize(z);
  for (int k = 0; k < z; k++) {
    mad[k].resize(y);
    for (int j = 0; j < y; j++) {
      mad[k][j].resize(x);
      for (int i = 0; i < x; i++) {
	mad[k][j][i] = i|j|k;
      }
    }
  }
  SIZE += max(x,1) * max(y,1) * max(z,1) * sizeof(int);
}

MemAccessDescriptor::MemAccessDescriptor(int val) {
  isValue = true;
  value = val;
  mad.resize(1);
  mad[0].resize(1);
  mad[0][0].push_back(val);
  SIZE += sizeof(int);
}

MemAccessDescriptor::MemAccessDescriptor(int dimension, int n) {
  isValue = false;
  if (dimension == 0) {
    init(n, 1, 1);
  } else if (dimension == 1) {
    init(1, n, 1);
  } else { //if (dimension == 2) {
    init(1, 1, n);
  }
}

MemAccessDescriptor::MemAccessDescriptor(const int x, const int y, const int z) {
  /* typically only one of (x,y,z) will be set to a value other-and-larger than 1 */
  init(x, y, z);
}

MemAccessDescriptor::MemAccessDescriptor(function<int(int, int)> f, MemAccessDescriptor &a, MemAccessDescriptor &b) {
  isValue = false;
  //int sizes[3] = {1, 1, 1};
  int sizesProduct = 1;
  for (int i = 0; i < 3; i++) {
    sizes[i] = max(a.sizes[i], b.sizes[i]);
    //hasDim[i] = sizes[i] > 1;
    sizesProduct *= max(sizes[i], 1);
  }
  SIZE += sizesProduct * sizeof(int);
  mad.resize(sizes[2]);
  for (int k = 0; k < sizes[2]; k++) {
    int aZ = a.hasDim(2) ? k : 0;
    int bZ = b.hasDim(2) ? k : 0;
    mad[k].resize(sizes[1]);
    for (int j = 0; j < sizes[1]; j++) {
      int aY = a.hasDim(1) ? j : 0;
      int bY = b.hasDim(1) ? j : 0;
      mad[k][j].resize(sizes[0]);
      for (int i = 0; i < sizes[0]; i++) {
        int aX = a.hasDim(0) ? i : 0;
        int bX = b.hasDim(0) ? i : 0;
        mad[k][j][i] = f(a.mad[aZ][aY][aX], b.mad[bZ][bY][bX]);
      }
    }
  }
}

bool MemAccessDescriptor::hasDim(int d) {
  return sizes[d] > 1;
}

MemAccessDescriptor MemAccessDescriptor::compute(function<int(int, int)> f, MemAccessDescriptor &operand) {
  if (isValue && operand.isValue) {
    return MemAccessDescriptor(f(value, operand.value));
  } else {
    return MemAccessDescriptor(f, *this, operand);
  }
}

MemAccessDescriptor MemAccessDescriptor::select(MemAccessDescriptor &a, MemAccessDescriptor &b) {
  // from the perspective of the predicate
  if (a.isValue && b.isValue) {
    return MemAccessDescriptor(mad[0][0][0] ? a.value : b.value);
  } else {
    MemAccessDescriptor result(max(a.sizes[0], b.sizes[0]),
                           max(a.sizes[1], b.sizes[1]),
                           max(a.sizes[2], b.sizes[2]));
    for (int k = 0; k < result.sizes[2]; k++) {
      int aZ = a.hasDim(2) ? k : 0;
      int bZ = b.hasDim(2) ? k : 0;
      int pZ = hasDim(2)   ? k : 0;
      for (int j = 0; j < result.sizes[1]; j++) {
	int aY = a.hasDim(1) ? j : 0;
	int bY = b.hasDim(1) ? j : 0;
        int pY = hasDim(1)   ? j : 0;
	for (int i = 0; i < result.sizes[0]; i++) {
	  int aX = a.hasDim(0) ? i : 0;
	  int bX = b.hasDim(0) ? i : 0;
          int pX = hasDim(0)   ? i : 0;
	  result.mad[k][j][i] = mad[pZ][pY][pX] ? a.mad[aZ][aY][aX] : b.mad[bZ][bY][bX];
	}
      }
    }
    return result;
  }
}

list<int> MemAccessDescriptor::getMemAccesses(int warpSize, int align, int cacheLineSize, bool *fullCoalescing) {
  list<int> result;
  set<int> warpAccess;
  int consecutiveAccessCounter = 0;
  int lastAccess = -1;
  *fullCoalescing = true;
  for (int k = 0; k < sizes[2]; k++) {
    for (int j = 0; j < sizes[1]; j++) {
      for (int i = 0; i < sizes[0]; i+=warpSize) {
        for (int c = i; c < i+warpSize && c < sizes[0]; c++) {
          warpAccess.insert(((mad[k][j][c] * align) / cacheLineSize) * cacheLineSize);

	  // test for consecutive accesses
	  if (consecutiveAccessCounter > 0 && lastAccess != mad[k][j][c] - 1) {
            *fullCoalescing = false;
	    consecutiveAccessCounter = -1; // restart counting
	  }
          lastAccess = mad[k][j][c];
	  if (++consecutiveAccessCounter == (cacheLineSize / align)) {
            consecutiveAccessCounter = 0;
	  }
        }
#ifdef DEBUG_PRINT
        for (int c : warpAccess) {
          llvm::errs() << c << " ";
        }
        llvm::errs() << "\n";
#endif
        result.insert(result.end(), warpAccess.begin(), warpAccess.end());
        warpAccess.clear();
      }
    }
  }
#ifdef DEBUG_PRINT
  if (fullCoalescing) {
    llvm::errs() << "Mem accesses are fully coalesced\n";
  }
#endif
  CACHE_SIZE += result.size() * sizeof(int);
  return result;
}

void MemAccessDescriptor::print() {
  for (unsigned int i = 0; i < mad.size(); i++) {
    for (unsigned int j = 0; j < mad[0].size(); j++) {
      for (unsigned int k = 0; k < mad[0][0].size(); k++) {
	llvm::errs() << mad[i][j][k] << " ";
      }
      llvm::errs () << "\n";
    }
    llvm::errs() << "---------------------------------------\n";
  }
}

