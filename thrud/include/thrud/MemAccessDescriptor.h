#ifndef MEM_ACCESS_DESCRIPTOR_H
#define MEM_ACCESS_DESCRIPTOR_H

#include <vector>
#include <list>
#include <functional>

using namespace std;

class MemAccessDescriptor {

  private:
    void init(const int x, const int y, const int z);

  public:
    static int SIZE;
    static int CACHE_SIZE;
    bool isValue;
    int value;
    //bool hasDim[3] = {false, false, false};
    int sizes [3] = {1, 1, 1};
    std::vector<vector<vector<int>>> mad;
    
  public:
    MemAccessDescriptor(int value);
    MemAccessDescriptor(int dimension, int n);
    MemAccessDescriptor(const int x, const int y, const int z);
    MemAccessDescriptor(function<int(int, int)> f, MemAccessDescriptor &a, MemAccessDescriptor &b);
    MemAccessDescriptor select(MemAccessDescriptor &a, MemAccessDescriptor &b);
    bool hasDim(int d);
    MemAccessDescriptor compute(function<int(int, int)> f, MemAccessDescriptor &op);
    list<int> getMemAccesses(int warpSize, int align, int cacheLineSize, bool *fullCoalescing);
    void print();
};

#endif
