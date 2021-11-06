#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;
const u16 null = UINT16_MAX;
struct Node
{
	u16 prev;
	u16 next;
};

struct LRU
{
	Node *head;
	Node *tail;
	Node *nodes;

	u16 count = 0; // how many frames in LRU, used for check IPT full and free frame.
};
struct VirtualMemory
{
	uchar *buffer;
	uchar *storage;
	u32 *invert_page_table;
	int *pagefault_num_ptr;
	LRU LRU;

	int PAGESIZE;
	int INVERT_PAGE_TABLE_SIZE;
	int PHYSICAL_MEM_SIZE;
	int STORAGE_SIZE;
	int PAGE_ENTRIES;
};

__device__ int getLocalThreadId();
// used for handle LRU
__device__ void init_LRU(VirtualMemory *vm);
__device__ void update_LRU(VirtualMemory *vm, u16 frame_number);
__device__ u16 get_LRU_frame_number(VirtualMemory *vm);
// TODO
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
						u32 *invert_page_table, int *pagefault_num_ptr,
						int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
						int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
						int PAGE_ENTRIES);
__device__ uchar vm_read(VirtualMemory *vm, u32 addr);
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
							int input_size);

#endif
