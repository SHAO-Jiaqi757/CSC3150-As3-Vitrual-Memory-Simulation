#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
__device__ int getLocalThreadId()
{
	return (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
}
__device__ void init_invert_page_table(VirtualMemory *vm)
{

	for (int i = 0; i < vm->PAGE_ENTRIES; i++)
	{
		vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
		vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
	}
}
__device__ bool is_page_fault(VirtualMemory *vm, u32 page_number, u16 &frame_number);
__device__ u32 get_physical_addr(VirtualMemory *vm, u32 addr);
__device__ void page_fault_handler(VirtualMemory *vm, u16 page_number, u16 &frame_number);
__device__ void init_LRU(VirtualMemory *vm)
{

	vm->LRU.count = 0; // empty LRU nodes;
	// initialize the LRU.head and LRU.tail...

	Node *head_ptr = vm->LRU.nodes;
	Node *tail_ptr = vm->LRU.nodes + (vm->PAGE_ENTRIES + 1); // 1025
	u16 head_idx = 0, tail_idx = (vm->PAGE_ENTRIES + 1);

	vm->LRU.head = head_ptr, vm->LRU.tail = tail_ptr;

	vm->LRU.head->next = tail_idx, vm->LRU.tail->prev = head_idx;

	vm->LRU.head->prev = null, vm->LRU.tail->next = null;
	for (int i = 1; i < vm->PAGE_ENTRIES + 1; i++)
	{
		vm->LRU.nodes[i].next = null;
		vm->LRU.nodes[i].prev = null;
	}
	return;
}
// LRU handler...
__device__ void add_to_LRU(VirtualMemory *vm, u16 frame_number)
{
	u16 oldhead = vm->LRU.head->next;
	u16 node_idx = frame_number + 1;
	vm->LRU.nodes[oldhead].prev = node_idx;
	vm->LRU.nodes[node_idx].next = oldhead;

	vm->LRU.nodes[node_idx].prev = 0;
	vm->LRU.head->next = node_idx;

	vm->LRU.count++;
}
__device__ void update_LRU(VirtualMemory *vm, u16 frame_number)
{
	u16 node_idx = frame_number + 1;
	// check the frame_number in LRU or not;
	if (vm->LRU.nodes[node_idx].prev == null)
	{
		// this frame number is not in the LRU;
		add_to_LRU(vm, frame_number);
	}
	else // the frame number in the LRU
	{
		// delete the node from LRU nodes,
		u16 prev = vm->LRU.nodes[node_idx].prev;
		u16 next = vm->LRU.nodes[node_idx].next;

		vm->LRU.nodes[prev].next = next;
		vm->LRU.nodes[next].prev = prev;

		// then, and add it at the front of the LRU nodes
		add_to_LRU(vm, frame_number);
		vm->LRU.count--;
	}
}
__device__ u16 get_LRU_frame_number(VirtualMemory *vm)
{
	// this function is used to get least recently unused frame number
	return vm->LRU.tail->prev - 1;
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
						u32 *invert_page_table, int *pagefault_num_ptr,
						int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
						int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
						int PAGE_ENTRIES)
{
	// init variables
	vm->buffer = buffer;   // 32KB data access in share memory
	vm->storage = storage; // gloabel memory 128KB storage
	vm->invert_page_table = invert_page_table;
	vm->pagefault_num_ptr = pagefault_num_ptr;

	// init constants
	vm->PAGESIZE = PAGESIZE;
	vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
	vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
	vm->STORAGE_SIZE = STORAGE_SIZE;
	vm->PAGE_ENTRIES = PAGE_ENTRIES;

	// before first vm_write or vm_read
	init_invert_page_table(vm);

	int page_size = vm->PAGE_ENTRIES * 4;
	vm->LRU.nodes = (Node *)(vm->invert_page_table + vm->PAGE_ENTRIES); // initialize the start addresss of LRU nodes;
	// initialize the LRU
	init_LRU(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr)
{
	u32 phy_addr = get_physical_addr(vm, addr);
	return vm->buffer[phy_addr];
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value)
{
	u32 phy_addr = get_physical_addr(vm, addr);
	vm->buffer[phy_addr] = value;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
							int input_size)
{
	// int thread_id = getLocalThreadId();
	/* to result buffer */
	for (int i = 0; i < input_size; i++)
	{
		results[i + offset] = vm_read(vm, i);
	}
}

__device__ bool is_page_fault(VirtualMemory *vm, u32 page_number, u16 &frame_number)
{
	int thread_id = getLocalThreadId();
	for (int i = 0; i < vm->PAGE_ENTRIES; i++)
	{
		if (vm->invert_page_table[i] >> 2 == page_number && vm->invert_page_table[i] & 0x3 == thread_id)
		{
			frame_number = i;
			return false;
		}
	}
	return true; // page fault happens
}

__device__ void page_fault_handler(VirtualMemory *vm, u16 page_number, u16 &frame_number)
{
	int thread_id = getLocalThreadId();
	printf("Thread id (page fault): %d \n", thread_id);
	(*vm->pagefault_num_ptr)++;
	// check whether the page table is full <= LRU size == page_entry
	u16 LRU_size = vm->LRU.count;

	u16 free_frame;

	if (LRU_size >= vm->PAGE_ENTRIES) // full, use LRU swapping-in and swapping-out
	{
		free_frame = get_LRU_frame_number(vm);

		u32 old_vpn = vm->invert_page_table[free_frame] >> 2;

		// swap out
		for (int i = 0; i < vm->PAGESIZE; i++)
		{
			vm->storage[old_vpn * vm->PAGESIZE + i] = vm->buffer[free_frame * vm->PAGESIZE + i];
		}
	}
	else // has holes in ITB
	{
		free_frame = LRU_size;
	}
	// update the physical memory
	for (int i = 0; i < vm->PAGESIZE; i++)
		vm->buffer[free_frame * vm->PAGESIZE + i] = vm->storage[page_number * vm->PAGESIZE + i];

	// update the IPT
	vm->invert_page_table[free_frame] = page_number << 2 + thread_id; // last two bit is thread_id
	frame_number = free_frame;
}
__device__ u32 get_physical_addr(VirtualMemory *vm, u32 addr)
{
	u16 page_number = addr / vm->PAGESIZE;
	u32 offset = addr % vm->PAGESIZE;
	u16 frame_number;
	bool page_fault = is_page_fault(vm, page_number, frame_number);
	// check whether the addr is valid in page table.
	if (page_fault)
	{ // if page fault,
		page_fault_handler(vm, page_number, frame_number);
	}

	update_LRU(vm, frame_number);

	u32 phy_addr = frame_number * vm->PAGESIZE + offset;
	return phy_addr;
}
