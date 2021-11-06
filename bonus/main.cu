#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>

#define DATAFILE "./data.bin"
// #define OUTFILE "./snapshot.bin"

// page size is 32bytes
#define PAGE_SIZE (1 << 5)
// 16 KB in page table
#define INVERT_PAGE_TABLE_SIZE (1 << 14)
// 32 KB in shared memory
#define PHYSICAL_MEM_SIZE (1 << 15)
// 128 KB in global memory
#define STORAGE_SIZE (1 << 17)

//// count the pagefault times
__device__ __managed__ int pagefault_num = 0;

// data input and output
__device__ __managed__ uchar results[4][STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

// memory allocation for virtual_memory
// secondary memory
__device__ __managed__ uchar storage[STORAGE_SIZE];
// page table
extern __shared__ u32 pt[];
__device__ __managed__ int priority = 0;

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
							 int input_size);
__host__ void write_binaryFile(char *fileName, void *buffer, int bufferSize);

__global__ void mykernel(int input_size)
{

	int thread_id = getLocalThreadId();

	__shared__ uchar data[PHYSICAL_MEM_SIZE]; // 32KB-data access in share memory

	// __shared__ int priority;

	if (thread_id == 0)
	{
		priority = 0;
	}
	__syncthreads();

	while (1)
	{
		if (thread_id == priority)
		{

			printf("Thread Id: %d \n", thread_id);
			// memory allocation for virtual_memory
			// take shared memory as physical memory

			VirtualMemory vm;
			vm_init(&vm, data, storage, pt, &pagefault_num, PAGE_SIZE,
					INVERT_PAGE_TABLE_SIZE, PHYSICAL_MEM_SIZE, STORAGE_SIZE,
					PHYSICAL_MEM_SIZE / PAGE_SIZE);
			user_program(&vm, input, results[thread_id], input_size);

			init_LRU(&vm);
			// clear the LRU ?
			printf("input size: %d\n", input_size);
			printf("pagefault number is %d\n", pagefault_num);
			priority++;
			// printf("priority: %d\n", priority);
			break;
		}
		else if (priority > 3)
			break;
		__syncthreads();
	}
}

__host__ void write_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;
	fp = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, fp);
	fclose(fp);
}

__host__ int load_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;

	fp = fopen(fileName, "rb");
	if (!fp)
	{
		printf("***Unable to open file %s***\n", fileName);
		exit(1);
	}

	// Get file length
	fseek(fp, 0, SEEK_END);
	int fileLen = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if (fileLen > bufferSize)
	{
		printf("****invalid testcase!!****\n");
		printf("****software warrning: the file: %s size****\n", fileName);
		printf("****is greater than buffer size****\n");
		exit(1);
	}

	// Read file contents into buffer
	fread(buffer, fileLen, 1, fp);
	fclose(fp);

	return fileLen;
}

int main()
{
	cudaError_t cudaStatus;

	int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE);

	char output_file1[] = "snapshot_1.bin";
	char output_file2[] = "snapshot_2.bin";
	char output_file3[] = "snapshot_3.bin";
	char output_file4[] = "snapshot_4.bin";
	char *output_file[4] = {output_file1, output_file2, output_file3, output_file4};
	// user program the access pattern for testing paging
	/* Launch kernel function in GPU, with single thread
	and dynamically allocate INVERT_PAGE_TABLE_SIZE bytes of share memory,
	which is used for variables declared as "extern __shared__" */
	mykernel<<<1, 4, INVERT_PAGE_TABLE_SIZE>>>(input_size);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "mykernel launch failed: %s\n",
				cudaGetErrorString(cudaStatus));
		return;
	}

	for (int i = 0; i < 4; i++)
	{
		write_binaryFile(output_file[i], results[i], input_size);
	}
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}
