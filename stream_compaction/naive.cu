#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

int *g_odata;
int *g_idata;
__global__ void kern_scan(int n, int *odata, const int *idata, int layer) {
	int thrId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (thrId >= layer) {
		
		odata[thrId] = idata[thrId - layer] + idata[thrId];
		
	}
	else {
		odata[thrId] = idata[thrId];
	}
	
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	cudaMalloc((void**)&g_odata, n * sizeof(int));
	cudaMalloc((void**)&g_idata, n * sizeof(int));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int blockSize = 128;
	dim3 numBlocks = (int)ceil((float)n / (float)blockSize);
	int powTwo = pow(2, ilog2ceil(n));
	dim3 fullBlocksPerGrid((powTwo + blockSize - 1) / blockSize);

	int* scanArray = new int[n];
	scanArray[0] = 0;
	for (int i = 1; i < n; i++) {
		scanArray[i] = idata[i - 1];
	}

	cudaMemcpy(g_odata, odata, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_idata, scanArray, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	for (int d = 1; d <= ilog2ceil(n); d++) {
		int layer = pow(2, d - 1);
		kern_scan<<<fullBlocksPerGrid, blockSize>>>(n, g_odata, g_idata, layer);
		g_idata = g_odata;
	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("%f - ", milliseconds);
	cudaMemcpy(odata, g_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
}

}
}
