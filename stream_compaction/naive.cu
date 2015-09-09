#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

int *g_odata;
int *g_idata;
__global__ void kern_scan(int n, int *odata, const int *idata, int layer) {
	int thrId = threadIdx.x;
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
		
	int* scanArray = new int[n];
	scanArray[0] = 0;
	for (int i = 1; i < n; i++) {
		scanArray[i] = idata[i - 1];
	}

	cudaMemcpy(g_odata, odata, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_idata, scanArray, n*sizeof(int), cudaMemcpyHostToDevice);
	for (int d = 1; d <= ilog2ceil(n); d++) {
		int layer = pow(2, d - 1);
		kern_scan<<<1, n>>>(n, g_odata, g_idata, layer);
		g_idata = g_odata;
	}
	cudaMemcpy(odata, g_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
}

}
}
