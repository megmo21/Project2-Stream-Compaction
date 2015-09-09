#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

int* g_odata;
int* g_idata;

__global__ void kern_up_sweep(int n, int *odata, const int *idata, int layer) {
	int thrId = layer*threadIdx.x;
	odata[thrId + layer - 1] += idata[thrId + (layer / 2) - 1];
}

__global__ void kern_down_sweep(int n, int *odata, const int *idata, int layer) {
	int thrId = n - 1 - layer*threadIdx.x;
	int temp = idata[thrId + (layer / 2) - 1];
	odata[thrId + (layer / 2) - 1] = idata[thrId + layer - 1];
	odata[thrId + layer - 1] = temp + idata[thrId + layer - 1];
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
	
    for (int d = 0; d <= ilog2ceil(n) - 1; d++) {
		int layer = pow(2, d + 1);
		float mult = 1.0f / (float)layer;
		g_odata = g_idata;
		int block = ceil(n*mult);
		kern_up_sweep<<<1, block>>>(n, g_odata, g_idata, layer);
		g_idata = g_odata;
	}
	cudaMemcpy(scanArray, g_idata, n*sizeof(int), cudaMemcpyDeviceToHost);
	scanArray[n-1] = 0;
	cudaMemcpy(g_idata, scanArray, n*sizeof(int), cudaMemcpyDeviceToHost);
	for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
		int layer = pow(2, d + 1);
		float mult = 1.0f / (float)layer;
		g_odata = g_idata;
		int block = ceil(n*mult);
		kern_down_sweep<<<1, block>>>(n, g_odata, g_idata, layer);
		g_idata = g_odata;
	}

	cudaMemcpy(odata, g_odata, n*sizeof(int), cudaMemcpyDeviceToHost);

}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
    // TODO
    return -1;
}

}
}
