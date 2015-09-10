#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
	odata[0] = 0;
	for (int i = 1; i < n; i++) {
		odata[i] = idata[i-1] + odata[i-1];
	}
    
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    int sum = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			odata[sum] = idata[i];
			sum++;
		}
		
	}
    return sum;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    int* bools = new int[n];
	for (int i = 0; i < n; i++) {
		if (idata[i] == 0) {
			bools[i] = 0;
		}
		else {
			bools[i] = 1;
		}
	}
	int* scanArray = new int[n];
	scan(n, scanArray, bools);
	for (int i = 0; i < n; i++) {
		if (bools[i] == 1) {
			odata[scanArray[i]] = idata[i];
		}
	}
    return scanArray[n - 1] + bools[n - 1];
}

}
}
