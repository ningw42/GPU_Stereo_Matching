#include "KernelFunction.cuh"

// guided filter
__global__ void kernelBoxFilter(float *src, float *dst, int r, int c, int rows, int cols)
{
	int row = blockIdx.x, col = threadIdx.x;
	int index = row * blockDim.x + col;
	float sum = 0;
	size_t left, right, upper, lower;
	upper = MAX(row - r, 0);
	lower = MIN(rows, row + r);
	left = MAX(col - c, 0);
	right = MIN(cols, col + c);
	for (size_t currRow = upper; currRow <= lower; currRow++)
	{
		for (size_t currCol = left; currCol <= right; currCol++)
		{
			sum += src[currRow * cols + currCol];
		}
	}
	//float temp = sum / (r * c);
	dst[index] = sum / ((lower - upper + 1) * (right - left + 1));
}

__global__ void kernelMul(float *first, float *second, float *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	result[index] = first[index] * second[index];
}

__global__ void kernelDivide(float *first, float *second, float *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	result[index] = first[index] / second[index];
}

__global__ void kernelSub(float *first, float *second, float *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	result[index] = first[index] - second[index];
}

__global__ void kernelAddEle(float *first, float e, float *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	result[index] = first[index] + e;
}

__global__ void kernelAdd(float *first, float *second, float *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	result[index] = first[index] + second[index];
}

__global__ void kernelConvertToFloat(uchar *src, float *dst)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	dst[index] = uchar2float(src[index]);
}

__global__ void kernelConvertToUchar(float *src, uchar *dst)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	dst[index] = float2uchar(src[index]);
}

__device__ __forceinline__ float uchar2float(uchar a)
{
	return float(a);
}

__device__ __forceinline__ uchar float2uchar(float a)
{
	unsigned int res = 0;
	asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(res) : "f"(a));
	return res;
}


// block matching
__global__ void kernelPreCal(uchar *left, uchar *right, uchar *difference, int numberOfCols, int numberOfRows, int total)
{
	int index = threadIdx.x;
	int th = index * total;

	for (size_t i = 0; i < total; i++)
	{
		int c = i % numberOfCols - index;
		if (c < 0) continue;
		difference[i + th] = (uchar)std::abs(left[i] - right[i - index]);
	}
}

__global__ void kernelPreCal_V2(uchar *left, uchar *right, uchar *difference, int numberOfCols, int numberOfRows, int total)
{
	int colIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int frameBias = rowIndex * numberOfCols + colIndex;
	int frameIndex = blockIdx.z;
	int index = frameIndex * total + frameBias;

	// calculate difference only if two pixels are at the same line 
	int refCol = colIndex - frameIndex;
	if (refCol >= 0)
	{

		//difference[index] = (uchar)__usad(left[frameBias] - right[frameBias - frameIndex]);
		difference[index] = (uchar)std::abs(left[frameBias] - right[frameBias - frameIndex]);
	}
}

__global__ void kernelFindCorr(uchar *difference, uchar *disparity, int numberOfCols, int numberOfRows, int windowArea, int searchRange, int total, int windowsLength, int SADWinwdowSize)
{
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int currentMinSAD = 100 * windowArea;
	int matchedPosDisp = 0;
	int col = threadIdx.x;
	int row = blockIdx.x;
	int th = 0;

	if (col < searchRange)
	{
		disparity[threadIndex] = 0;
		return;
	}

	for (int _search = 1; _search <= searchRange; _search++, th += total) {
		if (col + _search > numberOfCols) break;
		int SAD = 0;
		// calculate the SAD of the current disparity
		for (int i = -SADWinwdowSize; i <= SADWinwdowSize; i++)
		{
			for (int j = -SADWinwdowSize; j <= SADWinwdowSize; j++)
			{
				int _col = col + j;
				if (_col >= numberOfCols || _col < 0) continue;
				int _row = row + i;
				if (_row >= numberOfRows || _row < 0) continue;
				SAD += difference[th + threadIndex + numberOfCols * i + j];
			}
		}
		if (SAD < currentMinSAD) {
			matchedPosDisp = _search;
			currentMinSAD = SAD;
		}
	}

	//if (currentMinSAD < 2 * windowArea && matchedPosDisp < 5)
	//{
	//	matchedPosDisp = 0;
	//}

	disparity[threadIndex] = matchedPosDisp;
}

__global__ void kernelFindCorrLinear(uchar *difference, uchar *disparity, int numberOfCols, int numberOfRows, int searchRange, int total, int SADWinwdowSize)
{
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int currentMinSAD = 100 * (2 * SADWinwdowSize + 1);
	int matchedPosDisp = 255;
	int col = threadIdx.x;
	int row = blockIdx.x;
	int th = 0;
	//int rowBase = numberOfCols * row;

	if (col < searchRange)
	{
		disparity[threadIndex] = 0;
		return;
	}

	for (int _search = 1; _search <= searchRange; _search++, th += total) {
		if (col + _search > numberOfCols) break;
		int SAD = 0;
		// calculate the SAD of the current disparity !! Linear
			for (int j = -SADWinwdowSize; j <= SADWinwdowSize; j++)
			{
				int _col = col + j;
				if (_col >= numberOfCols || _col < 0) continue;
				SAD += difference[th + threadIndex + j];
			}
		if (SAD < currentMinSAD) {
			matchedPosDisp = _search;
			currentMinSAD = SAD;
		}
	}

	//if (currentMinSAD < 2 * windowArea && matchedPosDisp < 5)
	//{
	//	matchedPosDisp = 0;
	//}

	disparity[threadIndex] = matchedPosDisp;
}


__global__ void kernelFindCorrNonPreCal(uchar *left, uchar *right, uchar *disparity, int numberOfCols, int numberOfRows, int windowArea, int searchRange, int total, int windowsLength, int SADWinwdowSize)
{
	// grid and block should be <rows, cols> respectively
	int col = threadIdx.x;
	int row = blockIdx.x;
	int threadIndex = row * blockDim.x + col;
	int currentMinSAD = 20 * windowArea;
	int matchedPosDisp = 0;

	if (col < searchRange)
	{
		disparity[threadIndex] = 0;
		return;
	}

	for (int _search = 0; _search < searchRange; _search++) {
		if (col + _search > numberOfCols) break;
		int SAD = 0;
		// calculate the SAD of the current disparity
		for (int i = -SADWinwdowSize; i <= SADWinwdowSize; i++)
		{
			for (int j = -SADWinwdowSize; j <= SADWinwdowSize; j++)
			{
				int _col = col + j;
				if (_col >= numberOfCols || _col < 0) continue;
				int _row = row + i;
				if (_row >= numberOfRows || _row < 0) continue;
				int base = threadIndex + numberOfCols * i + j;
				SAD += (uchar)std::abs(left[base + _search] - right[base]);
			}
		}
		if (SAD < currentMinSAD) {
			matchedPosDisp = _search;
			currentMinSAD = SAD;
		}
	}

	disparity[threadIndex] = matchedPosDisp;
}

// a pair of function to get the matched position
__global__ void kernelFindAllSAD(uchar *left, uchar *right, uchar *difference, uchar *SAD_data, int numberOfCols, int numberOfRows, int windowArea, int searchRange, int total, int windowLength, int SADWinwdowSize)
{
	int colIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int frameBias = rowIndex * numberOfCols + colIndex;
	int frameIndex = blockIdx.z;
	int index = frameIndex * total + frameBias;

	if (colIndex + frameIndex > numberOfCols)
	{
		SAD_data[frameBias * searchRange + frameIndex] = 255;
		return;
	}

	int SAD = 0;
	int currCol, currRow;

	for (int i = -SADWinwdowSize; i <= SADWinwdowSize; i++)
	{
		for (int j = -SADWinwdowSize; j <= SADWinwdowSize; j++)
		{
			currCol = colIndex + j;
			if (currCol >= numberOfCols || currCol < 0) continue;
			currRow = rowIndex + i;
			if (currRow >= numberOfRows || currRow < 0) continue;
			SAD += difference[index + i * numberOfCols + j];
		}
	}

	SAD_data[frameBias * searchRange + frameIndex] = SAD;
}

__global__ void kernelFindMinSAD(uchar *SAD_data, uchar *disparity, int numberOfCols, int searchRange)
{
	// TO-DO: return the original index of the min element as the matched position
	extern __shared__ int idx[];
	for (size_t i = 0; i < searchRange; i++)
	{
		idx[i] = i;
	}
	int step = threadIdx.x;
	int frameBias = blockIdx.x * numberOfCols + blockIdx.y;
	int base = searchRange * frameBias;
	int matchedPos = 0;

	int index = step + base;
	for (size_t i = blockDim.x >> 1; i > 0; i = i >> 1)
	{
		if (step < i)
		{
			if (SAD_data[index] > SAD_data[index + i])
			{
				SAD_data[index] = SAD_data[index + i];
				idx[step] = idx[step + i];
			}
		}
		__syncthreads();
	}

	if (step == 0)
	{
		disparity[frameBias] = idx[0];
	}
}

__global__ void kernelRemap(uchar *src, uchar *dst, float *mapx, float *mapy, int rows, int cols)
{
	int index = blockIdx.x * cols + threadIdx.x;

	const float xcoo = mapx[index];
	const float ycoo = mapy[index];
	dst[index] = float2uchar(BilinearInterpolation(src, rows, cols, ycoo, xcoo));
}

__global__ void kernelCvtColor(uchar3 *src, uchar *dst, int rows, int cols)
{
	int index = blockIdx.x * cols + threadIdx.x;

	uchar3 rgb = src[index];
	float channelSum = .299f * rgb.x + .587f * rgb.y + .114f * rgb.z;
	dst[index] = float2uchar(channelSum);
}

//__device__ __forceinline__ uchar float2uchar(float a)
//{
//	unsigned int res = 0;
//	asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(res) : "f"(a));
//	return res;
//}

__device__ float BilinearInterpolation(uchar *src, int rows, int cols, float x, float y)
{
	int x1 = floorf(x), y1 = floorf(y), x2 = x1 + 1, y2 = y1 + 1;
	if (x1 < 0 || x2 >= rows || y1 < 0 || y2 >= cols) {
		return 0;
	}

	int baseIndex = x1 * cols + y1;
	uchar Q11 = src[baseIndex], Q12 = src[baseIndex + 1], Q21 = src[baseIndex + cols], Q22 = src[baseIndex + cols + 1];

	float left = (x2 - x) * Q11 + (x - x1) * Q21;
	float right = (x2 - x) * Q12 + (x - x1) * Q22;

	float result = (y2 - y) * left + (y - y1) * right;
	return result;
}
