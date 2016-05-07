#include "guidedFilter.cuh"

guidedFilterGPU::guidedFilterGPU(int _rows, int _cols, int _r, int _c, float _eps)
{
	// temp 
	cudaMalloc(&I, total * sizeof(float));
	cudaMalloc(&p, total * sizeof(float));
	cudaMalloc(&sq_I, total * sizeof(float));
	cudaMalloc(&sq_mean_I, total * sizeof(float));
	cudaMalloc(&mul_Ip, total * sizeof(float));
	cudaMalloc(&mul_mean_Ip_mean_p, total * sizeof(float));
	cudaMalloc(&sum_varI_eps, total * sizeof(float));
	cudaMalloc(&mul_a_meanI, total * sizeof(float));

	cudaMalloc(&mul_meana_I, total * sizeof(float));
	cudaMalloc(&result_float, total * sizeof(float));
	// useful
	cudaMalloc(&mean_I, total * sizeof(float));
	cudaMalloc(&mean_II, total * sizeof(float));
	cudaMalloc(&mean_p, total * sizeof(float));
	cudaMalloc(&var_I, total * sizeof(float));
	cudaMalloc(&mean_Ip, total * sizeof(float));
	cudaMalloc(&cov_Ip, total * sizeof(float));
	cudaMalloc(&a, total * sizeof(float));
	cudaMalloc(&b, total * sizeof(float));
	cudaMalloc(&mean_a, total * sizeof(float));
	cudaMalloc(&mean_b, total * sizeof(float));

	rows = _rows;
	cols = _cols;
	r = _r;
	c = _c;
	eps = _eps;
}

guidedFilterGPU::guidedFilterGPU()
{

}

guidedFilterGPU::~guidedFilterGPU()
{
}

void guidedFilterGPU::filter(uchar *I_uchar, uchar *p_uchar, uchar *result)
{
	kernalConvertToFloat(I_uchar, I); // I convert to float
	kernalConvertToFloat(p_uchar, p); // p convert to float

	kernalBoxFilter(I, mean_I, r, c, rows, cols); // mean_I = boxfilter(I, r, c)

	kernalMul(I, I, sq_I); // 
	kernalBoxFilter(sq_I, mean_II, r, c, rows, cols);// mean_II = boxfilter(I.mul(I), r, c)

	kernalMul(mean_I, mean_I, sq_mean_I); //
	kernalSub(mean_II, sq_mean_I, var_I); // var_I = mean_II - mean_I.mul(mean_I)

	kernalBoxFilter(p, mean_p, r, c, rows, cols); // mean_p = boxfilter(p, r, c)
	
	kernalMul(I, p, mul_Ip); // 
	kernalBoxFilter(mul_Ip, mean_Ip, r, c, rows, cols); // mean_Ip = boxfilter(I.mul(p), r, c)

	kernalMul(mean_I, mean_p, mul_mean_Ip_mean_p);
	kernalSub(mean_Ip, mul_mean_Ip_mean_p, cov_Ip); // cov_Ip = mean_Ip - mean_I.mul(mean_p);

	kernalAddEle(var_I, eps, sum_varI_eps); //
	kernalDivide(cov_Ip, sum_varI_eps, a); // a = cov_Ip / (var_I + eps)

	kernalMul(a, mean_I, mul_a_meanI); //
	kernalSub(mean_p, mul_a_meanI, b); // b = mean_p - a.mul(mean_I);

	kernalBoxFilter(a, mean_a, r, c, rows, cols); // mean_a = boxfilter(a, r, c)

	kernalBoxFilter(b, mean_b, r, c, rows, cols); // mean_b = boxfilter(b, r, c)

	kernalMul(mean_a, I, mul_meana_I);
	kernalAdd(mul_meana_I, mean_b, result_float); // return mean_a.mul(I) + mean_b

	kernalConvertToUchar(result_float, result); // 
}

__global__ void kernalBoxFilter(float *src, float *dst, int r, int c, int rows, int cols)
{
	int row = blockIdx.x, col = threadIdx.x;
	int index = row * blockDim.x + col;
	float sum = 0;
	for (size_t currRow = MAX(row - r, 0); currRow <= MIN(rows, row + r); currRow++)
	{
		for (size_t currCol = MAX(col - c, 0); currCol <= MIN(cols, col + c); currCol++)
		{
			sum += src[currRow * cols + currCol];
		}
	}
	dst[index] = sum / (r * c);
}

__global__ void kernalMul(float *first, float *second, float *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	result[index] = first[index] * second[index];
}

__global__ void kernalDivide(float *first, float *second, float *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	result[index] = first[index] / second[index];
}

__global__ void kernalSub(float *first, float *second, float *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	result[index] = first[index] - second[index];
}

__global__ void kernalAddEle(float *first, float e, float *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	result[index] = first[index] + e;
}

__global__ void kernalAdd(float *first, float *second, float *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	result[index] = first[index] + second[index];
}

__global__ void kernalConvertToFloat(uchar *src, float *dst)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	dst[index] = uchar2float(src[index]);
}

__global__ void kernalConvertToUchar(float *src, uchar *dst)
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