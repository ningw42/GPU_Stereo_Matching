#include "guidedFilter.cuh"
#include "KernelFunction.cuh"

guidedFilterGPU::guidedFilterGPU(int _rows, int _cols, int _r, int _c, float _eps)
{
	rows = _rows;
	cols = _cols;
	r = _r;
	c = _c;
	eps = _eps;
	total = rows * cols;
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
}

guidedFilterGPU::guidedFilterGPU()
{

}

guidedFilterGPU::~guidedFilterGPU()
{
}

void guidedFilterGPU::filter(uchar *I_uchar, uchar *p_uchar, uchar *result)
{
	kernelConvertToFloat << <rows, cols >> >(I_uchar, I); // I convert to float
	kernelConvertToFloat << <rows, cols >> >(p_uchar, p); // p convert to float

	kernelBoxFilter << <rows, cols >> >(I, mean_I, r, c, rows, cols); // mean_I = boxfilter(I, r, c)

	kernelMul << <rows, cols >> >(I, I, sq_I); // 
	kernelBoxFilter << <rows, cols >> >(sq_I, mean_II, r, c, rows, cols);// mean_II = boxfilter(I.mul(I), r, c)

	kernelMul << <rows, cols >> >(mean_I, mean_I, sq_mean_I); // can be optimized
	kernelSub << <rows, cols >> >(mean_II, sq_mean_I, var_I); // var_I = mean_II - mean_I.mul(mean_I)

	kernelBoxFilter << <rows, cols >> >(p, mean_p, r, c, rows, cols); // mean_p = boxfilter(p, r, c)
	
	kernelMul << <rows, cols >> >(I, p, mul_Ip); // 
	kernelBoxFilter << <rows, cols >> >(mul_Ip, mean_Ip, r, c, rows, cols); // mean_Ip = boxfilter(I.mul(p), r, c)

	kernelMul << <rows, cols >> >(mean_I, mean_p, mul_mean_Ip_mean_p); // can be optimized
	kernelSub << <rows, cols >> >(mean_Ip, mul_mean_Ip_mean_p, cov_Ip); // cov_Ip = mean_Ip - mean_I.mul(mean_p);

	kernelAddEle << <rows, cols >> >(var_I, eps, sum_varI_eps); // can be optimized
	kernelDivide << <rows, cols >> >(cov_Ip, sum_varI_eps, a); // a = cov_Ip / (var_I + eps)

	kernelMul << <rows, cols >> >(a, mean_I, mul_a_meanI); // can be optimized
	kernelSub << <rows, cols >> >(mean_p, mul_a_meanI, b); // b = mean_p - a.mul(mean_I);

	kernelBoxFilter << <rows, cols >> >(a, mean_a, r, c, rows, cols); // mean_a = boxfilter(a, r, c)

	kernelBoxFilter << <rows, cols >> >(b, mean_b, r, c, rows, cols); // mean_b = boxfilter(b, r, c)

	kernelMul << <rows, cols >> >(mean_a, I, mul_meana_I); // can be optimized
	kernelAdd << <rows, cols >> >(mul_meana_I, mean_b, result_float); // return mean_a.mul(I) + mean_b

	kernelConvertToUchar << <rows, cols >> >(result_float, result); // 
}
