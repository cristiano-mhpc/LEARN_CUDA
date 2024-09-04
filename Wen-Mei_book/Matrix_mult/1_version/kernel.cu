__global__ void matrix_mult(float* M, float* N, float* P, int width){

	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if (row < width && col < width){
		float pvalue = 0.0f;
		for (int k = 0; k < width; k++){
			pvalue += M[row*width + k]*N[k*width + col];
		}

		P[row*with + col] = pvalue;
	}
}
