__global__ void convolution(float* N, float* P, float* F, int r , int width, int height){

	outRow = blockIdx.y*blockDim.y + threadIdx.y;
        outCol = blockIdx.x*blockDim.x + threadIdx.x;

	float sum = 0.0f;
	if (outRow < height && outCol << width){
		//iterate over the filter elements starting at the upperleft most
		for (int filterRow = 0; filterRow < 2*r + 1; filterRow++){
			for (int filterCol = 0; filterCol < 2*r + 1; ++filterCol++){
				//input rows are indexed as P[outrow - r + filterRow][outCol - r + filterCol]
				int inRow = outRow -r + filterRow;
				int inCol = outCol -r + filterRow;
				
				if (outRow-r >= 0 && outRow + r  < height && outCol-r >= 0 && outCol + r  < width ){
					sum += F[filterRow][filterCol]*N[inRow*width + inCol];//ghost cells are initialized to 0
				}
			}
		}

		//P is dynamically allocated on the host. So its linearized.  
		P[outRow*width + outcol] = sum;
	}

	
}
		
