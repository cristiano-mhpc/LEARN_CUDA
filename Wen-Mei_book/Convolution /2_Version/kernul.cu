/*In this version, the pointer to the filter array is not passed as a paramter to the kernel.
  Insted, the host declares it as a constant global variable:

#define FILTER_RADIUS 2
__constant__ float F_h[2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1];

Then the contents are transferred to the device using 

cudaMemcpyTosymbol(F,F_h, (2*FILTER_RADIUS + 1)*(2*FILTER_RADIUS + 1)*sizeof(float));

stored in constant memory.
*/

__global__ void convolution(float* N, float* P, int r , int width, int height){

	outRow = blockIdx.y*blockDim.y + threadIdx.y;
        outCol = blockIdx.x*blockDim.x + threadIdx.x;

	float sum = 0.0f;
	if (outRow < height && outCol << width){
		//iterate over the filter elements starting at the upperleft most
		for (int filterRow = 0; filterRow < 2*r + 1; filterRow++){
			for (int filterCol = 0; filterCol < 2*r + 1; filterCol++){
				//input rows are indexed as P[outrow - r + filterRow][outCol - r + filterCol]
				int inRow = outRow - r + filterRow;
				int inCol = outCol - r + filterRow;
				
				if (outRow = 0 && outRow + r  < height && outCol = 0 && outCol + r  < width ){
					sum += F[filterRow][filterCol]*N[inRow*width + inCol];//ghost cells are initialized to 0
				}
			}
		}

		//P[outRow][outCol] = sum;
		P[outRow*width + outcol] = sum;
	}

	
}
		
