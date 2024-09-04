/*In this version, the block dimensions are equal to the output tiles. We also use a shared memory array 
  with dimension equal to the input tiles without the halo elements. The halo elements are sourced from 
  the DRAM but is expected to be served from the L2 cache. */


#define FILTER_RADIUS 2
#define TILE_DIM 32


__constant__ float F_h[2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1];
/*
Then the contents are transferred to the device using

cudaMemcpyTosymbol(F,F_h, (2*FILTER_RADIUS + 1)*(2*FILTER_RADIUS + 1)*sizeof(float));

This will be stored in constant memory of the device.
*/

__global__ void convolution(float* N, float* P, int width, int height){

	row = blockIdx.y*blockDim.y + threadIdx.y;
        col = blockIdx.x*blockDim.x + threadIdx.x;

	//load the input tile w/o halo elements
	__shared__ N_s[TILE_DIME][TILE_DIM];
       if (row < height && col << width){
		N_s[threadIdx.y][threadIdx.x] = [row*width + col];
       } else {
		N[threadIdx.y][threadIdx.x] = 0.0f;
       }

	//compute the P elements
	if (row < height && row << width ){
		float pvalue = 0.0f;
		for (int frow = 0; frow < (2* FILTER_RADIUS) + 1; frow++){
			for ( int fcol = 0; fcol < (2*FILTER_RADIUS) + 1;fcol++){
				int tilerow = threadIdx.y - FILTER_RADIUS + frow;
				int tilecol = threadIdx.x - FILTER_RADIUS + fcol;
				//check if internal cells (not halo or ghost)
				if (threadIdx.y - FILTER_RADIUS + frow >= 0 && 
				    threadIdx.y - FILTER_RADIUS + frow  < TILE_DIM &&
				    threadIdx.x - FILTER_RADIUS + fcol  >= 0 &&
				    threadIdx.x - FILTER_RADIUS + fcol  < TILE_DIM ){
					pvalue += F_h[frow][fcol] * P_s[threadIdx.y + frow][threadIdx.y + fcol];					

				} else {
					//check if halo cells
					if (row - FILTER_RADIUS + frow >= 0 &&
					    row - FILTER_RADIUS + frow < height && 
					    col - FILTER_RADIUS + fcol >=0 && 
					    col - FILTER_RADIUS + fcol < width){
						pvalue += F_h[frow][fcol] * P[(row - FILTER_RADIUS + frow)*with + col - FILTER_RADIUS+ fcol]; 
					}

				}	

			}

		}

		P[row*width + col] = pvalue;


	}

	
}
		
