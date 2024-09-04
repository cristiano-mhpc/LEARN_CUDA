/*This is a tiled version. The thread organization is 
  such that the number of threads in a block equals the number of elements in 
  the output tile. We again assume that the filter array F is in constant 
  memory, that is in the host code:

#define FILTER_RADIUS 2

__constant__ float F_h[2*FILTER_RADIUS +1][2*FILTER_RADIUS + 1];

Then the contents are transferred to the device constant memory using

cudaMemcpyTosymbol(F, F_h, (2*FILTER_RADIUS + 1)*(2*FILTER_RADIUS + 1)*sizeof(float));
*/

//declare compile constants 

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float F_h[2*FILTER_RADIUS +1][2*FILTER_RADIUS + 1];

__global__ void convolution(float* N, float* P , int width, int height){

	//Map of the threads in the block to the rows and columns of the output tile
	row = blockIdx.y*blockDim.y + threadIdx.y;
        col = blockIdx.x*blockDim.x + threadIdx.x;

	/*Each thread will load the input tile to a statically allocated array
          in the shared memory for this block.*/

	__shared__ N_s[IN_TILE_DIM][IN_TILE_DIM];

	//we load the elements in four phases
	if ((row - FILTER_RADIUS) >= 0 &&  (col - FILTER_RADIUS) >= 0 && (col + FILTER_RADIUS) < width && (row + FILTER_RADIUS) < height){

		N_s[threadIdx.y][threadIdx.x] = N[(row - FILTER_RADIUS)*width + (col - FILTER_RADIUS)];
       		

		//the threads in the last 2*FILTER_RADIUS of columns and 2*FILTER_RADIUS rows are assigned to the corresponding elements in N_s
	       if (threadIdx.x >= (blockDim.x - 2*FILTER_RADIUS ) || threadIdx.y >= (blockDim.y - 2*FILTER_RADIUS ) ){
			N_s[threadIdx.y + FILTER_RADIUS][threadIdx.x + FILTER_RADIUS] = N[(row + FILTER_RADIUS)*width + ( col + FILTER_RADIUS) ];

	       }
               
	       //the upper right corner (2*FILTER_RADIUS) by (2*FILTER_RADIUS)
	       if (threadIdx.x > (blockDim.x - 2*FILTER_RADIUS ) && threadIdx.y < 2*FILTER_RADIUS){
			N_s[threadIdx.y - FILTER_RADIUS][threadIdx.x + FILTER_RADIUS] = N[ (row - 2*FILTER_RADIUS)*width + (col + 2*FILTER_RADIUS)];
	       }

	    
	       //the lower left corner
	       if (threadIdx.y > (blockDim.y - 2*FILTER_RADIUS ) && threadIdx.x < 2*FILTER_RADIUS){
			N_s[threadIdx.y + 2*FILTER_RADIUS][threadIdx.x - 2*FILTER_RADIUS] = N[ (row + 2*FILTER_RADIUS)*width + (col - 2*FILTER_RADIUS)];
	       }

	     
	} else { 
		N_s[threadIdx.y][threadIdx.x] = 0.0f; //ghost cells 
	}
	__syncthreads();

	if (row < height && col < width ){
		pvalue = 0.0f; 
		for(int frow = 0; frow < (2*FILTER_RADIUS) + 1; frow++){
			for (int fcol = 0; fcol < (2*FILTER_RADIUS) + 1; fcol++){
				int inrow = row - FILTER_RADIUS + frow; 
				int incol = col - FILTER_RADIUS + frow;

				pvalue += F[frow][fcol]*N_s[inrow][incol]; 
			}

		}

	P[row*width + col] = pvalue;
	}	
		
}
		
