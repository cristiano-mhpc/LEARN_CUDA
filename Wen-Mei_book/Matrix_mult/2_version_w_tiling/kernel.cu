#define TILE_WIDTH 16


__global__ void matrix_mul(float* M, float* N, float* P, int width){

	//allocate shared memory arrays
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];


	//shorten the notation
	int bx = blockIdx.x;   
	int tx = threadIdx.x;

	int by = blockIdx.y;
	int ty = threadIdx.y;

	//identify the row and column of the P matrix elements assigned to each thread

	int row = by*TILE_WIDTH + ty; // We assume blockDim.y = blockDim.x = TILE_WIDTH. I.e. block dimensions = tile dimensions  
        int col = bx*TILE_WIDTH + tx; 

	//loop over the M and N tiles required to comppute the P element 

	float pvalue = 0.0f;
        //strip mining. Break the original long loop into a nested loop. 
	for (int phase = 0; phase < width/TILE_WIDTH; phase++){
		/*Load the elements from the input array to the shared memory. The column of the element to be accessed from M depends on the phase.
		Its given by (phase*TILE_WIDTH + tx). Likewise, the row of the element to be accessed from N is (phase*TILE_WIDTH + ty). */
		
		Mds[ty][tx] = M[row*width + phase*TILE_WIDTH + tx];	
		Nds[ty][tx] = N[(phase*TILE_WIDTH + ty)*width + col];

		__syncthreads();// make sure all threads loaded the values assigned to them before proceeding to compute the dot product in the present phase

		for(int k = 0; k < TILE_WIDTH; k++){
		       pvalue += Mds[ty][k]*Nds[k][tx];
		}

		__syncthreads(); // make sure all threads computed their pvalues before moving on to the next phase
 	}

	P[row*with + col] = pvalue;
		
}
