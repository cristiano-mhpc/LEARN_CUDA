#define TILE_WIDTH 16
#define COARSE_FACTOR 4 

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
        int col_start = bx*TILE_WIDTH*COARSE_FACTOR + tx; // each block takes care of COARSE_FACTOR consecutive tiles 

	//loop over the M and N tiles required to compute the P element 
	
	float pvalues[COARSE_FACTOR];
	for (int i =0; i < COARSE_FACTOR; ++i){
		pvalues[i] = 0.0f;
	}

        //Loop over the M and N tiles required to compute element 
	for (int phase = 0; phase < width/TILE_WIDTH; phase++){
		/*Load the elements from the input array to the shared memory. The column of the element to be accessed from M depends on the phase.
		Its given by (phase*TILE_WIDTH + tx). Likewise, the row of the element to be accessed from N is (phase*TILE_WIDTH + ty). */
		
		Mds[ty][tx] = M[row*width + phase*TILE_WIDTH + tx];
		
		for (int c = 0; c < COARSE_FACTOR; ++c){
			col = col_start + c * TILE_WIDTH;
			
			//collaborate loading of N tile into shared memory		
			Nds[ty][tx] = N[(phase*TILE_WIDTH + ty)*width + col];
			__syncthreads();

			for(int k = 0; k < TILE_WIDTH; k++){
				pvalues[c] += Mds[ty][k]*Nds[k][tx];
			
			}

			__syncthreads();

			P[row*width + col] = pvalues[c];
			
		}	

 	}
		
}
