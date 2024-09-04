/*This code demonstrates how to implement a parallel sum on a GPU using both warp shuffle and shared memory.
 Here we assume that we are using thread blocks consisting of 32 warps or 1024 threads.*/

#define WARPSZ 32

#include <stdio.h>
#include <cuda.h>

float f(float);

__device__ float func(float x){
	return x*x + 4.0;
}/*func*/

__device__ float Shared_mem_sum(float shared_vals[]){

	int my_lane = threadIdx.x%warpSize;

	for (int diff = warpSize/2; diff > 0; diff = diff/2){
		/*make sure that 0<=source < warpSize*/
		int source = (my_lane + diff) % warpSize;
		shared_vals[my_lane] += shared_vals[source];
	}
	return shared_vals[my_lane];

}/*Shared_mem_sum*/

__device__ float Warp_sum(float var){

	unsigned mask = 0xffffffff;

	for (int diff = warpSize/2; diff > 0; diff = diff/2)
		var += __shfl_down_sync(mask, var, diff);
	return var;

}/*Warp_sum*/
 
__global__ void Dev_trap(
		const float a        /*in*/,
		const float b        /*in*/,
		const float h        /*in*/,
		const int   n        /*in*/,
		float*      trap_p   /*in/out*/){


	__shared__ float warp_sum_arr[WARPSZ]; 

	int my_i = blockDim.x * blockIdx.x + threadIdx.x; 
	int my_warp = threadIdx.x / warpSize; 
	int my_lane = threadIdx.x % warpSize;

	/*f(x_0) and f(x_n) were computed on the host. So compute
	 f(x_1), f(x_2), ..., f(x_(n-1))*/

	/*All thread's warp sum must be initialized to 0. Including those
	which are not included in the following if-conditional*/	

	float my_trap = 0.0f;

	if (0 < my_i && my_i < n){
		float my_x = a + my_i*h;
		my_trap = my_x*my_x + 4.0;
		
	}
	//all threads in the warps participate in the warp shuffle
	float my_result = Warp_sum(my_trap);
	if (my_lane == 0) warp_sum_arr[my_warp] = my_result;
	__syncthreads();
	
	/*warp 0 will add all the warp sums in the array warp_sum_arr*/
	
	if (my_warp == 0){
		//check whether there are fewer than 32 warps in a block 
		if (threadIdx.x >= blockDim.x/warpSize)
			warp_sum_arr[threadIdx.x] = 0.0f;
		float blk_result = Shared_mem_sum(warp_sum_arr);
	
		/*thread 0 in warp 0 adds the block sum into the total across the grid.
		This will perform a sequential sum, one step for each block in the grid.*/
		//if (threadIdx.x == 0) atomicAdd(trap_p, blk_result); 
		if (my_lane == 0) atomicAdd(trap_p, blk_result);

	}
 
} /*Dev_trap*/


/*Host code*/

void Trap_wrapper(
		const float  a             /* in */, 
		const float  b             /* in */,
		const int    n             /* in */,
		float*       trap_p        /* out */,
		const int    blk_ct        /* in */, 
		const int    th_per_blk    /* in */){
	

	*trap_p = 0.5*(f(a)+f(b));
	float h = (b-a)/n;
	
	//Here we assume th_per_blk = 32, otherwise the code wont work;

   	Dev_trap<<<blk_ct, th_per_blk>>>(a, b, h, n, trap_p);
	cudaDeviceSynchronize();

	*trap_p = h*(*trap_p);

	printf("The result is %f\n", *trap_p);
	
}/* Trap_wrapper */


void Get_args(
        	const int argc         /*in*/,
	        char*     argv[]       /*in*/,
	        int*      n_p          /*out*/,
	        int*      blk_ct_p     /*out*/,
	        int*      th_per_blk_p /*out*/,
	        float*    a_p          /*out*/,
			float*    b_p          /*out*/){

	if (argc != 6){
		/*print an error message and exit*/

		printf("Error: argc must be six");
	}
    
	*n_p = strtol(argv[1], NULL, 10);
	*blk_ct_p = strtol(argv[2], NULL, 10);
	*th_per_blk_p = strtol(argv[3], NULL, 10);
	*a_p = strtol(argv[4], NULL, 10);
	*b_p = strtol(argv[5], NULL, 10);


	/*Is n > total thread count = blk_ct * th_per_blk?*/
	if (*n_p >(*blk_ct_p)*(*th_per_blk_p) ){

		/*print an error message and exit */

		printf("Error: number of partitions must be less than total threads used."); 
	}

} /*Get_args*/


float f(float x){
	return x*x + 4.0;
}


float Serial_trap(
	const float a /* in */,
	const float b /* in */,
	const int   n /* in*/){
		
	float x, h = (b-a)/n;
	float trap = 0.5*(f(a)+f(b));

	for (int i = 1; i <= n-1; i++){
		x = a + i*h;
		trap += f(x);
	}
	trap = trap*h;

	return trap;

}/*Serial_trap*/


int main(int argc, char* argv[]){

	int n, th_per_blk, blk_ct;

	float a, b;	

	float* trap_p;

	cudaMallocManaged(&trap_p, sizeof(float));
	
	/*Ge the command line arguments*/
	Get_args(argc, argv, &n, &blk_ct, &th_per_blk, &a, &b);

	/*Call the wrapper function*/
	Trap_wrapper(a, b, n, trap_p, blk_ct, th_per_blk); 

	/*call Serial_trap */
	float serial = Serial_trap(a, b, n);

	float error = serial - (*trap_p);

	//printf("The error is %f\n", error);

	return 0; 

} /*main*/

