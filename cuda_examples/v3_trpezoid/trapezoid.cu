/*We implement trpezoidal rule using GPU Threads. To accelerate the sum, we used shared memory and the thread blocks containe only a single warp ie 32 threads.*/

#include <stdio.h>
#include <cuda.h>


#define WARPSZ 32

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
 
__global__ void Dev_trap(
		const float a        /*in*/,
		const float b        /*in*/,
		const float h        /*in*/,
		const int   n        /*in*/,
		float*      trap_p   /*in/out*/){

        //allocate this much memory on the SM running this block
       	__shared__ float shared_vals[WARPSZ];

  	int my_i = blockDim.x * blockIdx.x + threadIdx.x;
        int my_lane = threadIdx.x % warpSize;	

	/*f(x_0) and f(x_n) were computed on the host. So compute
	 f(x_1), f(x_2), ..., f(x_(n-1))*/

	/*for threads in the last warp/block who will skip the following conditional
	but are gonna participate in the shared memory sum*/	
	shared_vals[my_lane] = 0.0f;


	if (0 < my_i && my_i < n){
		float my_x = a + my_i*h;
		shared_vals[my_lane] = func(my_x);
		
	}

	float result = Shared_mem_sum(shared_vals);

	/*the result is same on all threads in a block*/
	if (threadIdx.x == 0) atomicAdd(trap_p, result); 
 
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

   	Dev_trap<<<blk_ct, 32>>>(a, b, h, n, trap_p);
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

