#include <stdio.h>
#include <cuda.h> /*CUDA header filer*/

/* Device code */

__global__ void Hello(void){
	printf("Hello from thread %d!\n", threadIdx.x);
} /*Hello*/


/*Host code: Runs on CPU */

int main(int argc, char* argv[]){
	int thread_count; /*Number of threads to run on GPU*/
	int blk_count; /*Number of blocks to use*/



	blk_count = strtol(argv[1],NULL, 10);
	/*Get blk_count from commmand line*/


	thread_count = strtol(argv[2], NULL, 10);
	/*Get thread_count from command line*/

	Hello <<<blk_count, thread_count>>>(); /*Start thread_count threads on GPU*/

	cudaDeviceSynchronize(); /*Wait for GPU to finish*/

	return 0;

} /*main*/
