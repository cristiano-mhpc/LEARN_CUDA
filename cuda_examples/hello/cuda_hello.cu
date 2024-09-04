#include <stdio.h>
#include <assert.h>
#include <cuda.h> /*CUDA header filer*/
#include <cuda_runtime.h>

/* Device code */

__global__ void Hello(void){
	printf("Hello from thread %d!\n", threadIdx.x);
} /*Hello*/


/*Host code: Runs on CPU */

int main(int argc, char* argv[]){
	int thread_count; /*Number of threads to run on GPU*/

	//thread_count = strtol(argv[1], NULL, 10);
	/*Get thread_count from command line*/

	thread_count = 10;
	Hello <<<2, thread_count>>>(); /*Start thread_count threads on GPU*/
	cudaDeviceSynchronize(); /*Wait for GPU to finish*/

	return 0;

} /*main*/
