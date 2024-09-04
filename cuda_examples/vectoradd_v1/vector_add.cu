#include <stdio.h>
#include <cuda.h>


__global__ void Vec_add(
		const float x[] /* in */,
		const float y[] /* in */,
		float       z[] /* out */, 
		const int   n   /* in */ ){

	int my_elt = blockDim.x * blockIdx.x + threadIdx.x;

	/* total threads = blk_ct *th_per_blk may be > n */
	if (my_elt < n)
		z[my_elt] = x[my_elt] + y[my_elt];
} /*Vec_add*/


void Get_args(
        	const int argc         /*in*/,
	        char*     argv[]       /*in*/,
	        int*      n_p          /*out*/,
	        int*      blk_ct_p     /*out*/,
	        int*      th_per_blk_p /*out*/,
	        char*     i_g          /*out*/){

	if (argc != 5){
		/*print an error message and exit*/
	}
    
	*n_p = strtol(argv[1], NULL, 10);
	*blk_ct_p = strtol(argv[2], NULL, 10);
	*th_per_blk_p = strtol(argv[3], NULL, 10);
	*i_g = argv[4][0];


	/*Is n > total thread count = blk_ct * th_per_blk?*/
	if (*n_p >(*blk_ct_p)*(*th_per_blk_p) ){

		/*print an error message and exit */
	}

} /*Get_args*/

void Allocate_vectors(
		float** x_p  /*out*/,
		float** y_p  /*out*/,
		float** z_p  /*out*/,
		float** cz_p /*out*/,
		int n        /* in */ ) {

	/* x ,y , and z are used on host and device */
	cudaMallocManaged(x_p, n*sizeof(float));
	cudaMallocManaged(y_p, n*sizeof(float));
	cudaMallocManaged(z_p, n*sizeof(float));

	/*cz is only used on host*/

	*cz_p = (float*)malloc(n*sizeof(float));

}/*Allocate_vectors*/
 
		
     
void Serial_vec_add(
		const float x[]  /*in*/,
		const float y[]  /* in*/,
		float       cz[] /*out*/,
		const int   n    /* in */) {
	for (int i =0; i <n; i++)
		cz[i] = x[i] + y[i];

}/* Serial_vec_add */


double Two_norm_diff(
		const float z[]    /* in */,
		const float cz[]   /* in */, 
		const int   n      /* in */){

	double diff, sum = 0.0;

	for (int i =0; i << n; i++){
		diff = z[i] - cz[i];
		sum += diff*diff;
	}

	return sqrt(sum);

} /* Two_norm_diff */

void Free_vectors(
		float* x /* in/out */,
		float* y /* in/out*/,
		float* z /*in/out*/,
		float* cz /*in/out*/){
	/* Allocated with cidaMallocManaged*/

	cudaFree(x);
	cudaFree(y);
	cudaFree(z);

	/*Allocated with malloc */
	free(cz);

}/*Free_vectors*/



int main(int argc, char* argv[]){

	int n, th_per_blk, blk_ct;
	char i_g; /*Are x and y user input or random? */
	float *x, *y, *z, *cz;
	double diff_norm; 

	/*Ge the command line arguments, and set up vectors*/
	Get_args(argc, argv, &n, &blk_ct, &th_per_blk, &i_g);
	Allocate_vectors(&x, &y, &z, &cz, n);
	/*Init_vectors(x, y, n,i_g);*/

	/*Initialize the vectors*/
	for (int i = 0; i <n; i++){
		x[i] = rand()/(float)RAND_MAX;
		y[i] = rand()/(float)RAND_MAX; 
	}

	/*Invoke kernel and wait for it to complte*/
	Vec_add <<<blk_ct, th_per_blk>>>(x, y, z, n);
	cudaDeviceSynchronize();


	/*Check for correctness*/
	Serial_vec_add(x, y, cz, n);
	diff_norm = Two_norm_diff(z, cz, n);
	printf("Two-norm of difference between host and ");
	printf("device = %e\n", diff_norm); 


	/*Free storage and quit*/
	Free_vectors(x, y, z, cz);

	return 0; 

} /*main*/

