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
		float** hx_p  /*out*/,
		float** hy_p  /*out*/,
		float** hz_p  /*out*/,
		float** cz_p /*out*/,

		float** dx_p /*out*/,
		float** dy_p /*out*/,
		float** dz_p /*out*/,
		int n        /* in */ ) {

	/* dx ,yd , and dz are used on device */
	cudaMalloc(dx_p, n*sizeof(float));
	cudaMalloc(dy_p, n*sizeof(float));
	cudaMalloc(dz_p, n*sizeof(float));

	/*hx, hy, hz and cz are used on host*/
	*hx_p = (float*) malloc(n*sizeof(float));
	*hy_p = (float*) malloc(n*sizeof(float));
	*hz_p = (float*) malloc(n*sizeof(float));
	*cz_p = (float*) malloc(n*sizeof(float));

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
		float* hx /* in/out */,
		float* hy /* in/out*/,
		float* hz /*in/out*/,
		float* cz /*in/out*/,
     		float* dx /*in/out*/,
	        float* dy /*in/out*/,
		float* dz /*in/out*/){	
	
	/* Allocated with cidaMallocManaged*/

	cudaFree(dx);
	cudaFree(dy);
	cudaFree(dz);

	/*Allocated with malloc */
	free(hx);
	free(hy);
	free(hz);

	free(cz);

}/*Free_vectors*/



int main(int argc, char* argv[]){

	int n, th_per_blk, blk_ct;
	char i_g; /*Are x and y user input or random? */
	float *hx, *hy, *hz, *cz; /*Host arrays*/
	float *dx, *dy, *dz; /*Device arrays*/
	double diff_norm; 

	/*Ge the command line arguments, and set up vectors*/
	Get_args(argc, argv, &n, &blk_ct, &th_per_blk, &i_g);
	Allocate_vectors(&hx, &hy, &hz, &cz, &dx, &dy, &dz, n);
	/*Init_vectors(x, y, n,i_g);*/

	/*Initialize the host vectors hx and hy*/
	for (int i = 0; i <n; i++){
		hx[i] = rand()/(float)RAND_MAX;
		hy[i] = rand()/(float)RAND_MAX; 
	}

	/*copy vectors hx and hy to device*/
	cudaMemcpy(dx, hx, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dy, hy, n*sizeof(float), cudaMemcpyHostToDevice);

	/*Invoke kernel*/
	Vec_add <<<blk_ct, th_per_blk>>>(dx, dy, dz, n);
	
	/*Wait for kernel to finish and copy result to host*/
	cudaMemcpy(hz, dz, n*sizeof(float), cudaMemcpyDeviceToHost);

	/*Check for correctness*/
	Serial_vec_add(hx, hy, cz, n);
	diff_norm = Two_norm_diff(hz, cz, n);
	printf("Two-norm of difference between host and ");
	printf("device = %e\n", diff_norm); 


	/*Free storage and quit*/
	Free_vectors(hx, hy, hz, cz, dx, dy, dz);

	return 0; 

} /*main*/

