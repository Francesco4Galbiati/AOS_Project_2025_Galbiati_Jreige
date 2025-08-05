#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include "CUSPIS.cuh"

#define NUM_REPLICAS CUSPIS::NUM_REPLICAS
#define NUMBER_PAR_PER_BOX 192
#define NUMBER_THREADS 192
#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))
#define MAX_LOGGED_ERRORS_PER_STREAM 100
#define GENERATE 0

#define PRECISION_DOUBLE
// #define PRECISION_SINGLE
// #define PRECISION_HALF

#ifdef PRECISION_HALF 
#include "half.hpp"
#include <cuda_fp16.h>
#endif

#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)

#if defined(PRECISION_DOUBLE)

const char test_precision_description[] = "double";
typedef double tested_type;
typedef double tested_type_host;

#elif defined(PRECISION_SINGLE)

const char test_precision_description[] = "single";
typedef float tested_type;
typedef float tested_type_host;

#elif defined(PRECISION_HALF)

#define H2_DOT(A,B) (__hfma2((A.x), (B.x), __hfma2((A.y), (B.y), __hmul2((A.z), (B.z)))))

const char test_precision_description[] = "half";
typedef half tested_type;
typedef half_float::half tested_type_host;

#else 
#error TEST TYPE NOT DEFINED OR INCORRECT. USE PRECISION=<double|single|half>.
#endif

typedef struct {
	tested_type x, y, z;
} THREE_VECTOR;

typedef struct {
	tested_type v, x, y, z;
} FOUR_VECTOR;

typedef struct {
	tested_type_host x, y, z;
} THREE_VECTOR_HOST;

typedef struct {
	tested_type_host v, x, y, z;
} FOUR_VECTOR_HOST;

__host__ inline bool operator==(const FOUR_VECTOR_HOST& lhs,
		const FOUR_VECTOR_HOST& rhs) {
	return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z)
			&& (lhs.v == rhs.v);
}
__host__ inline bool operator!=(const FOUR_VECTOR_HOST& lhs,
		const FOUR_VECTOR_HOST& rhs) {
	return !operator==(lhs, rhs);
}

#if defined(PRECISION_HALF)
typedef struct
{
	half2 x, y, z;
}THREE_H2_VECTOR;

typedef struct
{
	half2 v, x, y, z;
}FOUR_H2_VECTOR;
#endif

typedef struct nei_str {
	// neighbor box
	int x, y, z;
	int number;
	long offset;
} nei_str;

typedef struct box_str {
	// home box
	int x, y, z;
	int number;
	long offset;
	// neighbor boxes
	int nn;
	nei_str nei[26];
} box_str;

typedef struct par_str {
	tested_type alpha;
} par_str;

typedef struct dim_str {
	// input arguments
	int cur_arg;
	int arch_arg;
	int cores_arg;
	int boxes1d_arg;
	// system memory
	long number_boxes;
	long box_mem;
	long space_elem;
	long space_mem;
	long space_mem2;
} dim_str;

void __checkFrameworkErrors(cudaError_t error, int line, char* file) {
	if (error != cudaSuccess) {
		printf("CUDA Framework error: %s. Bailing.", cudaGetErrorString(error));
		exit (EXIT_FAILURE);
	}
}

double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void generateInput(dim_str dim_cpu, char *input_distances,
		FOUR_VECTOR_HOST **rv_cpu, char *input_charges,
		tested_type_host **qv_cpu) {
	// random generator seed set to random value - time in this case
	FILE *fp;
	int i;

	printf("Generating input...\n");

	srand (time(NULL));

	// input (distances)
	if(	(fp = fopen(input_distances, "wb" )) == 0 ) {
		printf( "The file 'input_distances' was not opened\n" ); exit(EXIT_FAILURE);
	}
	*rv_cpu = (FOUR_VECTOR_HOST*) malloc(dim_cpu.space_mem);
	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		// get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].v = (tested_type_host)(rand() % 10 + 1) / 10.0;
		fwrite(&((*rv_cpu)[i].v), 1, sizeof(tested_type), fp);
		// get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].x = (tested_type_host)(rand() % 10 + 1) / 10.0;
		fwrite(&((*rv_cpu)[i].x), 1, sizeof(tested_type), fp);
		// get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].y = (tested_type_host)(rand() % 10 + 1) / 10.0;
		fwrite(&((*rv_cpu)[i].y), 1, sizeof(tested_type), fp);
		// get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].z = (tested_type_host)(rand() % 10 + 1) / 10.0;
		fwrite(&((*rv_cpu)[i].z), 1, sizeof(tested_type), fp);
	}
	fclose(fp);

	// input (charge)
	if ((fp = fopen(input_charges, "wb")) == 0) {
		printf("The file 'input_charges' was not opened\n");
		exit (EXIT_FAILURE);
	}

	*qv_cpu = (tested_type_host*) malloc(dim_cpu.space_mem2);
	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		// get a number in the range 0.1 - 1.0
		(*qv_cpu)[i] = (tested_type_host)(rand() % 10 + 1) / 10.0;
		fwrite(&((*qv_cpu)[i]), 1, sizeof(tested_type), fp);
	}
	fclose(fp);
}

void readInput(dim_str dim_cpu, char *input_distances,
		FOUR_VECTOR_HOST **rv_cpu, char *input_charges,
		tested_type_host **qv_cpu, int fault_injection) {
	FILE *fp;
	int i;
	size_t return_value[4];
	// size_t return_value;

	// input (distances)
	if ((fp = fopen(input_distances, "rb")) == 0) {
		printf("The file 'input_distances' was not opened\n");
		exit (EXIT_FAILURE);
	}

	*rv_cpu = (FOUR_VECTOR_HOST*) malloc(dim_cpu.space_mem);
	if (*rv_cpu == NULL) {
		printf("error rv_cpu malloc\n");
		exit(1);
	}

	// return_value = fread(*rv_cpu, sizeof(FOUR_VECTOR_HOST), dim_cpu.space_elem, fp);
	// if (return_value != dim_cpu.space_elem) {
	// 	printf("error reading rv_cpu from file\n");
	// 	#ifdef LOGS
	// 		log_error_detail((char *)"error reading rv_cpu from file"); end_log_file();
	// 	#endif
	// 	exit(1);
	// }

	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		return_value[0] = fread(&((*rv_cpu)[i].v), 1, sizeof(tested_type), fp);
		return_value[1] = fread(&((*rv_cpu)[i].x), 1, sizeof(tested_type), fp);
		return_value[2] = fread(&((*rv_cpu)[i].y), 1, sizeof(tested_type), fp);
		return_value[3] = fread(&((*rv_cpu)[i].z), 1, sizeof(tested_type), fp);
		if (return_value[0] == 0 || return_value[1] == 0 || return_value[2] == 0
				|| return_value[3] == 0) {
			printf("error reading rv_cpu from file\n");
			exit(1);
		}
	}
	fclose(fp);

	// input (charge)
	if ((fp = fopen(input_charges, "rb")) == 0) {
		printf("The file 'input_charges' was not opened\n");
		exit (EXIT_FAILURE);
	}

	*qv_cpu = (tested_type_host*) malloc(dim_cpu.space_mem2);
	if (*qv_cpu == NULL) {
		printf("error qv_cpu malloc\n");
		exit(1);
	}

	// return_value = fread(*qv_cpu, sizeof(tested_type_host), dim_cpu.space_elem, fp);
	// if (return_value != dim_cpu.space_elem) {
	// 	printf("error reading qv_cpu from file\n");
	// 	#ifdef LOGS
	// 		log_error_detail((char *)"error reading qv_cpu from file"); end_log_file();
	// 	#endif
	// 	exit(1);
	// }

	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		return_value[0] = fread(&((*qv_cpu)[i]), 1, sizeof(tested_type), fp);
		if (return_value[0] == 0) {
			printf("error reading qv_cpu from file\n");
			exit(1);
		}
	}
	fclose(fp);

	// =============== Fault injection
	if (fault_injection) {
		(*qv_cpu)[2] = 0.732637263; // must be in range 0.1 - 1.0
		printf("!!> Fault injection: qv_cpu[2]=%f\n", (double) (*qv_cpu)[2]);
	}
	// ========================
}

void readGold(dim_str dim_cpu, char *output_gold,
		FOUR_VECTOR_HOST *fv_cpu_GOLD) {
	FILE *fp;
	size_t return_value[4];
	// size_t return_value;
	int i;

	if ((fp = fopen(output_gold, "rb")) == 0) {
		printf("The file 'output_forces' was not opened\n");
		exit (EXIT_FAILURE);
	}

	// return_value = fread(fv_cpu_GOLD, sizeof(FOUR_VECTOR_HOST), dim_cpu.space_elem, fp);
	// if (return_value != dim_cpu.space_elem) {
	// 	printf("error reading rv_cpu from file\n");
	// 	#ifdef LOGS
	// 		log_error_detail((char *)"error reading rv_cpu from file"); end_log_file();
	// 	#endif
	// 	exit(1);
	// }
	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		return_value[0] = fread(&(fv_cpu_GOLD[i].v), 1, sizeof(tested_type),
				fp);
		return_value[1] = fread(&(fv_cpu_GOLD[i].x), 1, sizeof(tested_type),
				fp);
		return_value[2] = fread(&(fv_cpu_GOLD[i].y), 1, sizeof(tested_type),
				fp);
		return_value[3] = fread(&(fv_cpu_GOLD[i].z), 1, sizeof(tested_type),
				fp);
		if (return_value[0] == 0 || return_value[1] == 0 || return_value[2] == 0
				|| return_value[3] == 0) {
			printf("error reading rv_cpu from file\n");
			exit(1);
		}
	}
	fclose(fp);
}

void writeGold(dim_str dim_cpu, char *output_gold, FOUR_VECTOR_HOST **fv_cpu) {
	FILE *fp, *fp_txt;
	int i;
  /*
	if ((fp = fopen(output_gold, "wb")) == 0) {
		printf("The file 'output_forces' was not opened\n");
		exit (EXIT_FAILURE);
	}
  */
	if ((fp_txt = fopen("outpud_cuspis_test.txt", "w")) == 0) {
		printf("The file 'output_forces' was not opened\n");
		exit (EXIT_FAILURE);
	}

	int number_zeros = 0;
	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		if ((*fv_cpu)[i].v == tested_type_host(0.0))
			number_zeros++;
		if ((*fv_cpu)[i].x == tested_type_host(0.0))
			number_zeros++;
		if ((*fv_cpu)[i].y == tested_type_host(0.0))
			number_zeros++;
		if ((*fv_cpu)[i].z == tested_type_host(0.0))
			number_zeros++;
    /*
		fwrite(&((*fv_cpu)[i].v), 1, sizeof(tested_type), fp);
		fwrite(&((*fv_cpu)[i].x), 1, sizeof(tested_type), fp);
		fwrite(&((*fv_cpu)[i].y), 1, sizeof(tested_type), fp);
		fwrite(&((*fv_cpu)[i].z), 1, sizeof(tested_type), fp);
		*/
		fprintf(fp_txt, "v: %f, x: %f, y: %f, z: %f\n", ((*fv_cpu)[i].v), ((*fv_cpu)[i].x), ((*fv_cpu)[i].y), ((*fv_cpu)[i].z));
		
	}
	fclose(fp_txt);
}

void gpu_memory_setup(int nstreams, bool gpu_check, dim_str dim_cpu,
		box_str **d_box_gpu, box_str *box_cpu, FOUR_VECTOR **d_rv_gpu,
		FOUR_VECTOR_HOST *rv_cpu, tested_type **d_qv_gpu,
		tested_type_host *qv_cpu, FOUR_VECTOR **d_fv_gpu,//,		FOUR_VECTOR_HOST *fv_cpu
		FOUR_VECTOR *d_fv_gold_gpu, FOUR_VECTOR_HOST *fv_cpu_GOLD) {

	for (int streamIdx = 0; streamIdx < nstreams; streamIdx++) {
		//=====================================================================
		//	GPU SETUP MEMORY
		//=====================================================================

		//==================================================
		//	boxes
		//==================================================
		checkFrameworkErrors(CUSPIS::cuspisMalloc((void ** )&(d_box_gpu[streamIdx]), dim_cpu.box_mem));
		//==================================================
		//	rv
		//==================================================
		checkFrameworkErrors(CUSPIS::cuspisMalloc((void ** )&(d_rv_gpu[streamIdx]), dim_cpu.space_mem));
		//==================================================
		//	qv
		//==================================================
		checkFrameworkErrors(CUSPIS::cuspisMalloc((void ** )&(d_qv_gpu[streamIdx]), dim_cpu.space_mem2));

		//==================================================
		//	fv
		//==================================================
		checkFrameworkErrors(CUSPIS::cuspisMalloc((void ** )&(d_fv_gpu[streamIdx]), dim_cpu.space_mem));

		//=====================================================================
		//	GPU MEMORY			COPY
		//=====================================================================

		//==================================================
		//	boxes
		//==================================================

		checkFrameworkErrors(CUSPIS::cuspisMemcpyToDevice(d_box_gpu[streamIdx], box_cpu, dim_cpu.box_mem));
		//==================================================
		//	rv
		//==================================================

		checkFrameworkErrors(CUSPIS::cuspisMemcpyToDevice(d_rv_gpu[streamIdx], rv_cpu, dim_cpu.space_mem));
		//==================================================
		//	qv
		//==================================================

		checkFrameworkErrors(CUSPIS::cuspisMemcpyToDevice(d_qv_gpu[streamIdx], qv_cpu, dim_cpu.space_mem2));
		//==================================================
		//	fv
		//==================================================

		// This will be done with memset at the start of each iteration.
		// checkFrameworkErrors( cudaMemcpy( d_fv_gpu[streamIdx], fv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice) );
	}

	//==================================================
	//	fv_gold for GoldChkKernel
	//==================================================
	if (gpu_check) {
		checkFrameworkErrors(CUSPIS::cuspisMalloc((void** )&d_fv_gold_gpu, dim_cpu.space_mem));
		checkFrameworkErrors(CUSPIS::cuspisMemcpyToDevice(d_fv_gold_gpu, fv_cpu_GOLD, dim_cpu.space_mem2));
	}
}

void gpu_memory_unset(int nstreams, int gpu_check, box_str **d_box_gpu,
		FOUR_VECTOR **d_rv_gpu, tested_type **d_qv_gpu, FOUR_VECTOR **d_fv_gpu,
		FOUR_VECTOR *d_fv_gold_gpu) {

	//=====================================================================
	//	GPU MEMORY DEALLOCATION
	//=====================================================================
	for (int streamIdx = 0; streamIdx < nstreams; streamIdx++) {
		CUSPIS::cuspisFree(&d_rv_gpu[streamIdx]);
		CUSPIS::cuspisFree(&d_qv_gpu[streamIdx]);
		CUSPIS::cuspisFree(&d_fv_gpu[streamIdx]);
		CUSPIS::cuspisFree(&d_box_gpu[streamIdx]);
	}
	if (gpu_check) {
		CUSPIS::cuspisFree(&d_fv_gold_gpu);
	}
}

__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
		box_str* d_box_gpu, FOUR_VECTOR* d_rv_gpu, tested_type* d_qv_gpu,
		FOUR_VECTOR* d_fv_gpu) {

  

	//---------------------------------------------------------------------
	//	THREAD PARAMETERS
	//---------------------------------------------------------------------

	int bx = blockIdx.x;		 // get current horizontal block index (0-n)
	int tx = threadIdx.x;		 // get current horizontal thread index (0-n)
	int wtx = tx;
	
	int block_replica = bx / d_dim_gpu.number_boxes + tx / NUMBER_THREADS;
	int thread_replica = tx / NUMBER_THREADS;

	//---------------------------------------------------------------------
	//	DO FOR THE NUMBER OF BOXES
	//---------------------------------------------------------------------

	if (bx < d_dim_gpu.number_boxes * NUM_REPLICAS) {

		//-------------------------------------------------------------
		//	Extract input parameters
		//-------------------------------------------------------------

		// parameters
		tested_type a2 = tested_type(2.0) * d_par_gpu.alpha * d_par_gpu.alpha;

		// home box
		int first_i;
		FOUR_VECTOR *rA;
		FOUR_VECTOR *fA;
		__shared__ FOUR_VECTOR rA_shared[200 * NUM_REPLICAS];

		// nei box
		int pointer;
		int k = 0;
		int first_j;
		FOUR_VECTOR *rB;
		tested_type *qB;
		int j = 0;
		__shared__ FOUR_VECTOR rB_shared[200 * NUM_REPLICAS];
		__shared__ tested_type qB_shared[200 * NUM_REPLICAS];

		// common
		tested_type r2;
		tested_type u2;
		tested_type vij;
		tested_type fs;
		tested_type fxij;
		tested_type fyij;
		tested_type fzij;
		THREE_VECTOR d;

		//-------------------------------------------------------------
		//	Home box
		//-------------------------------------------------------------

		//-------------------------------------------------------------
		//	Setup parameters
		//-------------------------------------------------------------

		// home box - box parameters
		first_i = d_box_gpu[bx + thread_replica * d_dim_gpu.number_boxes].offset + (block_replica * NUMBER_PAR_PER_BOX * d_dim_gpu.number_boxes);

		// home box - distance, force, charge and type parameters
		rA = &d_rv_gpu[first_i];
		fA = &d_fv_gpu[first_i];

		//-------------------------------------------------------------
		//	Copy to shared memory
		//-------------------------------------------------------------

		// home box - shared memory
		while (wtx < NUMBER_PAR_PER_BOX * NUM_REPLICAS) {
			rA_shared[wtx] = rA[wtx % NUMBER_PAR_PER_BOX];
			wtx = wtx + NUMBER_THREADS * NUM_REPLICAS;
		}
		wtx = tx;

		// synchronize threads - not needed, but just to be safe
		__syncthreads();

		//-------------------------------------------------------------
		//	nei box loop
		//-------------------------------------------------------------

		// loop over neiing boxes of home box
		for (k = 0; k < (1 + d_box_gpu[bx + thread_replica * d_dim_gpu.number_boxes].nn); k++) {

			//---------------------------------------------
			//	nei box - get pointer to the right box
			//---------------------------------------------

			if (k == 0) {
				pointer = bx;	 // set first box to be processed to home box
			} else {
				// remaining boxes are nei boxes
				pointer = d_box_gpu[bx + thread_replica * d_dim_gpu.number_boxes].nei[k - 1].number;
			}

			//-----------------------------------------------------
			//	Setup parameters
			//-----------------------------------------------------

			// nei box - box parameters
			first_j = d_box_gpu[pointer + thread_replica * d_dim_gpu.number_boxes].offset + (block_replica * NUMBER_PAR_PER_BOX * d_dim_gpu.number_boxes);

			// nei box - distance, (force), charge and (type) parameters
			rB = &d_rv_gpu[first_j];
			qB = &d_qv_gpu[first_j];

			//-----------------------------------------------------
			//	Setup parameters
			//-----------------------------------------------------

			// nei box - shared memory
			while (wtx < NUMBER_PAR_PER_BOX * NUM_REPLICAS) {
				rB_shared[wtx] = rB[wtx % NUMBER_PAR_PER_BOX];
				qB_shared[wtx] = qB[wtx % NUMBER_PAR_PER_BOX];
				wtx = wtx + NUMBER_PAR_PER_BOX * NUM_REPLICAS;
			}
			wtx = tx;

			// synchronize threads because in next section each thread accesses data brought in by different threads here
			__syncthreads();

			//-----------------------------------------------------
			//	Calculation
			//-----------------------------------------------------
			while (wtx < NUMBER_PAR_PER_BOX * NUM_REPLICAS) {

				for (j = NUMBER_THREADS * thread_replica; j < NUMBER_PAR_PER_BOX + (thread_replica * NUMBER_PAR_PER_BOX); j++) {

					r2 = rA_shared[wtx].v + rB_shared[j].v -
					DOT(
							rA_shared[wtx],
							rB_shared[j]
					);

					u2 = a2 * r2;
#if defined(PRECISION_DOUBLE) or defined(PRECISION_SINGLE)
					vij= exp(-u2);
#elif defined(PRECISION_HALF)
					vij= hexp(-u2);
#endif
					fs = tested_type(2.0) * vij;

					d.x = rA_shared[wtx].x - rB_shared[j].x;

					fxij = fs * d.x;

					d.y = rA_shared[wtx].y - rB_shared[j].y;

					fyij = fs * d.y;

					d.z = rA_shared[wtx].z - rB_shared[j].z;

					fzij = fs * d.z;

					fA[wtx % NUMBER_PAR_PER_BOX].v += (tested_type)(qB_shared[j] * vij);
					fA[wtx % NUMBER_PAR_PER_BOX].x += (tested_type)(qB_shared[j] * fxij);
					fA[wtx % NUMBER_PAR_PER_BOX].y += (tested_type)(qB_shared[j] * fyij);
					fA[wtx % NUMBER_PAR_PER_BOX].z += (tested_type)(qB_shared[j] * fzij);
				}
				// increment work thread index
				wtx = wtx + NUMBER_PAR_PER_BOX * NUM_REPLICAS;
			}

			// reset work index
			wtx = tx;

			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
			__syncthreads();
		}
	}
}

int main(int argc, char* argv[]){
	
	// timer
	double timestamp;

	// counters
	int i, j, k, l, m, n;
	int iterations = 5;
	int generate = GENERATE;
	int verbose = 1;
	int fault_injection = 0;
	int	gpu_check = 0;

	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str *box_cpu;
	FOUR_VECTOR_HOST *rv_cpu;
	tested_type_host *qv_cpu;
	FOUR_VECTOR_HOST *fv_cpu_GOLD;
	int nh;
	int nstreams = 1;
	int	streamIdx;
	char input_distances[] = "lava_single_distances_23";
	char input_charges[] = "lava_single_charges_23";
	char output_gold[] = "lava_single_gold_23_cuspis";
	int number_nn = 0;
	
	printf("Number of replicas: %d\n", NUM_REPLICAS);
	
	dim_cpu.boxes1d_arg = 23;
	
	char test_info[200];
	char test_name[200];
	snprintf(test_info, 200,
			"type:%s-precision streams:%d boxes:%d block_size:%d",
			test_precision_description, nstreams, dim_cpu.boxes1d_arg,
			NUMBER_THREADS);
	snprintf(test_name, 200, "cuda_%s_lava", test_precision_description);
	printf(
			"\n=================================\n%s\n%s\n=================================\n\n",
			test_name, test_info);
	
	// INPUTS
	par_cpu.alpha = 0.5;
	
	// DIMENSIONS
	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;
	// total number of particles
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	// total space occupied by the particles
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(tested_type);
	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);
	
	// SYSTEM MEMORY
	// prepare host memory to receive kernel output - output (forces)
	FOUR_VECTOR_HOST *fv_cpu[nstreams];
	for (int streamIdx = 0; streamIdx < nstreams; streamIdx++) {
		fv_cpu[streamIdx] = (FOUR_VECTOR_HOST*) malloc(dim_cpu.space_mem);
		if (fv_cpu[streamIdx] == NULL) {
			printf("error fv_cpu malloc\n");
			exit(1);
		}
	}
	
	fv_cpu_GOLD = (FOUR_VECTOR_HOST*) malloc(dim_cpu.space_mem);
	if (fv_cpu_GOLD == NULL) {
		printf("error fv_cpu_GOLD malloc\n");
		exit(1);
	}

	// BOX
	// allocate boxes
	box_cpu = (box_str*) malloc(dim_cpu.box_mem);
	if (box_cpu == NULL) {
		printf("error box_cpu malloc\n");
		exit(1);
	}
	// initialize number of home boxes
	nh = 0;
	// home boxes in z direction
	for (i = 0; i < dim_cpu.boxes1d_arg; i++) {
		// home boxes in y direction
		for (j = 0; j < dim_cpu.boxes1d_arg; j++) {
			// home boxes in x direction
			for (k = 0; k < dim_cpu.boxes1d_arg; k++) {

				// current home box
				box_cpu[nh].x = k;
				box_cpu[nh].y = j;
				box_cpu[nh].z = i;
				box_cpu[nh].number = nh;
				box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

				// initialize number of neighbor boxes
				box_cpu[nh].nn = 0;

				// neighbor boxes in z direction
				for (l = -1; l < 2; l++) {
					// neighbor boxes in y direction
					for (m = -1; m < 2; m++) {
						// neighbor boxes in x direction
						for (n = -1; n < 2; n++) {

							// check if (this neighbor exists) and (it is not the same as home box)
							if ((((i + l) >= 0 && (j + m) >= 0 && (k + n) >= 0)
									== true
									&& ((i + l) < dim_cpu.boxes1d_arg
											&& (j + m) < dim_cpu.boxes1d_arg
											&& (k + n) < dim_cpu.boxes1d_arg)
											== true)
									&& (l == 0 && m == 0 && n == 0) == false) {

								// current neighbor box
								box_cpu[nh].nei[box_cpu[nh].nn].x = (k + n);
								box_cpu[nh].nei[box_cpu[nh].nn].y = (j + m);
								box_cpu[nh].nei[box_cpu[nh].nn].z = (i + l);
								box_cpu[nh].nei[box_cpu[nh].nn].number =
										(box_cpu[nh].nei[box_cpu[nh].nn].z
												* dim_cpu.boxes1d_arg
												* dim_cpu.boxes1d_arg)
												+ (box_cpu[nh].nei[box_cpu[nh].nn].y
														* dim_cpu.boxes1d_arg)
												+ box_cpu[nh].nei[box_cpu[nh].nn].x;
								box_cpu[nh].nei[box_cpu[nh].nn].offset =
										box_cpu[nh].nei[box_cpu[nh].nn].number
												* NUMBER_PAR_PER_BOX;

								// increment neighbor box
								box_cpu[nh].nn = box_cpu[nh].nn + 1;
								number_nn += box_cpu[nh].nn;

							}

						}	 // neighbor boxes in x direction
					}		 // neighbor boxes in y direction
				}			 // neighbor boxes in z direction

				// increment home box
				nh = nh + 1;

			}				 // home boxes in x direction
		}					 // home boxes in y direction
	}

	// PARAMETERS, DISTANCE, CHARGE AND FORCE
	if (generate) {
		generateInput(dim_cpu, input_distances, &rv_cpu, input_charges,
				&qv_cpu);
	} else {
		readInput(dim_cpu, input_distances, &rv_cpu, input_charges, &qv_cpu,
				fault_injection);
	}
	
	// EXECUTION PARAMETERS
	dim3 threads;
	dim3 blocks;
	blocks.x = dim_cpu.number_boxes;
	blocks.y = 1;
	// define the number of threads in the block
	threads.x = NUMBER_THREADS;
	threads.y = 1;
	
	// GPU_CUDA
	
	// STREAMS
	cudaStream_t *streams = (cudaStream_t *) malloc(
			nstreams * sizeof(cudaStream_t));
	for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
		checkFrameworkErrors(
				cudaStreamCreateWithFlags(&(streams[streamIdx]),
						cudaStreamNonBlocking));
	}
	
	// VECTORS
	box_str *d_box_gpu[nstreams];
	FOUR_VECTOR *d_rv_gpu[nstreams];
	tested_type *d_qv_gpu[nstreams];
	FOUR_VECTOR *d_fv_gpu[nstreams];
	FOUR_VECTOR *d_fv_gold_gpu;
	
	// GPU MEMORY SETUP
	gpu_memory_setup(nstreams, gpu_check, dim_cpu, d_box_gpu, box_cpu, d_rv_gpu,
			rv_cpu, d_qv_gpu, qv_cpu, d_fv_gpu, d_fv_gold_gpu, fv_cpu_GOLD);
			
	// LOOP START
	int loop;
	for (loop = 0; loop < iterations; loop++) {

		if (verbose)
			printf("======== Iteration #%06u ========\n", loop);

		double globaltimer = mysecond();
		timestamp = mysecond();

		// for(i=0; i<dim_cpu.space_elem; i=i+1) {
		// 	// set to 0, because kernels keeps adding to initial value
		// 	fv_cpu[i].v = tested_type_host(0.0);
		// 	fv_cpu[i].x = tested_type_host(0.0);
		// 	fv_cpu[i].y = tested_type_host(0.0);
		// 	fv_cpu[i].z = tested_type_host(0.0);
		// }

		//=====================================================================
		//	GPU SETUP
		//=====================================================================
		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
			memset(fv_cpu[streamIdx], 0x00, dim_cpu.space_elem);
			checkFrameworkErrors(cudaMemset(d_fv_gpu[streamIdx], 0x00, dim_cpu.space_mem * NUM_REPLICAS));
		}

		if (verbose)
			printf("Setup prepare time: %.4fs\n", mysecond() - timestamp);

		//=====================================================================
		//	KERNEL
		//=====================================================================

		double kernel_time = mysecond();
		// launch kernel - all boxes
		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
			CUSPIS::Kernel<par_str, dim_str, box_str*, FOUR_VECTOR*, tested_type*, FOUR_VECTOR*> 
				kernel_b(blocks.x, threads.x, 0, streams[streamIdx], kernel_gpu_cuda, CUSPIS::cuspisRedundantKernel);
			kernel_b.launch(par_cpu, dim_cpu, d_box_gpu[streamIdx], d_rv_gpu[streamIdx], 
				d_qv_gpu[streamIdx], d_fv_gpu[streamIdx]);
			checkFrameworkErrors(cudaPeekAtLastError());
		}
		
		//printf("All kernels were commited.\n");
		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
			checkFrameworkErrors(cudaStreamSynchronize(streams[streamIdx]));
			checkFrameworkErrors(cudaPeekAtLastError());
			kernel_time = mysecond() - kernel_time;
		}
		

		//=====================================================================
		//	COMPARE OUTPUTS / WRITE GOLD
		//=====================================================================
		if (generate) {
			checkFrameworkErrors(CUSPIS::cuspisMemcpyToHost(fv_cpu_GOLD, d_fv_gpu[0], dim_cpu.space_mem));
			writeGold(dim_cpu, output_gold, &fv_cpu_GOLD);
		} else {
			checkFrameworkErrors(CUSPIS::cuspisMemcpyToHost(fv_cpu_GOLD, d_fv_gpu[0], dim_cpu.space_mem));
			writeGold(dim_cpu, output_gold, &fv_cpu_GOLD);
		}

		//================= PERF
		// iterate for each neighbor of a box (number_nn)
		double flop = number_nn;
		// The last for iterate NUMBER_PAR_PER_BOX times
		flop *= NUMBER_PAR_PER_BOX;
		// the last for uses 46 operations plus 2 exp() functions
		flop *= 46;
		flop *= nstreams;
		double flops = (double) flop / kernel_time;
		double outputpersec = (double) dim_cpu.space_elem * 4 * nstreams
				/ kernel_time;
		if (verbose)
			printf("BOXES:%d BLOCK:%d OUTPUT/S:%.2f FLOPS:%.2f (GFLOPS:%.2f)\n",
					dim_cpu.boxes1d_arg, NUMBER_THREADS, outputpersec, flops,
					flops / 1000000000);
		if (verbose)
			printf("Kernel time:%f\n", kernel_time);
		//=====================

		printf(".");
		fflush (stdout);

		double iteration_time = mysecond() - globaltimer;
		if (verbose)
			printf("Iteration time: %.4fs (%3.1f%% Device)\n", iteration_time,
					(kernel_time / iteration_time) * 100.0);
		if (verbose)
			printf("===================================\n");

		fflush(stdout);
	}
	
	gpu_memory_unset(nstreams, gpu_check, d_box_gpu, d_rv_gpu, d_qv_gpu,
			d_fv_gpu, d_fv_gold_gpu);
			
	// SYSTEM MEMORY DEALLOCATION
	if (!generate && fv_cpu_GOLD)
		free(fv_cpu_GOLD);

	//if (fv_cpu) free(fv_cpu);
	for (int streamIdx = 0; streamIdx < nstreams; streamIdx++) {
		free(fv_cpu[streamIdx]);
	}

	if (rv_cpu)
		free(rv_cpu);
	if (qv_cpu)
		free(qv_cpu);
	if (box_cpu)
		free(box_cpu);
	printf("\n");

	return 0;
}