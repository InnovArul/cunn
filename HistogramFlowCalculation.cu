extern "C"
{
#include <lualib.h>
#include <lauxlib.h>
#include <lua.h>

}

#include <iostream>
#include "utils.h"
#include "common.h"
#include "stdio.h"
#include "transportSimplex.cuh"

#define ELEMENT(array, i, j, k)  array[((i) * ((inputHeight) * (inputWidth))) + ((j) * (inputWidth)) + (k)]
#define LOGICAL_OUTPUT_COLUMNS  ((inputWidth/stride) + 1)
#define LOGICAL_OUTPUT_ROWS  ((inputHeight/stride) + 1)
#define TOTAL_PATCH_CELLS (patchwidth * patchwidth)

using namespace dt_simplex;
using namespace std;

// Define this to turn on error checking
#define CUDA_ERROR_CHECK
//#define DEBUG_LEVEL 1
#define MALLOC_LIMIT  2047*1024*1024

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

bool HistogramFlowCalculation_IsMallocSet = false;

/**
 * API to call Cuda APIs safely
 * @param err
 * @param file
 * @param line
 */
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

/**
 * API to check the last returned cuda error
 * @param file
 * @param line
 */
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

/**
 * function to return the ground distance
 */
__device__
float HistogramFlowCalculation_Distance(int *F1, int *F2, float** groundDistance) {
    //return 1;
	return groundDistance[*F1][*F2];
}

/**
 * API written for CUDA functionalities
 * It parallelizes the computation of EMD for the input map
 */
#define AMAZINGFAST
//#define OKFAST

__global__
void _HistogramFlowCalculation_calcFlow(const float* input, int layerIndex, int inputLayers,
		int inputWidth, float* output1, float* output2,
		float* groundDistance1, float* groundDistance2, int numberOfBins) {

	// get the current layer and element indices
	int flowIndex;
	// get the correct row and column indices
	int halfLayersCount = inputLayers / 2;
	bool isFtoG = true;

#ifdef AMAZINGFAST

	int currentLayerIndex = threadIdx.x;
	if(currentLayerIndex >= halfLayersCount)
	{
		isFtoG = false;
		currentLayerIndex = currentLayerIndex - halfLayersCount;
	}
    
#else
	int currentLayerIndex = layerIndex;
	//note down the current row
	int currentRow = blockIdx.x * STRIDE;

	// note down the current column
	int currentColumn = threadIdx.x * STRIDE;
#endif

#ifdef DEBUG_LEVEL
	printf("block: %d,  thread: %d\n", blockIdx.x, threadIdx.x);
#endif

#ifdef DEBUG_LEVEL
	printf("totallayers: %d, colcount: %d", inputLayers, inputWidth);
#endif

	// feature holding buffers
	float *elementsInA = new float[numberOfBins], 
            *elementsInB = new float[numberOfBins];
	int *features1 = new int[numberOfBins], 
         *features2 = new int[numberOfBins];

	// flow holding buffers
	TsFlow *flowElements1 = new TsFlow[numberOfBins * numberOfBins + numberOfBins], 
           *flowElements2 = new TsFlow[numberOfBins * numberOfBins + numberOfBins];
           
	int flowLength1 = 0, flowLength2 = 0;
        
	// collect histogram elements
	for (int index = 0; index < numberOfBins; index++) {
        // prepare feature indicators
		features1[index] = index;
		features2[index] = index;

        elementsInA[index] = input[currentLayerIndex * numberOfBins + index];
		elementsInB[index] = input[(currentLayerIndex + halfLayersCount) * numberOfBins + index];
	}


	TsSignature<int> srcSig(numberOfBins, features1, elementsInA);
	TsSignature<int> snkSig(numberOfBins, features2, elementsInB);

#ifdef DEBUG_LEVEL
	printf( "features:");
	for (int i = 0; i < numberOfBins; i++)
	{
		printf( "features1: %d --> %f, features2: %d --> %f \n ", features1[i], elementsInA[i], features2[i], elementsInB[i]);
	}
#endif

	int currParamIndexStart = (currentLayerIndex) * numberOfBins * numberOfBins;

	// PARALLELIZING THE F to G & G to F EMD Calculation
	// TODO: SET THE GroundDistance before calculating EMD flow
	if(isFtoG) {
		float** currGroundDistance1 = new float*[numberOfBins];

		for (int i = 0; i < numberOfBins; i++) {
            currGroundDistance1[i] = new float[numberOfBins];
			for (int j = 0; j < numberOfBins; j++) {

				currGroundDistance1[i][j] = groundDistance1[currParamIndexStart	+ (i * numberOfBins) + j];
				//printf("gnd1 %d to %d : %f \n", i, j, currGroundDistance1[i][j]);
			}
		}
        
        //printf("After ground distance collection\n");

		EMDSolver emd;
		//set the ground distance
		emd.setGroundDistance(currGroundDistance1, numberOfBins);
        
        //printf("After set ground distance\n");
        
		float emdDistance1 = emd.transportSimplex<int>(&srcSig, &snkSig, HistogramFlowCalculation_Distance, numberOfBins, flowElements1, &flowLength1);

        //printf("EMD is calculated\n");
        
#ifdef DEBUG_LEVEL
        printf( "Total cost: %f\n", emdDistance1);
        printf( "Flows: %d \n", flowLength1);
        for (int i = 0; i < flowLength1; i++)
            printf( "%d to %d : %f ", flowElements1[i].from, flowElements1[i].to, flowElements1[i].amount);
#endif

	    int currentPatchOutputIndexStart = ((currentLayerIndex) * numberOfBins * numberOfBins);

		// fill the flow of f_i --> g_i
		for(flowIndex = 0; flowIndex < flowLength1; flowIndex++)
		{
			// get the current flow
			TsFlow currFlow = flowElements1[flowIndex];
			int outputIndex = currentPatchOutputIndexStart + (currFlow.from * numberOfBins) + currFlow.to;
		 	output1[outputIndex] = currFlow.amount;

		}
        
        //printf("Flow is filled\n");
        
		for (int i = 0; i < numberOfBins; i++) 
            delete[] currGroundDistance1[i];
        
        delete[] currGroundDistance1;
	}
	else
	{
		float** currGroundDistance2 = new float*[numberOfBins];

		for (int i = 0; i < numberOfBins; i++) {
            currGroundDistance2[i] = new float[numberOfBins];
            
			for (int j = 0; j < numberOfBins; j++) {

				currGroundDistance2[i][j] = groundDistance2[currParamIndexStart	+ (i * numberOfBins) + j];
				//printf("gnd1 %d to %d : %f \n", i, j, currGroundDistance1[i][j]);
			}
		}
        
		// TODO: SET THE GroundDistance before calculating EMD flow
		EMDSolver emd;
		//set the ground distance
		emd.setGroundDistance(currGroundDistance2, numberOfBins);
		float emdDistance2 = emd.transportSimplex(&snkSig, &srcSig, HistogramFlowCalculation_Distance, numberOfBins, flowElements2, &flowLength2);

#ifdef DEBUG_LEVEL
        printf( "Total cost: %f\n", emdDistance2);
        printf( "Flows: %d\n", flowLength2);
        for (int i = 0; i < flowLength2; i++)
        printf( "%d to %d : %f ", flowElements2[i].from, flowElements2[i].to, flowElements2[i].amount);
#endif

		int currentPatchOutputIndexStart = ((currentLayerIndex) * numberOfBins * numberOfBins);
		//printf("gtof patch start : %d", currentPatchOutputIndexStart);

		// fill the flow of g_i --> f_i
		 for(flowIndex = 0; flowIndex < flowLength2; flowIndex++)
		 {
			 // get the current flow
			 TsFlow currFlow = flowElements2[flowIndex];
			 int outputIndex = currentPatchOutputIndexStart + (currFlow.from * numberOfBins) + currFlow.to;
			 //printf("gtof index %d to %d : %f", currFlow.from, currFlow.to, currFlow.amount);
			 output2[outputIndex] = currFlow.amount;
		 }
         
		for (int i = 0; i < numberOfBins; i++)
            delete[] currGroundDistance2[i];
            
         delete[] currGroundDistance2;
	}
    
    delete[] elementsInA; delete[] elementsInB;
    delete[] features1; delete[] features2;
    delete[] flowElements1; delete[] flowElements2;
}

/**
 * API to report the memory usage of the GPU
 */
static void reportMemStatus() {

	// show memory usage of GPU
	size_t free_byte;
	size_t total_byte;
	size_t malloc_byte;

	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

	if (cudaSuccess != cuda_status) {
		printf("Error: cudaMemGetInfo fails, %s \n",
				cudaGetErrorString(cuda_status));
		return;
	}

	cuda_status = cudaDeviceGetLimit(&malloc_byte, cudaLimitMallocHeapSize);
	if (cudaSuccess != cuda_status) {
			printf("Error: cudaDeviceGetLimit fails, %s \n",
					cudaGetErrorString(cuda_status));
			return;
	}

	double free_db = (double) free_byte;
	double total_db = (double) total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB, malloc limit = %f MB\n",
			used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
			total_db / 1024.0 / 1024.0, malloc_byte / 1024.0 / 1024.0);

}

/**
 * API to set the malloc limit of GPU
 */
static void setMallocLimit() {

	// cudaDeviceSetLimit can be called only once to set the malloc limit
	// this if loop is to prevent multiple calls of cudaDeviceSetLimit
	if(!HistogramFlowCalculation_IsMallocSet)
	{
		cudaError_t	cuda_status = cudaDeviceSetLimit(cudaLimitMallocHeapSize, MALLOC_LIMIT);
		if (cudaSuccess != cuda_status) {
				printf("Error: cudaDeviceSetLimit fails, %s \n",
						cudaGetErrorString(cuda_status));
				return;
		}

		HistogramFlowCalculation_IsMallocSet = true;
	}
}

/**
 * Update output function of the module
 * @param L
 * @return
 */
static int cunn_HistogramFlowCalculation_updateOutput(lua_State *L) {
	THCState *state = getCutorchState(L);

	// Input
	THCudaTensor *input = (THCudaTensor*) luaT_checkudata(L, 2,
			"torch.CudaTensor");

	//fields
	THCudaTensor *output1 = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "output1", "torch.CudaTensor");
	THCudaTensor *output2 = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "output2", "torch.CudaTensor");
	THCudaTensor *groundDistance1 = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "groundDistance1", "torch.CudaTensor");
	THCudaTensor *groundDistance2 = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "groundDistance2", "torch.CudaTensor");
    int numberOfBins = luaT_getfieldchecknumber(L, 1, "numberOfBins");

	THAssert(THCudaTensor_checkGPU(state, 3, input, output1, output2));

	int inputLayers = input->size[0];
	int inputWidth = input->size[1];

	float* inputContents = THCudaTensor_data(state, input);

#ifdef DEBUG_LEVEL
	printf("\n inside cuda function *********** ");
#endif

	//report the current memory status
	//reportMemStatus();

	//set malloc limit to a higher limit
	setMallocLimit();

	//resize output
	//for each layer, 25 neighbors (5 x 5) of each pixel
	THCudaTensor_resize3d(state, output1, (inputLayers / 2), numberOfBins, numberOfBins);
	THCudaTensor_resize3d(state, output2, (inputLayers / 2), numberOfBins, numberOfBins);
	THCudaTensor_fill(state, output1, 0);
	THCudaTensor_fill(state, output2, 0);

#ifdef DEBUG_LEVEL
	printf("%d x %d \n", inputLayers, inputWidth);
	printf("%d x %d x %d \n", output1->size[0], output1->size[1], output1->size[2]);
#endif

	float* outputPtr1 = THCudaTensor_data(state, output1);
	float* outputPtr2 = THCudaTensor_data(state, output2);
	float* groundDistancePtr1 = THCudaTensor_data(state, groundDistance1);
	float* groundDistancePtr2 = THCudaTensor_data(state, groundDistance2);

	//report the current memory status
	//reportMemStatus();

#ifdef AMAZINGFAST
	//here layerIndex is a dummy variable
	int layerIndex = 0;
	_HistogramFlowCalculation_calcFlow<<<1, inputLayers>>>(inputContents, layerIndex, inputLayers, inputWidth, outputPtr1, outputPtr2, groundDistancePtr1, groundDistancePtr2, numberOfBins);
	CudaCheckError();

#else
	for (int layerIndex = 0; layerIndex < inputLayers / 2; layerIndex++) {
		_HistogramFlowCalculation_calcFlow<<<LOGICAL_OUTPUT_ROWS, LOGICAL_OUTPUT_COLUMNS>>>(inputContents, layerIndex, inputLayers, inputHeight, inputWidth, outputPtr1, outputPtr2, groundDistancePtr1, groundDistancePtr2, PATCHWIDTH, STRIDE);
		CudaCheckError();

#ifdef DEBUG_LEVEL
	reportMemStatus();
#endif
	}
#endif

#ifdef DEBUG_LEVEL
printf("\n ****** completed \n ");
#endif

return 1;
}

__global__ void _HistogramFlowCalculation_calcFlowGradInput(const float* input, int layersCount,
	int rowsCount, int columnsCount, float* gradOutput1, float* gradOutput2,
	float* gradInput) {

}

static int cunn_HistogramFlowCalculation_updateGradInput(lua_State *L) {
    return 1;
}

static const struct luaL_Reg cunn_HistogramFlowCalculation__[] = { {
	"HistogramFlowCalculation_updateOutput", cunn_HistogramFlowCalculation_updateOutput }, {
	"HistogramFlowCalculation_updateGradInput", cunn_HistogramFlowCalculation_updateGradInput }, {
	NULL, NULL } };

void cunn_HistogramFlowCalculation_init(lua_State *L) {
luaT_pushmetatable(L, "torch.CudaTensor");
luaT_registeratname(L, cunn_HistogramFlowCalculation__, "nn");
lua_pop(L, 1);
}
