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

bool IsMallocSet = false;

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
float Distance(int *F1, int *F2, float** groundDistance) {
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
void _calcFlow(const float* input, int layerIndex, int inputLayers,
		int inputHeight, int inputWidth, float* output1, float* output2,
		float* groundDistance1, float* groundDistance2, int patchwidth, int stride) {

	// get the current layer and element indices
	int xIndex, yIndex, flowIndex;
	// get the correct row and column indices
	int halfLayersCount = inputLayers / 2;
	bool isFtoG = true;

#ifdef AMAZINGFAST

	int currentLayerIndex = blockIdx.x;
	if(currentLayerIndex >= halfLayersCount)
	{
		isFtoG = false;
		currentLayerIndex = currentLayerIndex - halfLayersCount;
	}

	int currentElementIndex = threadIdx.x;

	//note down the current row
	int logicalOutputRowIndex = (currentElementIndex / LOGICAL_OUTPUT_COLUMNS);
	int currentRow = logicalOutputRowIndex * stride;

	// note down the current column
	int logicalOutputColumnIndex = (currentElementIndex % LOGICAL_OUTPUT_COLUMNS);
	int currentColumn = (currentElementIndex % LOGICAL_OUTPUT_COLUMNS) * stride;
    
   // if(!(currentLayerIndex == 0 && isFtoG && currentRow == 0 && currentColumn == 0)) return;

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
	printf("totallayers: %d, rowcount: %d, colcount: %d, layer: %d, row: %d,  column: %d, logicrow = %d, logiccol = %d\n", inputLayers, inputHeight, inputWidth, layerIndex, currentRow, currentColumn, logicalOutputRowIndex, logicalOutputColumnIndex);
#endif

	// feature holding buffers
	float *elementsInA = new float[TOTAL_PATCH_CELLS], 
            *elementsInB = new float[TOTAL_PATCH_CELLS];
	int *features1 = new int[TOTAL_PATCH_CELLS], 
         *features2 = new int[TOTAL_PATCH_CELLS];

	// flow holding buffers
	TsFlow *flowElements1 = new TsFlow[TOTAL_PATCH_CELLS * TOTAL_PATCH_CELLS], 
           *flowElements2 = new TsFlow[TOTAL_PATCH_CELLS * TOTAL_PATCH_CELLS];
           
	int flowLength1 = 0, flowLength2 = 0;

	// calculate the neighborhood differences
	int elementIndex = 0;

    int halfPatch = patchwidth / 2;
        
	// collect histogram elements
	for (xIndex = -halfPatch; xIndex <= halfPatch; xIndex++) {
		for (yIndex = -halfPatch; yIndex <= halfPatch; yIndex++) {
            
			// A - B calculation
			int currentXIndex = currentRow + xIndex;
			int currentYIndex = currentColumn + yIndex;
			float elementInA = 0, elementInB = 0;

			// prepare feature indicators
			features1[elementIndex] = elementIndex;
			features2[elementIndex] = elementIndex;

			//collect the elements if they belong to valid indices
			//       ---------------------> y
			//      |
			//      | x
			//
			if (currentXIndex >= 0 && currentXIndex < inputHeight
					&& currentYIndex >= 0 && currentYIndex < inputWidth) {
				elementInA = ELEMENT(input, currentLayerIndex, currentXIndex, currentYIndex);
				elementInB = ELEMENT(input, currentLayerIndex + halfLayersCount, currentXIndex, currentYIndex);
				//printf("element(%d, %d, %d) = %f, element(%d, %d, %d) = %f\n", currentLayerIndex, currentXIndex, currentYIndex, elementInA, currentLayerIndex + halfLayersCount, currentXIndex, currentYIndex, elementInB);
				//printf("\n", );
			}

			elementsInA[elementIndex] = elementInA;
			elementsInB[elementIndex] = elementInB;

			//printf("A=%f , B= %f\n", elementInA, elementInB);
			elementIndex++;
		}
	}


	TsSignature<int> srcSig(TOTAL_PATCH_CELLS, features1, elementsInA);
	TsSignature<int> snkSig(TOTAL_PATCH_CELLS, features2, elementsInB);

#ifdef DEBUG_LEVEL
	printf( "features:");
	for (int i = 0; i < patchwidth * patchwidth; i++)
	{
		printf( "features1: %d --> %f, features2: %d --> %f \n ", features1[i], elementsInA[i], features2[i], elementsInB[i]);
	}
#endif

	int currParamIndexStart = (currentLayerIndex) * TOTAL_PATCH_CELLS * TOTAL_PATCH_CELLS;

	// PARALLELIZING THE F to G & G to F EMD Calculation
	// TODO: SET THE GroundDistance before calculating EMD flow
	if(isFtoG) {
		float** currGroundDistance1 = new float*[TOTAL_PATCH_CELLS];

		for (int i = 0; i < TOTAL_PATCH_CELLS; i++) {
            currGroundDistance1[i] = new float[TOTAL_PATCH_CELLS];
			for (int j = 0; j < TOTAL_PATCH_CELLS; j++) {

				currGroundDistance1[i][j] = groundDistance1[currParamIndexStart	+ (i * TOTAL_PATCH_CELLS) + j];
				//printf("gnd1 %d to %d : %f \n", i, j, currGroundDistance1[i][j]);
			}
		}

		EMDSolver emd;
		//set the ground distance
		emd.setGroundDistance(currGroundDistance1, patchwidth);
		float emdDistance1 = emd.transportSimplex<int>(&srcSig, &snkSig, Distance, patchwidth, flowElements1, &flowLength1);

#ifdef DEBUG_LEVEL
        printf( "Total cost: %f\n", emdDistance1);
        printf( "Flows: %d \n", flowLength1);
        for (int i = 0; i < flowLength1; i++)
            printf( "%d to %d : %f ", flowElements1[i].from, flowElements1[i].to, flowElements1[i].amount);
#endif

	    int currentPatchOutputIndexStart = ((currentLayerIndex) * LOGICAL_OUTPUT_ROWS * TOTAL_PATCH_CELLS * LOGICAL_OUTPUT_COLUMNS * TOTAL_PATCH_CELLS) + (logicalOutputRowIndex * TOTAL_PATCH_CELLS *  LOGICAL_OUTPUT_COLUMNS * TOTAL_PATCH_CELLS) + (logicalOutputColumnIndex * TOTAL_PATCH_CELLS);

		// fill the flow of f_i --> g_i
		for(flowIndex = 0; flowIndex < flowLength1; flowIndex++)
		{
			// get the current flow
			TsFlow currFlow = flowElements1[flowIndex];
			int outputIndex = currentPatchOutputIndexStart + (currFlow.from * (LOGICAL_OUTPUT_COLUMNS * TOTAL_PATCH_CELLS)) + currFlow.to;
		 	output1[outputIndex] = currFlow.amount;

		}
        
		for (int i = 0; i < TOTAL_PATCH_CELLS; i++)
            delete[] currGroundDistance1[i];
        
        delete[] currGroundDistance1;
	}
	else
	{
		float** currGroundDistance2 = new float*[TOTAL_PATCH_CELLS];

		for (int i = 0; i < TOTAL_PATCH_CELLS; i++) {
            currGroundDistance2[i] = new float[TOTAL_PATCH_CELLS];
			for (int j = 0; j < TOTAL_PATCH_CELLS; j++) {
				currGroundDistance2[i][j] = groundDistance2[currParamIndexStart + (i * TOTAL_PATCH_CELLS) + j];
				//printf("gnd2 %d to %d : %f \n", i, j, currGroundDistance2[i][j]);
			}
		}

		// TODO: SET THE GroundDistance before calculating EMD flow
		EMDSolver emd;
		//set the ground distance
		emd.setGroundDistance(currGroundDistance2, patchwidth);
		float emdDistance2 = emd.transportSimplex(&snkSig, &srcSig, Distance, patchwidth, flowElements2, &flowLength2);

#ifdef DEBUG_LEVEL
        printf( "Total cost: %f\n", emdDistance2);
        printf( "Flows: %d\n", flowLength2);
        for (int i = 0; i < flowLength2; i++)
        printf( "%d to %d : %f ", flowElements2[i].from, flowElements2[i].to, flowElements2[i].amount);
#endif

		int currentPatchOutputIndexStart = ((currentLayerIndex) * LOGICAL_OUTPUT_ROWS * TOTAL_PATCH_CELLS * LOGICAL_OUTPUT_COLUMNS * TOTAL_PATCH_CELLS) + (logicalOutputRowIndex * TOTAL_PATCH_CELLS * LOGICAL_OUTPUT_COLUMNS * TOTAL_PATCH_CELLS) + (logicalOutputColumnIndex * TOTAL_PATCH_CELLS);
		//printf("gtof patch start : %d", currentPatchOutputIndexStart);

		// fill the flow of g_i --> f_i
		 for(flowIndex = 0; flowIndex < flowLength2; flowIndex++)
		 {
			 // get the current flow
			 TsFlow currFlow = flowElements2[flowIndex];
			 int outputIndex = currentPatchOutputIndexStart + (currFlow.from * (LOGICAL_OUTPUT_COLUMNS * patchwidth * patchwidth)) + currFlow.to;
			 //printf("gtof index %d to %d : %f", currFlow.from, currFlow.to, currFlow.amount);
			 output2[outputIndex] = currFlow.amount;
		 }
         
		for (int i = 0; i < TOTAL_PATCH_CELLS; i++)
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
	if(!IsMallocSet)
	{
		cudaError_t	cuda_status = cudaDeviceSetLimit(cudaLimitMallocHeapSize, MALLOC_LIMIT);
		if (cudaSuccess != cuda_status) {
				printf("Error: cudaDeviceSetLimit fails, %s \n",
						cudaGetErrorString(cuda_status));
				return;
		}

		IsMallocSet = true;
	}
}

/**
 * Update output function of the module
 * @param L
 * @return
 */
static int cunn_FlowCalculation_updateOutput(lua_State *L) {
	THCState *state = getCutorchState(L);

	// Input
	THCudaTensor *input = (THCudaTensor*) luaT_checkudata(L, 2,
			"torch.CudaTensor");

	//fields
	THCudaTensor *output1 = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "output1", "torch.CudaTensor");
	THCudaTensor *output2 = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "output2", "torch.CudaTensor");
	THCudaTensor *groundDistance1 = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "groundDistance1", "torch.CudaTensor");
	THCudaTensor *groundDistance2 = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "groundDistance2", "torch.CudaTensor");
    int stride = luaT_getfieldchecknumber(L, 1, "stride");
    int PATCHWIDTH = luaT_getfieldchecknumber(L, 1, "patchwidth");

	THAssert(THCudaTensor_checkGPU(state, 3, input, output1, output2));

	luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2,
			"3D or 4D (batch mode) tensor is expected");

	int inputLayers = input->size[0];
	int inputHeight = input->size[1];
	int inputWidth = input->size[2];

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
	THCudaTensor_resize3d(state, output1, (inputLayers / 2), LOGICAL_OUTPUT_ROWS * PATCHWIDTH * PATCHWIDTH, LOGICAL_OUTPUT_COLUMNS * PATCHWIDTH * PATCHWIDTH);
	THCudaTensor_resize3d(state, output2, (inputLayers / 2), LOGICAL_OUTPUT_ROWS * PATCHWIDTH * PATCHWIDTH, LOGICAL_OUTPUT_COLUMNS * PATCHWIDTH * PATCHWIDTH);
	THCudaTensor_fill(state, output1, 0);
	THCudaTensor_fill(state, output2, 0);

#ifdef DEBUG_LEVEL
	printf("%d x %d x %d \n", inputLayers, inputHeight, inputWidth);
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
	_calcFlow<<<inputLayers, LOGICAL_OUTPUT_COLUMNS * LOGICAL_OUTPUT_ROWS>>>(inputContents, layerIndex, inputLayers, inputHeight, inputWidth, outputPtr1, outputPtr2, groundDistancePtr1, groundDistancePtr2, PATCHWIDTH, stride);
	CudaCheckError();

#else
	for (int layerIndex = 0; layerIndex < inputLayers / 2; layerIndex++) {
		_calcFlow<<<LOGICAL_OUTPUT_ROWS, LOGICAL_OUTPUT_COLUMNS>>>(inputContents, layerIndex, inputLayers, inputHeight, inputWidth, outputPtr1, outputPtr2, groundDistancePtr1, groundDistancePtr2, PATCHWIDTH, STRIDE);
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

__global__ void _calcFlowGradInput(const float* input, int layersCount,
	int rowsCount, int columnsCount, float* gradOutput1, float* gradOutput2,
	float* gradInput) {

// get the current layer and element indices
int currentLayerIndex = blockIdx.x;
int currentElementIndex = threadIdx.x;

//printf("element(%d) = %f, element(%d) = %f\n", currentElementIndex, input[currentElementIndex], 12 + currentElementIndex, input[12 + currentElementIndex]);

// get the correct row and column indices
int halfLayersCount = layersCount / 2;
int currentRow = currentElementIndex / columnsCount;
int currentColumn = currentElementIndex % columnsCount;

// get the elements that are in focus now
float gradForElementInA = 0;
float gradForElementInB = 0;

// calculate the neighborhood differences
for (int xIndex = -2; xIndex <= 2; xIndex++) {
	for (int yIndex = -2; yIndex <= 2; yIndex++) {
		int outputLayerIndex = (currentLayerIndex * halfLayersCount) + (xIndex + 2) * 5
				+ (yIndex + 2);

		// add positive gradients (independent of currentXIndex, currentYIndex), only dependent on outputLayerIndex
		//gradForElementInA += ELEMENT(gradOutput1, outputLayerIndex, currentRow, currentColumn);
		//gradForElementInB += ELEMENT(gradOutput2, outputLayerIndex, currentRow, currentColumn);

		// subtract gradients from gradOutput2 (or gradOutput1)
		// to find the correct gradOutput element of current layer, refer below

		//0  0  0  0  0
		//0  0  0  0  0
		//0  0  x  0  0
		//10 9  8  7  6
		//5  4  3  2  1

		int currentNegGradXIndex = currentRow - xIndex;
		int currentNegGradYIndex = currentColumn - yIndex;

		if (currentNegGradXIndex >= 0 && currentNegGradXIndex < rowsCount
				&& currentNegGradYIndex >= 0
				&& currentNegGradYIndex < columnsCount) {
			//gradForElementInA -= ELEMENT(gradOutput2, outputLayerIndex, currentNegGradXIndex, currentNegGradYIndex);
			//gradForElementInB -= ELEMENT(gradOutput1, outputLayerIndex, currentNegGradXIndex, currentNegGradYIndex);
		}
	}
}

//ELEMENT(gradInput, currentLayerIndex, currentRow, currentColumn) = gradForElementInA;
//ELEMENT(gradInput, currentLayerIndex + halfLayersCount, currentRow, currentColumn) = gradForElementInB;
}

static int cunn_FlowCalculation_updateGradInput(lua_State *L) {
//printf("inside updateGradInput cunn");

THCState *state = getCutorchState(L);

// Inputs
THCudaTensor *input = (THCudaTensor *) luaT_checkudata(L, 2,
		"torch.CudaTensor");

//fields
THCudaTensor *gradOutput1 = (THCudaTensor*) luaT_getfieldcheckudata(L, 1,
		"gradOutput1", "torch.CudaTensor");
THCudaTensor *gradOutput2 = (THCudaTensor*) luaT_getfieldcheckudata(L, 1,
		"gradOutput2", "torch.CudaTensor");
THCudaTensor *gradInput = (THCudaTensor*) luaT_getfieldcheckudata(L, 1,
		"gradInput", "torch.CudaTensor");

//determine gradInput sizes
int layersCount = input->size[0];
int rowsCount = input->size[1];
int columnsCount = input->size[2];

//resize gradInputs
THCudaTensor_resize3d(state, gradInput, layersCount, rowsCount, columnsCount);

//get elementary datatype pointers
float* inputContents = THCudaTensor_data(state, input);
float* gradOutputPtr1 = THCudaTensor_data(state, gradOutput1);
float* gradOutputPtr2 = THCudaTensor_data(state, gradOutput2);
float* gradInputPtr = THCudaTensor_data(state, gradInput);

//calculate gradient of final output with respect to each input element
_calcFlowGradInput<<<layersCount/2, rowsCount * columnsCount>>>(inputContents, layersCount, rowsCount, columnsCount, gradOutputPtr1, gradOutputPtr2, gradInputPtr);

//return gradInput
return 1;
}

static const struct luaL_Reg cunn_FlowCalculation__[] = { {
	"FlowCalculation_updateOutput", cunn_FlowCalculation_updateOutput }, {
	"FlowCalculation_updateGradInput", cunn_FlowCalculation_updateGradInput }, {
	NULL, NULL } };

void cunn_FlowCalculation_init(lua_State *L) {
luaT_pushmetatable(L, "torch.CudaTensor");
luaT_registeratname(L, cunn_FlowCalculation__, "nn");
lua_pop(L, 1);
}
