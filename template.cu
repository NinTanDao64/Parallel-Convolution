#include <wb.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"

#define MASK_WIDTH 5
#define Mask_radius (MASK_WIDTH/2)
#define O_TILE_WIDTH 16
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE 
//implement the tiled 2D convolution kernel with adjustments for channels
//use shared memory to reduce the number of global accesses, handle the boundary conditions in when loading input list elements into the shared memory
__global__ void convolution2D(float* I, const float* __restrict__ M, float* P,
	int channels, int width, int height) {

	__shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH];

	int outputRow = blockIdx.y * O_TILE_WIDTH + threadIdx.y;
	int outputCol = blockIdx.x * O_TILE_WIDTH + threadIdx.x;

	int inputRow = outputRow - (MASK_WIDTH / 2);
	int inputCol = outputCol - (MASK_WIDTH / 2);

	for (int i = 0; i < channels; i++) {

		if ((inputRow >= 0) && (inputRow < height) && (inputCol >= 0) && (inputCol < width)) {
			N_ds[threadIdx.y][threadIdx.x] = I[(inputRow * width + inputCol) * channels + i];
		}
		else {
			N_ds[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		float output = 0.0f;
		if (threadIdx.x < O_TILE_WIDTH && threadIdx.y < O_TILE_WIDTH) {
			for (int k = 0; k < MASK_WIDTH; k++) {
				for (int j = 0; j < MASK_WIDTH; j++) {
					output += M[k * MASK_WIDTH + j] * N_ds[threadIdx.y + k][threadIdx.x + j];
				}
			}
		}

		if (threadIdx.x < O_TILE_WIDTH && threadIdx.y < O_TILE_WIDTH) {
			if (outputRow < height && outputCol < width) {
				P[(outputRow * width + outputCol) * channels + i] = clamp(output);
				assert(P[(outputRow * width + outputCol) * channels + i] >= 0 && P[(outputRow * width + outputCol) * channels + i] <= 1);
			}
		}

		__syncthreads();

	}
}


int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
  assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  //allocate device memory
  cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  //copy host memory to device
  cudaMemcpy(deviceInputImageData,
	  hostInputImageData,
	  imageWidth * imageHeight * imageChannels * sizeof(float),
	  cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData,
	  hostMaskData,
	  maskRows * maskColumns * sizeof(float),
	  cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  //initialize thread block and kernel grid dimensions
  //invoke CUDA kernel	

  dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
  wbLog(TRACE, "blockDim is ", BLOCK_WIDTH, ",", BLOCK_WIDTH);

  wbLog(TRACE, "Image is ", imageWidth, " x ", imageHeight);

  int gridWidth = (imageWidth - 1) / O_TILE_WIDTH + 1; // Ceiling
  int gridHeight = (imageHeight - 1) / O_TILE_WIDTH + 1; // Ceiling
  dim3 gridDim(gridWidth, gridHeight);
  wbLog(TRACE, "gridDim is ", gridWidth, ",", gridHeight);

  convolution2D << <gridDim, blockDim >> >(deviceInputImageData, deviceMaskData,
	  deviceOutputImageData, imageChannels,
	  imageWidth, imageHeight);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  //copy results from device to host
  cudaMemcpy(hostOutputImageData,
	  deviceOutputImageData,
	  imageWidth * imageHeight * imageChannels * sizeof(float),
	  cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ INSERT CODE HERE
  //deallocate device memory
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
