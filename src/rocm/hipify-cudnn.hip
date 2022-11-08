
#include "hipify-cudnn.h"
#include "common/logging.h"

#define HIP_CHECK(expr) do {                                                                      \
  hipError_t rc = (expr);                                                                         \
  ABORT_IF(rc != hipSuccess,                                                                      \
        "HIP error {} '{}' - {}:{}: {}", rc, hipGetErrorString(rc),  __FILE__, __LINE__, #expr);  \
} while(0)

#define MI_CHECK(expr) do {                                                                      	   \
  miopenStatus_t rc = (expr);                                                                          \
  ABORT_IF(rc != miopenStatusSuccess,                                                                  \
        "MIOpen error {} '{}' - {}:{}: {}", rc, miopenGetErrorString(rc),  __FILE__, __LINE__, #expr); \
} while(0)

namespace {
void *SaveAsPriorBuffer(void *dData) {
    void *dPrior = NULL;    // Pointer to keep track of priorDst value
    size_t dPriorSize = 0;  // PriorDstSize
    HIP_CHECK(hipMemPtrGetInfo(
        dData, &dPriorSize));  // Get the info of the gradient dx size
    HIP_CHECK(hipMalloc(&dPrior, dPriorSize));  // Allocate priorDst
    HIP_CHECK(hipMemcpy(
        dPrior, dData, dPriorSize,
        hipMemcpyDeviceToDevice));  // Copy gradient to prior Destination
    return dPrior;
}

// Revoke the PriorBuffer
void deallocPrior(void *dData) {
    size_t dPriorSize = 0;  // PriorDstSize
    HIP_CHECK(hipMemPtrGetInfo(dData, &dPriorSize));
    if (dPriorSize > 0) HIP_CHECK(hipFree(dData));
}

template <typename T>
__global__ void TensorAdd(T *C_d, T *A_d, T beta, int N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;
    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = beta * A_d[i] + C_d[i];
    }
}
} // anonymous namespace

//
// Valid only for beta!=0
//
miopenStatus_t cudnnConvolutionBackwardData(
    miopenHandle_t handle, const void *alpha,
    const miopenTensorDescriptor_t wDesc, const void *w,
    const miopenTensorDescriptor_t dyDesc, const void *dy,
    const miopenConvolutionDescriptor_t convDesc,
	miopenConvBwdDataAlgorithm_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const miopenTensorDescriptor_t dxDesc, void *dx) {

	ABORT_IF( !((*static_cast<const float *>(beta)) != 0),
	        "Hipified cuDNN error 'only beta!=0 supported' - {}:{}",  __FILE__, __LINE__);

	int gradientArray[5], gradientStride[5];
	miopenDataType_t dataType;
	MI_CHECK(miopenGetTensorDescriptor(dxDesc, &dataType, gradientArray, gradientStride));

    size_t expectedWorkSpaceSize = 0;
    void *workSpaceInternal = NULL;

	const float tempBeta = 0;
	void *dxPrior = SaveAsPriorBuffer(dx);
	MI_CHECK(miopenConvolutionBackwardData(
	    handle, alpha, dyDesc,
	    dy, wDesc, w,
	    convDesc, algo, &tempBeta,
	    dxDesc, dx, workSpaceInternal,
	    expectedWorkSpaceSize));

	//accumulateGradients(dx, dxPrior, dxDesc, beta, &dataType)

    int totalElements = gradientArray[0] * gradientArray[1] * gradientArray[2] *
                        gradientArray[3];
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

   if(dataType == miopenFloat) {

    float betaVal = *(static_cast<const float *>(beta));
    float *gradientF = static_cast<float *>(dx);
    float *gradientPriorF = static_cast<float *>(dxPrior);
    TensorAdd<<<blocks, threadsPerBlock>>>(gradientF, gradientPriorF, betaVal, totalElements);
       HIP_CHECK(hipDeviceSynchronize());
    }
    else if (dataType == miopenHalf){

    __half betaVal = *(static_cast<const __half *>(beta));
    __half *gradientF = static_cast<__half *>(dx);
    __half *gradientPriorF = static_cast<__half *>(dxPrior);
    TensorAdd<<<blocks, threadsPerBlock>>>(gradientF, gradientPriorF, betaVal, totalElements);
    HIP_CHECK(hipDeviceSynchronize());
    }
	deallocPrior(dxPrior);
	return miopenStatusSuccess;
}

miopenStatus_t cudnnConvolutionBackwardFilter(
		miopenHandle_t handle, const void *alpha,
    const miopenTensorDescriptor_t xDesc, const void *x,
    const miopenTensorDescriptor_t dyDesc, const void *dy,
    const miopenConvolutionDescriptor_t convDesc,
	miopenConvBwdWeightsAlgorithm_t algo, void *workSpace,
	size_t workSpaceSizeInBytes, const void *beta,
    const miopenTensorDescriptor_t dwDesc, void *dw) {

	int gradientArray[5], gradientStride[5];
	miopenDataType_t dataType;
	MI_CHECK(miopenGetTensorDescriptor(dwDesc, &dataType, gradientArray, gradientStride));

    size_t expectedWorkSpaceSize = 0;
    void *workSpaceInternal = NULL;

	const float tempBeta = 0;
	void *dwPrior = SaveAsPriorBuffer(dw);
	MI_CHECK(miopenConvolutionBackwardWeights(
            handle, alpha, dyDesc, dy,
            xDesc, x,
            convDesc, algo, &tempBeta,
            dwDesc, dw, workSpaceInternal,
            expectedWorkSpaceSize));

	//accumulateGradients(dx, dxPrior, dxDesc, beta, &dataType)

    int totalElements = gradientArray[0] * gradientArray[1] * gradientArray[2] *
                        gradientArray[3];
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

   if(dataType == miopenFloat) {

    float betaVal = *(static_cast<const float *>(beta));
    float *gradientF = static_cast<float *>(dw);
    float *gradientPriorF = static_cast<float *>(dwPrior);
    TensorAdd<<<blocks, threadsPerBlock>>>(gradientF, gradientPriorF, betaVal, totalElements);
       HIP_CHECK(hipDeviceSynchronize());
    }
    else if (dataType == miopenHalf){

    __half betaVal = *(static_cast<const __half *>(beta));
    __half *gradientF = static_cast<__half *>(dw);
    __half *gradientPriorF = static_cast<__half *>(dwPrior);
    TensorAdd<<<blocks, threadsPerBlock>>>(gradientF, gradientPriorF, betaVal, totalElements);
    HIP_CHECK(hipDeviceSynchronize());
    }
	deallocPrior(dwPrior);
	return miopenStatusSuccess;
}

//
// Valid only for alpha=1 and beta=0.
//
miopenStatus_t cudnnPoolingForward(
	    miopenHandle_t handle, const miopenPoolingDescriptor_t poolingDesc,
	    const void *alpha, const miopenTensorDescriptor_t xDesc, const void *x,
	    const void *beta, const miopenTensorDescriptor_t yDesc, void *y) {

	ABORT_IF( !((*static_cast<const float *>(alpha)) == 1 && (*static_cast<const float *>(beta)) == 0),
	        "Hipified cuDNN error 'only alpha=1 and beta=0 supported' - {}:{}",  __FILE__, __LINE__);

    void *devptr = 0;
    size_t workSpaceSize = 0;

    MI_CHECK(miopenPoolingGetWorkSpaceSize(yDesc,&workSpaceSize));
    HIP_CHECK(hipMalloc(&devptr, workSpaceSize));

    // We are destroying the workspace so no point in saving data there for backward pass.
    MI_CHECK(miopenPoolingForward(handle,
                                  poolingDesc,
                                  alpha,xDesc, x,
                                  beta,yDesc, y,
                                  /*do_backward=*/false,
                                  devptr, workSpaceSize));
    HIP_CHECK(hipFree(devptr));
    return miopenStatusSuccess;
}

//
// Valid only for alpha=1 and beta=1
//
miopenStatus_t cudnnPoolingBackward(
	miopenHandle_t handle, const miopenPoolingDescriptor_t poolingDesc,
    const void *alpha, const miopenTensorDescriptor_t yDesc, const void *y,
    const miopenTensorDescriptor_t dyDesc, const void *dy,
    const miopenTensorDescriptor_t xDesc, const void *x, const void *beta,
    const miopenTensorDescriptor_t dxDesc, void *dx) {

	ABORT_IF( !((*static_cast<const float *>(alpha)) == 1 && (*static_cast<const float *>(beta)) == 1),
	        "Hipified cuDNN error 'only alpha=1 and beta=1 supported' - {}:{}",  __FILE__, __LINE__);

    void *devptr = 0;
    size_t workSpaceSize = 0;

    MI_CHECK(miopenPoolingGetWorkSpaceSize(yDesc,&workSpaceSize));
    HIP_CHECK(hipMalloc(&devptr, workSpaceSize));

    void *dwPrior = SaveAsPriorBuffer(dx);
    const float alpha1 = 1;
    const float beta1 = 0;

    MI_CHECK(miopenPoolingBackward(
        handle, poolingDesc, &alpha1,
        yDesc, y, dyDesc,
        dy, xDesc, x, &beta1,
        dxDesc, dx,
        devptr));

    int alpha2 =0;
    MI_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, alpha,
                            dxDesc, dx, beta,
                            dxDesc, dwPrior, &alpha2,
                            dxDesc, dx));
    deallocPrior(dwPrior);

    HIP_CHECK(hipFree(devptr));
    return miopenStatusSuccess;
}

