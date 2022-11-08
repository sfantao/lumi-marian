#pragma once

#include <hip/hip_runtime.h>
#include "hip/hip_fp16.h"
#include <miopen/miopen.h>

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
    const miopenTensorDescriptor_t dxDesc, void *dx);

miopenStatus_t cudnnConvolutionBackwardFilter(
		miopenHandle_t handle, const void *alpha,
    const miopenTensorDescriptor_t xDesc, const void *x,
    const miopenTensorDescriptor_t dyDesc, const void *dy,
    const miopenConvolutionDescriptor_t convDesc,
	miopenConvBwdWeightsAlgorithm_t algo, void *workSpace,
	size_t workSpaceSizeInBytes, const void *beta,
    const miopenTensorDescriptor_t dwDesc, void *dw);

//
// Valid only for alpha=1 and beta=0.
//
miopenStatus_t cudnnPoolingForward(
		miopenHandle_t handle, const miopenPoolingDescriptor_t poolingDesc,
	    const void *alpha, const miopenTensorDescriptor_t xDesc, const void *x,
	    const void *beta, const miopenTensorDescriptor_t yDesc, void *y);

//
// Valid only for alpha=1 and beta=1
//
miopenStatus_t cudnnPoolingBackward(
		miopenHandle_t handle, const miopenPoolingDescriptor_t poolingDesc,
    const void *alpha, const miopenTensorDescriptor_t yDesc, const void *y,
    const miopenTensorDescriptor_t dyDesc, const void *dy,
    const miopenTensorDescriptor_t xDesc, const void *x, const void *beta,
    const miopenTensorDescriptor_t dxDesc, void *dx);

#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 				miopenConvolutionBwdDataAlgoGEMM
#define CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 			miopenConvolutionBwdWeightsAlgoGEMM
#define CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM 		miopenConvolutionFwdAlgoGEMM
#define CUDNN_CROSS_CORRELATION 						miopenConvolution
#define CUDNN_DATA_FLOAT 								miopenFloat
#define CUDNN_NOT_PROPAGATE_NAN 						0
#define CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING 	miopenPoolingAverageInclusive
#define CUDNN_POOLING_MAX 								miopenPoolingMax
#define CUDNN_STATUS_SUCCESS 							miopenStatusSuccess

#define cudnnAddTensor(a,b,c,d,e,f,g) 					miopenOpTensor(a,miopenTensorOpAdd,b,f,g,b,c,d,e,f,g)
#define cudnnConvolutionBackwardBias 					miopenConvolutionBackwardBias
#define cudnnConvolutionDescriptor_t              		miopenConvolutionDescriptor_t
#define cudnnConvolutionForward(a,b,c,d,e,f,g,h,j,k,l,m,n) \
														miopenConvolutionForward(a,b,c,d,e,f,g,h,l,m,n,j,k)  // This mapping is only valid for alpha=1 and beta=0
#define cudnnCreate 									miopenCreate
#define cudnnCreateConvolutionDescriptor 				miopenCreateConvolutionDescriptor
#define cudnnCreateFilterDescriptor 					miopenCreateTensorDescriptor
#define cudnnCreatePoolingDescriptor					miopenCreatePoolingDescriptor
#define cudnnCreateTensorDescriptor 					miopenCreateTensorDescriptor
#define cudnnDestroy 									miopenDestroy
#define cudnnDestroyConvolutionDescriptor 				miopenDestroyConvolutionDescriptor
#define cudnnDestroyFilterDescriptor 					miopenDestroyTensorDescriptor
#define cudnnDestroyPoolingDescriptor 					miopenDestroyPoolingDescriptor
#define cudnnDestroyTensorDescriptor 					miopenDestroyTensorDescriptor
#define cudnnFilterDescriptor_t                   		miopenTensorDescriptor_t //TODO: Assess use of tensor descriptor - hipDNN does that.
#define cudnnGetConvolution2dForwardOutputDim 			miopenGetConvolutionForwardOutputDim
#define cudnnGetErrorString 							miopenGetErrorString
#define cudnnGetPooling2dForwardOutputDim 				miopenGetPoolingForwardOutputDim
#define cudnnHandle_t                             		miopenHandle_t
#define cudnnPoolingDescriptor_t				  		miopenPoolingDescriptor_t
#define cudnnPoolingMode_t						 		miopenPoolingMode_t
#define cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode) \
														miopenInitConvolutionDescriptor(convDesc, mode, pad_h, pad_w, u, v, upscalex, upscaley)
#define cudnnSetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride) \
														miopenSet2dPoolingDescriptor(poolingDesc,mode,windowHeight,windowWidth,horizontalPadding,verticalPadding,horizontalStride,verticalStride)
#define cudnnSetFilter4dDescriptor(a,b,c,d,e,f,g) 		miopenSet4dTensorDescriptor(a,b,d,e,f,g) // Only NCHW is supported.
#define cudnnSetTensor4dDescriptor(a,b,c,d,e,f,g) 		miopenSet4dTensorDescriptor(a,c,d,e,f,g) // Only NCHW is supported.
#define cudnnTensorDescriptor_t                   		miopenTensorDescriptor_t
