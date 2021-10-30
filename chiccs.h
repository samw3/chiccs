// Copyright (c) 2021, Sam Washburn.
// SPDX-License-Identifier: MIT
// https://github.com/samw3/chiccs
// Based on smol-compute by Aras Pranckevicius

#ifndef CHICCS_H
#define CHICCS_H

#include <stddef.h>

typedef struct ChiccsBuffer_s ChiccsBuffer;
typedef struct ChiccsKernel_s ChiccsKernel;

typedef enum {
  ChiccsBackend_Metal = 0,
  ChiccsBackend_D3D11,
  ChiccsBackend_Vulkan
} ChiccsBackend;

// D3D11 cares about the data buffer type
typedef enum {
  ChiccsBufferType_Constant = 0,
  ChiccsBufferType_Structured,
} ChiccsBufferType;

// D3D11 cares how we use buffers
typedef enum {
  ChiccsBufferBinding_Constant = 0,
  ChiccsBufferBinding_Input,
  ChiccsBufferBinding_Output
} ChiccsBufferBinding;

bool ChiccsCreateCompute();
void ChiccsDeleteCompute();
ChiccsBackend ChiccsGetBackend();

ChiccsBuffer *ChiccsCreateBuffer(size_t size, ChiccsBufferType type);
ChiccsBuffer *ChiccsCreateStructuredBuffer(size_t size, ChiccsBufferType type, size_t structuredElementSize);
void ChiccsDeleteBuffer(ChiccsBuffer *buffer);
void ChiccsGetBuffer(ChiccsBuffer *buffer, void *dest, size_t size, size_t srcOffset);
void ChiccsSetBuffer(ChiccsBuffer *buffer, void *src, size_t size, size_t destOffset);

ChiccsKernel *ChiccsCreateKernel(void *shaderCode, size_t shaderSize, char *entryPoint);
void ChiccsDeleteKernel(ChiccsKernel *kernel);
void ChiccsSetKernel(ChiccsKernel *kernel);
void ChiccsSetKernelBuffer(ChiccsBuffer *buffer, int index, ChiccsBufferBinding binding);
void ChiccsDispatchKernel(int xThreads, int yThreads, int zThreads, int xGroupSize, int yGroupSize, int zGroupSize);


#endif //CHICCS_H INCLUDED

#ifdef CHICCS_IMPLEMENTATION

#include <assert.h>

#ifndef CHICCS_METAL
#ifndef CHICCS_D3D11
#error Choose a backend: CHICCS_METAL, CHICCS_D3D11
#endif
#endif

//////////////////////
/// Metal
#ifdef CHICCS_METAL

#include <TargetConditionals.h>
#import <Metal/Metal.h>

struct ChiccsBuffer_s {
  id <MTLBuffer> buffer;
  size_t size;
  bool writtenByGpuSinceLastRead;
};

static id <MTLDevice> sMetalDevice;
static id <MTLCommandQueue> sMetalCommandQueue;
static id <MTLCommandBuffer> sMetalCommandBuffer;
static id <MTLComputeCommandEncoder> sMetalComputeEncoder;

static void MetalReportError(NSString *message, NSError *error) {
  NSString *description = [error localizedDescription];
  description = (description == nil) ? @"<unknown>" : description;
  NSString *reason = [error localizedFailureReason];
  reason = (reason == nil) ? @"<unknown>" : reason;
  NSLog(@"%@  Description: %@  Reason: %@", message, description, reason);
}

static void MetalFlushActiveEncoders() {
  if (sMetalComputeEncoder != nil) {
    [sMetalComputeEncoder endEncoding];
    sMetalComputeEncoder = nil;
  }
}

static void MetalFinishWork() {
  if (sMetalCommandBuffer == nil)
    return;
  MetalFlushActiveEncoders();
  [sMetalCommandBuffer commit];
  [sMetalCommandBuffer waitUntilCompleted];
  sMetalCommandBuffer = nil;
}

static void MetalBufferMakeGpuDataVisibleToCpu(ChiccsBuffer *buffer) {
  assert(sMetalCommandBuffer != nil);
  MetalFlushActiveEncoders();
  id <MTLBlitCommandEncoder> blit = [sMetalCommandBuffer blitCommandEncoder];
  [blit synchronizeResource:buffer->buffer];
  [blit endEncoding];
}

static void MetalStartCommandBufferIfNeeded() {
  if (sMetalCommandBuffer == nil)
    sMetalCommandBuffer = [sMetalCommandQueue commandBufferWithUnretainedReferences];
}

bool ChiccsCreateCompute() {
  sMetalDevice = MTLCreateSystemDefaultDevice();
  sMetalCommandQueue = [sMetalDevice newCommandQueue];
  return true;
}

void ChiccsDeleteCompute() {
  MetalFinishWork();
  sMetalCommandQueue = nil;
  sMetalDevice = nil;
}

ChiccsBackend ChiccsGetBackend() {
  return ChiccsBackend_Metal;
}

ChiccsBuffer *ChiccsCreateBuffer(size_t size, ChiccsBufferType type) {
  ChiccsBuffer *buf = malloc(sizeof(ChiccsBuffer));
  buf->buffer = [sMetalDevice newBufferWithLength:size options:MTLResourceStorageModeManaged];
  buf->size = size;
  buf->writtenByGpuSinceLastRead = FALSE;
  return buf;
}

ChiccsBuffer *ChiccsCreateStructuredBuffer(size_t size, ChiccsBufferType type, size_t structuredElementSize) {
  return ChiccsCreateBuffer(size, type);
}

void ChiccsDeleteBuffer(ChiccsBuffer *buffer) {
  if (buffer == NULL) return;
  assert(buffer->buffer != nil);
  buffer->buffer = nil;
  free(buffer);
}

void ChiccsGetBuffer(ChiccsBuffer *buffer, void *dest, size_t size, size_t srcOffset) {
  assert(buffer);
  assert(srcOffset + size <= buffer->size);
  if (buffer->writtenByGpuSinceLastRead) {
    MetalBufferMakeGpuDataVisibleToCpu(buffer);
    MetalFinishWork();
    buffer->writtenByGpuSinceLastRead = false;
  }
}

void ChiccsSetBuffer(ChiccsBuffer *buffer, void *src, size_t size, size_t destOffset) {
  assert(buffer);
  assert(destOffset + size <= buffer->size);
  uint8_t *dest = (uint8_t *) [buffer->buffer contents];
  memcpy(dest + destOffset, src, size);
  [buffer->buffer didModifyRange:NSMakeRange(destOffset, size)];
}

struct ChiccsKernel_s {
  id <MTLComputePipelineState> kernel;
};

void MetalReportError(NSString *string, NSError *error);
ChiccsKernel *ChiccsCreateKernel(void *shaderCode, size_t shaderSize, char *functionName) {
  NSString *shaderCodeString = [[NSString alloc] initWithBytes:shaderCode length:shaderSize encoding:NSASCIIStringEncoding];

  MTLCompileOptions *options = [MTLCompileOptions new];
  // TODO: options.fastMathEnabled?

  NSError *error = nil;
  id <MTLLibrary> library = [sMetalDevice newLibraryWithSource:shaderCodeString options:options error:&error];
  if (error != nil) MetalReportError(@"ChiccsCreateKernel failed!", error);
  if (library == nil) return NULL;

  id <MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:functionName]];
  if (function == nil) return NULL;

  MTLComputePipelineDescriptor *descriptor = [[MTLComputePipelineDescriptor alloc] init];
  descriptor.computeFunction = function;
  id <MTLComputePipelineState> pipeline = [sMetalDevice newComputePipelineStateWithDescriptor:descriptor
                                                                                      options:MTLPipelineOptionNone
                                                                                   reflection:nil
                                                                                        error:&error];

  if (error != nil) MetalReportError(@"ChiccsCreateKernel error creating kernel pipeline state", error);
  if (pipeline == nil) return NULL;

  ChiccsKernel *kernel = malloc(sizeof(ChiccsKernel));
  kernel->kernel = pipeline;
  return kernel;
}

void ChiccsDeleteKernel(ChiccsKernel *kernel) {
  if (kernel == NULL) return;
  kernel->kernel = nil;
  free(kernel);
}

void ChiccsSetKernel(ChiccsKernel *kernel) {
  MetalStartCommandBufferIfNeeded();
  if (sMetalComputeEncoder == nil) {
    MetalFlushActiveEncoders();
    sMetalComputeEncoder = [sMetalCommandBuffer computeCommandEncoder];
  }
  [sMetalComputeEncoder setComputePipelineState:kernel->kernel];
}

void ChiccsSetKernelBuffer(ChiccsBuffer *buffer, int index, ChiccsBufferBinding binding) {
  assert(sMetalComputeEncoder != nil);
  if (binding == ChiccsBufferBinding_Output) {
    buffer->writtenByGpuSinceLastRead = TRUE;
  }
  [sMetalComputeEncoder setBuffer:buffer->buffer offset:0 atIndex:index];
}

void ChiccsDispatchKernel(int xThreads, int yThreads, int zThreads, int xGroupSize, int yGroupSize, int zGroupSize) {
  assert(sMetalComputeEncoder != nil);
  int xGroups = (xGroupSize - 1 + xThreads) / xGroupSize;
  int yGroups = (yGroupSize - 1 + yThreads) / yGroupSize;
  int zGroups = (zGroupSize - 1 + zThreads) / zGroupSize;
  [sMetalComputeEncoder dispatchThreadgroups:MTLSizeMake(xGroups, yGroups, zGroups)
                       threadsPerThreadgroup:MTLSizeMake(xGroupSize, yGroupSize, zGroupSize)];
}

#endif // CHICCS_METAL

//////////////////////
/// Direct3D 11

#ifdef CHICCS_D3D11
#endif // CHICCS_D3D11


#endif // CHICCS_IMPLEMENTATION