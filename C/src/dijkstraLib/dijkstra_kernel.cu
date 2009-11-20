//
//
//  Description:
//      Implementation of Dijkstra's Single-Source Shortest Path (SSSP) algorithm on the GPU.
//      The basis of this implementation is the paper:
//
//          "Accelerating large graph algorithms on the GPU using CUDA" by
//          Parwan Harish and P.J. Narayanan
//
//
//  Author:
//      Dan Ginsburg
//
//  Children's Hospital Boston
//  GPL v2
//
// includes, project
#include <cutil_inline.h>
#include <stdio.h>
#include <float.h>
#include <multithreading.h>


#include "dijkstra_kernel.h"

///
//  Types
//

// This structure is used in the multi-GPU implementation of the algorithm.
// This structure defines the workload for each GPU.  The code chunks up
// the work on a per-GPU basis.
typedef struct
{
    // GPU number to run algorithm on
    int device;

    // Pointer to graph data
    GraphData *graph;

    // Source vertex indices to process
    int *sourceVertices;

    // End vertex indices to process
    int *endVertices;

    // Results of processing
    float *outResultCosts;

    // Number of results
    int numResults;

} GPUPlan;

///////////////////////////////////////////////////////////////////////////////
//
//  CUDA Compute Kernels
//
//

///
/// This is part 1 of the Kernel from Algorithm 4 in the paper
///
__global__  void CUDA_SSSP_KERNEL1( int *vertexArray, int *edgeArray, float *weightArray,
                                    unsigned char *maskArray, float *costArray, float *updatingCostArray,
                                    int vertexCount, int edgeCount )
{
    // access thread id
    unsigned int tid = threadIdx.x;

    if ( maskArray[tid] != 0 )
    {
        maskArray[tid] = 0;

        int edgeStart = vertexArray[tid];
        int edgeEnd;
        if (tid + 1 < (vertexCount))
        {
            edgeEnd = vertexArray[tid + 1];
        }
        else
        {
            edgeEnd = edgeCount;
        }

        for(int edge = edgeStart; edge < edgeEnd; edge++)
        {
            int nid = edgeArray[edge];

            // One note here: whereas the paper specified weightArray[nid], I
            //  found that the correct thing to do was weightArray[edge].  I think
            //  this was a typo in the paper.  Either that, or I misunderstood
            //  the data structure.
            if (updatingCostArray[nid] > (costArray[tid] + weightArray[edge]))
            {
                updatingCostArray[nid] = (costArray[tid] + weightArray[edge]);
            }
        }
    }
}

///
/// This is part 2 of the Kernel from Algorithm 5 in the paper.  The only modification
/// is to stop the search after hitting endVertex
///
__global__  void CUDA_SSSP_KERNEL2(  int *vertexArray, int *edgeArray, float *weightArray,
                                     unsigned char *maskArray, float *costArray, float *updatingCostArray,
                                     int endVertex )
{
    // access thread id
    unsigned int tid = threadIdx.x;

    if (costArray[tid] > updatingCostArray[tid])
    {
        costArray[tid] = updatingCostArray[tid];

        // Stop if we have hit the final vertex
        if (tid != endVertex)
        {
            maskArray[tid] = 1;
        }
    }

    updatingCostArray[tid] = costArray[tid];
}

///////////////////////////////////////////////////////////////////////////////
//
//  Private Functions
//
//

///
/// Check whether the mask array is empty.  This tells the algorithm whether
/// it needs to continue running or not.
///
bool maskArrayEmpty(unsigned char *maskArray, int count)
{
    for(int i = 0; i < count; i++ )
    {
        if (maskArray[i] == 1)
        {
            return false;
        }
    }

    return true;
}

///
///  Allocate memory for input CUDA buffers and copy the data into device memory
///
void allocateCUDABuffers(GraphData *graph,
                         int **vertexArrayDevice, int **edgeArrayDevice, float **weightArrayDevice,
                         unsigned char **maskArrayDevice, float **costArrayDevice, float **updatingCostArrayDevice,
                         float **infinitiArrayDevice)
{
    // V
    cutilSafeCall( cudaMalloc( (void**) vertexArrayDevice, sizeof(int) * graph->vertexCount) );
    cutilSafeCall( cudaMemcpy( *vertexArrayDevice, graph->vertexArray, sizeof(int) * graph->vertexCount, cudaMemcpyHostToDevice) );

    // E
    cutilSafeCall( cudaMalloc( (void**) edgeArrayDevice, sizeof(int) * graph->edgeCount) );
    cutilSafeCall( cudaMemcpy( *edgeArrayDevice, graph->edgeArray, sizeof(int) * graph->edgeCount, cudaMemcpyHostToDevice) );

    // W
    cutilSafeCall( cudaMalloc( (void**) weightArrayDevice, sizeof(float) * graph->edgeCount) );
    cutilSafeCall( cudaMemcpy( *weightArrayDevice, graph->weightArray, sizeof(float) * graph->edgeCount, cudaMemcpyHostToDevice) );

    // M, C, U
    cutilSafeCall( cudaMalloc( (void**) maskArrayDevice, sizeof(unsigned char) * graph->vertexCount) );
    cutilSafeCall( cudaMalloc( (void**) costArrayDevice, sizeof(float) * graph->vertexCount) );
    cutilSafeCall( cudaMalloc( (void**) updatingCostArrayDevice, sizeof(float) * graph->vertexCount) );

    // This is quite annoying, but at the moment I can't find a way to set a float
    // value to a buffer in CUDA (cudaMemSet operates on bytes).  So I create an
    // infiniti array that is used to clear the buffers
    float *infinityArray = (float*) malloc(sizeof(float) * graph->vertexCount);
    for(int i = 0; i < graph->vertexCount; i++)
    {
        infinityArray[i] = FLT_MAX;
    }

    cutilSafeCall( cudaMalloc( (void**) infinitiArrayDevice, sizeof(float) * graph->vertexCount) );
    cutilSafeCall( cudaMemcpy( *infinitiArrayDevice,infinityArray, sizeof(float) * graph->vertexCount, cudaMemcpyHostToDevice) );

    free (infinityArray);
}

///
/// Initialize CUDA buffers for single run of Dijkstra
///
void initializeCUDABuffers(GraphData *graph, int sourceVertex,
                           unsigned char *maskArrayDevice, float *costArrayDevice, float *updatingCostArrayDevice,
                           float *infinityArrayDevice)
{
    cudaMemset( maskArrayDevice, 0, sizeof(unsigned char) * graph->vertexCount );

    // FUTURE OPTIMIZATION: Figure out how to do this with a memset, or at least something not requiring a
    //                      full memcpy.
    cudaMemcpy( costArrayDevice, infinityArrayDevice, sizeof(float) * graph->vertexCount, cudaMemcpyDeviceToDevice );
    cudaMemcpy( updatingCostArrayDevice, infinityArrayDevice, sizeof(float) * graph->vertexCount, cudaMemcpyDeviceToDevice );

    // Set M[S] = true, C[S] = 0, U[S] = 0
    cudaMemset( &maskArrayDevice[sourceVertex], 1, sizeof(unsigned char) );
    cudaMemset( &costArrayDevice[sourceVertex], 0, sizeof(float) );
    cudaMemset( &updatingCostArrayDevice[sourceVertex], 0, sizeof(float) );
}

///
/// Worker thread for running the algorithm on one of the GPUs
///
CUT_THREADPROC dijkstraThread(GPUPlan *plan)
{
    // Set GPU device
    cutilSafeCall( cudaSetDevice(plan->device) );

    runDijkstra( plan->graph, plan->sourceVertices, plan->endVertices,
                 plan->outResultCosts, plan->numResults );

}


///////////////////////////////////////////////////////////////////////////////
//
//  Public Functions
//
//

///
/// Run Dijkstra's shortest path on the GraphData provided to this function.  This
/// function will compute the shortest path distance from sourceVertices[n] ->
/// endVertices[n] and store the cost in outResultCosts[n].  The number of results
/// it will compute is given by numResults.
///
/// This function will run the algorithm on a single GPU.
///
/// \param graph Structure containing the vertex, edge, and weight arra
///              for the input graph
/// \param startVertices Indices into the vertex array from which to
///                      start the search
/// \param endVertices Indices into the vertex array from which to end
///                    the search.
/// \param outResultsCosts A pre-allocated array where the results for
///                        each shortest path search will be written
/// \param numResults Should be the size of all three passed inarrays
///
void runDijkstra( GraphData* graph, int *sourceVertices, int *endVertices,
                   float *outResultCosts, int numResults)
{
    int *vertexArrayDevice;
    int *edgeArrayDevice;
    float *weightArrayDevice;
    unsigned char *maskArrayDevice;
    float *costArrayDevice;
    float *updatingCostArrayDevice;
    float *infinityArrayDevice;



    // Allocate buffers in Device memory
    allocateCUDABuffers( graph, &vertexArrayDevice, &edgeArrayDevice, &weightArrayDevice,
                         &maskArrayDevice, &costArrayDevice, &updatingCostArrayDevice,
                         &infinityArrayDevice );

    unsigned char *maskArrayHost = (unsigned char*) malloc(sizeof(unsigned char) * graph->vertexCount);

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    for ( int i = 0 ; i < numResults; i++ )
    {
        // Initialize mask array to false, C and U to infiniti
        initializeCUDABuffers( graph, sourceVertices[i],
                              maskArrayDevice, costArrayDevice, updatingCostArrayDevice,
                              infinityArrayDevice );


        dim3  grid( 1, 1, 1);
        dim3  threads( graph->vertexCount, 1, 1);


        cudaMemcpy( maskArrayHost, maskArrayDevice, sizeof(unsigned char) * graph->vertexCount, cudaMemcpyDeviceToHost );

        while(!maskArrayEmpty(maskArrayHost, graph->vertexCount))
        {
            // execute the kernel
            CUDA_SSSP_KERNEL1<<< grid, threads >>>( vertexArrayDevice, edgeArrayDevice, weightArrayDevice,
                                                    maskArrayDevice, costArrayDevice, updatingCostArrayDevice,
                                                    graph->vertexCount, graph->edgeCount );
            CUT_CHECK_ERROR("CUDA_SSSP_KERNEL1");

            CUDA_SSSP_KERNEL2<<< grid, threads >>>( vertexArrayDevice, edgeArrayDevice, weightArrayDevice,
                                                    maskArrayDevice, costArrayDevice, updatingCostArrayDevice,
                                                    endVertices[i] );
            CUT_CHECK_ERROR("CUDA_SSSP_KERNEL2");

            cudaMemcpy( maskArrayHost, maskArrayDevice, sizeof(unsigned char) * graph->vertexCount, cudaMemcpyDeviceToHost );
        }


        float result;

        // Copy the result back
        cutilSafeCall( cudaMemcpy( &result, &costArrayDevice[endVertices[i]], sizeof(float), cudaMemcpyDeviceToHost) );
        outResultCosts[i] = result;
    }

    cutilCheckError(cutStopTimer(timer));
    printf("Kernel GPU Processing time: %f (ms) \n", cutGetTimerValue(timer));

    free (maskArrayHost);

    // Free all the buffers
    cutilSafeCall(cudaFree(vertexArrayDevice));
    cutilSafeCall(cudaFree(edgeArrayDevice));
    cutilSafeCall(cudaFree(weightArrayDevice));
    cutilSafeCall(cudaFree(maskArrayDevice));
    cutilSafeCall(cudaFree(costArrayDevice));
    cutilSafeCall(cudaFree(updatingCostArrayDevice));
    cutilSafeCall(cudaFree(infinityArrayDevice));
}



///
/// Run Dijkstra's shortest path on the GraphData provided to this function.  This
/// function will compute the shortest path distance from sourceVertices[n] ->
/// endVertices[n] and store the cost in outResultCosts[n].  The number of results
/// it will compute is given by numResults.
///
/// This function will run the algorithm on as many GPUs as is available.  It will
/// create N threads, one for each GPU, and chunk the workload up to perform
/// (numResults / N) searches per GPU.
///
/// \param graph Structure containing the vertex, edge, and weight arra
///              for the input graph
/// \param startVertices Indices into the vertex array from which to
///                      start the search
/// \param endVertices Indices into the vertex array from which to end
///                    the search.
/// \param outResultsCosts A pre-allocated array where the results for
///                        each shortest path search will be written
/// \param numResults Should be the size of all three passed inarrays
///
///

void runDijkstraMultiGPU( GraphData* graph, int *sourceVertices, int *endVertices,
                          float *outResultCosts, int numResults )
{
    int numGPUs;

    cutilSafeCall( cudaGetDeviceCount(&numGPUs) );
    printf("CUDA-capable device count: %i\n", numGPUs);

    if (numGPUs == 0)
    {
        // ERORR: no GPUs present!
        return;
    }

    GPUPlan *gpuPlans = (GPUPlan*) malloc(sizeof(GPUPlan) * numGPUs);
    CUTThread *threadIDs = (CUTThread*) malloc(sizeof(CUTThread) * numGPUs);

    // Divide the workload out per GPU
    int resultsPerGPU = numResults / numGPUs;

    int offset = 0;

    for (int i = 0; i < numGPUs; i++)
    {
        gpuPlans[i].device = i;
        gpuPlans[i].graph = graph;
        gpuPlans[i].sourceVertices = &sourceVertices[offset];
        gpuPlans[i].endVertices = &endVertices[offset];
        gpuPlans[i].outResultCosts = &outResultCosts[offset];
        gpuPlans[i].numResults = resultsPerGPU;

        offset += resultsPerGPU;
    }

    // Add any remaining work to the last GPU
    if (offset < numResults)
    {
        gpuPlans[numGPUs - 1].numResults += (numResults - offset);
    }

    // Launch all the threads
    for (int i = 0; i < numGPUs; i++)
    {
        threadIDs[i] = cutStartThread((CUT_THREADROUTINE)dijkstraThread, (void*)(gpuPlans + i));
    }

    // Wait for the results from all threads
    cutWaitForThreads(threadIDs, numGPUs);

    free (gpuPlans);
    free (threadIDs);
}

