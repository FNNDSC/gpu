/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/* Template project which demonstrates the basics on how to setup a project 
 * example application.
 * Device code.
 */

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>

#define INFINITI    (9999999) // Should really fix how infiniti is set for the float buffers.  Will do this in the
                              // real version.

///
//  Types
//
//
//  This data structure and algorithm implementation is based on
//  Accelerating larger graph algorithms on the GPU using CUDA by
//  Parwan Harish and P.J. Narayanan
//
typedef struct
{
    // (V) This contains a pointer to the edge list for each vertex
    int *vertexArray;

    // Vertex count
    int vertexCount;

    // (E) This contains pointers to the vertices that each edge is attached to
    int *edgeArray;

    // Edge count
    int edgeCount;

    // (W) Weight array
    float *weightArray;

    // (M) Mask array
    unsigned char *maskArray;

    // (C) Cost array
    float *costArray;

    // (U) Updating cost array
    float *updatingCostArray;
} GraphData;

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
/// This is part 2 of the Kernel from Algorithm 5 in the paper
///
__global__  void CUDA_SSSP_KERNEL2(  int *vertexArray, int *edgeArray, float *weightArray,
                                     unsigned char *maskArray, float *costArray, float *updatingCostArray )
{
    // access thread id
    unsigned int tid = threadIdx.x;

    if (costArray[tid] > updatingCostArray[tid])
    {
        costArray[tid] = updatingCostArray[tid];
        maskArray[tid] = 1;
    }

    updatingCostArray[tid] = costArray[tid];
}

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
/// Run Dijkstra's shortest path on the GraphData provided to this function.  This function
/// assumes that the caller has allocated GPU memory for each of the arrays and has placed
/// vertex, edge, and weight data in each of the arrays.  This will determine the shortest
/// path from the sourceVertex to any other vertices.
///
/// This is a prototype that will be developed further when I optimize mris_pmake.
///
void runDijkstra( GraphData* graph, int sourceVertex )
{
    // Initialize mask array to false, C and U to infiniti
    cudaMemset( graph->maskArray, 0, sizeof(unsigned char) * graph->vertexCount );
    cudaMemset( graph->costArray, INFINITI, sizeof(float) * graph->vertexCount ); // This needs to be replaced, not correct for float
    cudaMemset( graph->updatingCostArray, INFINITI, sizeof(float) * graph->vertexCount ); // This needs to be replaced, not correct for float

    // Set M[S] = true, C[S] = 0, U[S] = 0
    cudaMemset( &graph->maskArray[sourceVertex], 1, sizeof(unsigned char) );
    cudaMemset( &graph->costArray[sourceVertex], 0, sizeof(float) );
    cudaMemset( &graph->updatingCostArray[sourceVertex], 0, sizeof(float) );

    dim3  grid( 1, 1, 1);
    dim3  threads( graph->vertexCount, 1, 1);

    unsigned char *maskArrayHost = (unsigned char*) malloc(sizeof(unsigned char) * graph->vertexCount);
    cudaMemcpy( maskArrayHost, graph->maskArray, sizeof(unsigned char) * graph->vertexCount, cudaMemcpyDeviceToHost );

    while(!maskArrayEmpty(maskArrayHost, graph->vertexCount))
    {
        // execute the kernel
        CUDA_SSSP_KERNEL1<<< grid, threads >>>( graph->vertexArray, graph->edgeArray, graph->weightArray,
                                                graph->maskArray, graph->costArray, graph->updatingCostArray,
                                                graph->vertexCount, graph->edgeCount );
        CUT_CHECK_ERROR("CUDA_SSSP_KERNEL1");

        CUDA_SSSP_KERNEL2<<< grid, threads >>>( graph->vertexArray, graph->edgeArray, graph->weightArray,
                                                graph->maskArray, graph->costArray, graph->updatingCostArray );
        CUT_CHECK_ERROR("CUDA_SSSP_KERNEL2");

        cudaMemcpy( maskArrayHost, graph->maskArray, sizeof(unsigned char) * graph->vertexCount, cudaMemcpyDeviceToHost );
    }

    free (maskArrayHost);
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
