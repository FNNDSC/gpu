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

////////////////////////////////////////////////////////////////////////////////
// Simple reduction kernel.
// Refer to the 'reduction' CUDA SDK sample describing
// reduction optimization strategies
////////////////////////////////////////////////////////////////////////////////
__kernel void reduce(__global float *d_Result, __global float *d_Input, int N){
    const int tid = get_global_id(0);
    const int threadN = get_global_size(0);

    float sum = 0;

    for(int pos = tid; pos < N; pos += threadN)
        sum += d_Input[pos];

    d_Result[tid] = sum;
}



///
/// This is part 1 of the Kernel from Algorithm 4 in the paper
///
__kernel  void OCL_SSSP_KERNEL1(__global int *vertexArray, __global int *edgeArray, __global float *weightArray,
                               __global unsigned char *maskArray, __global float *costArray, __global float *updatingCostArray,
                               int vertexCount, int edgeCount )
{
    // access thread id
    unsigned int tid = get_global_id(0);

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
__kernel  void OCL_SSSP_KERNEL2(__global int *vertexArray, __global int *edgeArray, __global float *weightArray,
                                __global unsigned char *maskArray, __global float *costArray, __global float *updatingCostArray,
                                __global int endVertex )
{
    // access thread id
    unsigned int tid = get_global_id(0);

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

///
/// Kernel to initialize buffers
///
__kernel void initializeBuffers( __global unsigned char *maskArray, __global float *costArray, __global float *updatingCostArray,
                                 int sourceVertex )
{
    // access thread id
    unsigned int tid = get_global_id(0);

    if (sourceVertex == tid)
    {
        maskArray[tid] = 1;
        costArray[tid] = 0.0;
        updatingCostArray[tid] = 0.0;
    }
    else
    {
        maskArray[tid] = 0;
        costArray[tid] = FLT_MAX;
        updatingCostArray[tid] = FLT_MAX;
    }
}
