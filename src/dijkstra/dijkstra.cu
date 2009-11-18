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
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <dijkstra_kernel.cu>

///
//  Some test data
//      http://en.literateprograms.org/Dijkstra%27s_algorithm_%28Scala%29
char *vNames[] =
{
        "Barcelona",
        "Narbonne",
        "Marseille",
        "Toulouse",
        "Geneve",
        "Paris",
        "Lausanne"
};


int vertexArray[] =
{
        0, // "Barcelona(0)",
        1, // "Narbonne(1)",
        5, // "Marseille(2)",
        7, // "Toulouse(3)",
        10, // "Geneve(4)",
        15, // "Paris(5)",
        18  // "Lausanne(6)"
};

int edgeArray[] =
{
        1, // 0  <arc from="Barcelona(0)" to="Narbonne(1)" cost="250" />

        2, // 1  <arc from="Narbonne(1)" to="Marseille(2)" cost="260" />
        3, // 2  <arc from="Narbonne(1)" to="Toulouse(3)" cost="150" />
        4, // 3  <arc from="Narbonne(1)" to="Geneve(4)" cost="550" />
        0, // 4  <arc from "Narbonne(1)" to="Barcelona(0)" cost="250" />

        4, // 5  <arc from="Marseille(2)" to="Geneve(4)" cost="470" />
        1, // 6  <arc from="Marseille(2)" to="Narbonne(1)" cost="260" />

        5, // 7  <arc from="Toulouse(3)" to="Paris(5)" cost="680" />
        4, // 8  <arc from="Toulouse(3)" to="Geneve(4)" cost="700" />
        1, // 9  <arc from="Toulouse(3)" to="Narbonne(1)" cost="150" />

        5, // 10 <arc from="Geneve(4)" to="Paris(5)" cost="540" />
        6, // 11 <arc from="Geneve(4)" to="Lausanne(6)" cost="64" />
        1, // 12 <arc from="Geneve(4)" to="Narbonne(1)" cost="550" />
        2, // 13  <arc from="Geneve(4)" to="Marseille(2)" cost="470" />
        3, // 14  <arc from="Geneve(4)" to="Toulouse(3)" cost="700" />


        6, // 15 <arc from="Paris(5)" to="Lausanne(6)" cost="536" />
        4, // 16 <arc from="Paris(5)" to="Geneve(4)" cost="540" />
        3, // 17 <arc from="Paris(5)" to="Toulouse(3)" cost="680" />

        5, // 18 <arc from="Lausanne(6)" to="Paris(5)" cost="536" />
        4  // 19 <arc from="Lausanne(6)" to="Geneve(4)" cost="64" />

};

float weightArray[] =
{
        250, // 0  <arc from="Barcelona(0)" to="Narbonne(1)" cost="250" />

        260, // 1  <arc from="Narbonne(1)" to="Marseille(2)" cost="260" />
        150, // 2  <arc from="Narbonne(1)" to="Toulouse(3)" cost="150" />
        550, // 3  <arc from="Narbonne(1)" to="Geneve(4)" cost="550" />
        250, // 4  <arc from "Narbonne(1)" to="Barcelona(0)" cost="250" />

        470, // 5  <arc from="Marseille(2)" to="Geneve(4)" cost="470" />
        260, // 6  <arc from="Marseille(2)" to="Narbonne(1)" cost="260" />

        680, // 7  <arc from="Toulouse(3)" to="Paris(5)" cost="680" />
        700, // 8  <arc from="Toulouse(3)" to="Geneve(4)" cost="700" />
        150, // 9  <arc from="Toulouse(3)" to="Narbonne(1)" cost="150" />

        540, // 10 <arc from="Geneve(4)" to="Paris(5)" cost="540" />
        64,  // 11 <arc from="Geneve(4)" to="Lausanne(6)" cost="64" />
        550, // 12 <arc from="Geneve(4)" to="Narbonne(1)" cost="550" />
        470, // 13  <arc from="Geneve(4)" to="Marseille(2)" cost="470" />
        700, // 14  <arc from="Geneve(4)" to="Toulouse(3)" cost="700" />


        536, // 15 <arc from="Paris(5)" to="Lausanne(6)" cost="536" />
        540, // 16 <arc from="Paris(5)" to="Geneve(4)" cost="540" />
        680, // 17 <arc from="Paris(5)" to="Toulouse(3)" cost="680" />

        536, // 18 <arc from="Lausanne(6)" to="Paris(5)" cost="536" />
        64   // 19 <arc from="Lausanne(6)" to="Geneve(4)" cost="64" />
};


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    // Allocate memory for arrays
    GraphData graph;
    graph.vertexCount = sizeof(vertexArray) / sizeof(int);
    graph.edgeCount = sizeof(edgeArray) / sizeof(int);
    printf("Vertex Count: %d\n", graph.vertexCount);
    printf("Edge Count: %d\n", graph.edgeCount);

    // V
    cutilSafeCall( cudaMalloc( (void**) &graph.vertexArray, sizeof(int) * graph.vertexCount) );
    cutilSafeCall( cudaMemcpy( graph.vertexArray, vertexArray, sizeof(int) * graph.vertexCount, cudaMemcpyHostToDevice) );

    // E
    cutilSafeCall( cudaMalloc( (void**) &graph.edgeArray, sizeof(int) * graph.edgeCount) );
    cutilSafeCall( cudaMemcpy( graph.edgeArray, edgeArray, sizeof(int) * graph.edgeCount, cudaMemcpyHostToDevice) );

    // W
    cutilSafeCall( cudaMalloc( (void**) &graph.weightArray, sizeof(float) * graph.edgeCount) );
    cutilSafeCall( cudaMemcpy( graph.weightArray, weightArray, sizeof(float) * graph.edgeCount, cudaMemcpyHostToDevice) );

    // M, C, U
    cutilSafeCall( cudaMalloc( (void**) &graph.maskArray, sizeof(unsigned char) * graph.vertexCount) );
    cutilSafeCall( cudaMalloc( (void**) &graph.costArray, sizeof(float) * graph.vertexCount) );
    cutilSafeCall( cudaMalloc( (void**) &graph.updatingCostArray, sizeof(float) * graph.vertexCount) );


    float *resultCosts = (float*) malloc(sizeof(float) * graph.vertexCount);

    for(int source = 0; source < graph.vertexCount; source++)
    {
        runDijkstra(&graph, source);

        cutilSafeCall( cudaMemcpy( resultCosts, graph.costArray, sizeof(float) * graph.vertexCount, cudaMemcpyDeviceToHost) );

        printf("--\n");
        printf("From %s:\n", vNames[source]);
        for (int i = 0; i < graph.vertexCount; i++ )
        {
            if (i != source)
            {
                printf(" --> %s: %f\n", vNames[i], resultCosts[i]);
            }
        }

    }

    cutilSafeCall(cudaFree(graph.vertexArray));
    cutilSafeCall(cudaFree(graph.edgeArray));
    cutilSafeCall(cudaFree(graph.weightArray));
    cutilSafeCall(cudaFree(graph.maskArray));
    cutilSafeCall(cudaFree(graph.costArray));
    cutilSafeCall(cudaFree(graph.updatingCostArray));

    cudaThreadExit();
}
