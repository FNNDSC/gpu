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
#include <vector>
#include <multithreading.h>
#include <sstream>

// includes, project
#include <cutil_inline.h>
#include "shrUtils.h"

// includes, kernels
#include "dijkstra_kernel.h"

///
//  Macro Options
//
//#define CITY_DATA

///
//  Some test data
//      http://en.literateprograms.org/Dijkstra%27s_algorithm_%28Scala%29
const char *vNames[] =
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
void printDeviceInfo( int argc, char** argv);

///
//  Generate a random graph
//
void generateRandomGraph(GraphData *graph, int numVertices, int neighborsPerVertex)
{
    graph->vertexCount = numVertices;
    graph->vertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->edgeCount = numVertices * neighborsPerVertex;
    graph->edgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->weightArray = (float*)malloc(graph->edgeCount * sizeof(float));

    for(int i = 0; i < graph->vertexCount; i++)
    {
        graph->vertexArray[i] = i * neighborsPerVertex;
    }

    for(int i = 0; i < graph->edgeCount; i++)
    {
        graph->edgeArray[i] = (rand() % graph->vertexCount);
        graph->weightArray[i] = (float)(rand() % 1000) / 1000.0f;
    }
}

///
//  Parse command line arguments
//
void parseCommandLineArgs(int argc, const char **argv, bool &doGPU,
                          bool &doMultiGPU, bool &doRef,
                          int *numSources, int *generateVerts, int *generateEdgesPerVert)
{
    doGPU = shrCheckCmdLineFlag(argc, argv, "gpu");
    doMultiGPU = shrCheckCmdLineFlag(argc, argv, "multigpu");
    doRef = shrCheckCmdLineFlag(argc, argv, "ref");
    shrGetCmdLineArgumenti(argc, argv, "sources", numSources);   
    shrGetCmdLineArgumenti(argc, argv, "verts", generateVerts);
    shrGetCmdLineArgumenti(argc, argv, "edges", generateEdgesPerVert);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    printDeviceInfo(argc, argv);
    runTest( argc, argv);

   // cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
//@TEMP - why do I need this for link to work on Linux?
cutWaitForThreads(NULL,0);
//@TEMP
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

    bool doGPU;
    bool doRef;
    bool doMultiGPU;
    int generateVerts;
    int generateEdgesPerVert;
    int numSources;

    parseCommandLineArgs(argc, (const char**)argv, doGPU, doMultiGPU, doRef, &numSources, &generateVerts, &generateEdgesPerVert);

    // Allocate memory for arrays
    GraphData graph;
#ifdef CITY_DATA
    graph.vertexCount = sizeof(vertexArray) / sizeof(int);
    graph.edgeCount = sizeof(edgeArray) / sizeof(int);
    graph.vertexArray = &vertexArray[0];
    graph.edgeArray = &edgeArray[0];
    graph.weightArray = &weightArray[0];
#else
    generateRandomGraph(&graph, generateVerts, generateEdgesPerVert);
#endif

    printf("Vertex Count: %d\n", graph.vertexCount);
    printf("Edge Count: %d\n", graph.edgeCount);

    std::vector<int> sourceVertices;


    for(int source = 0; source < numSources; source++)
    {
        sourceVertices.push_back(source % graph.vertexCount);
    }

    int *sourceVertArray = (int*) malloc(sizeof(int) * sourceVertices.size());
    std::copy(sourceVertices.begin(), sourceVertices.end(), sourceVertArray);

    float *results = (float*) malloc(sizeof(float) * sourceVertices.size() * graph.vertexCount);


    unsigned int gpuTimer = 0;
    cutilCheckError(cutCreateTimer(&gpuTimer));
    cutilCheckError(cutStartTimer(gpuTimer));

    // Run Dijkstra's algorithm
    if ( doGPU )
    {
        runDijkstra(&graph, sourceVertArray, results, sourceVertices.size() );
    }

    cutilCheckError(cutStopTimer(gpuTimer));


    unsigned int multiGPUTimer = 0;
    cutilCheckError(cutCreateTimer(&multiGPUTimer));
    cutilCheckError(cutStartTimer(multiGPUTimer));

    if ( doMultiGPU )
    {
        runDijkstraMultiGPU(&graph, sourceVertArray, results, sourceVertices.size() );
    }

    cutilCheckError(cutStopTimer(multiGPUTimer));

    unsigned int refTimer = 0;
    cutilCheckError(cutCreateTimer(&refTimer));
    cutilCheckError(cutStartTimer(refTimer));

    if ( doRef )
    {
        runDijkstraRef(&graph, sourceVertArray, results, sourceVertices.size() );
    }

    cutilCheckError(cutStopTimer(refTimer));

#ifdef CITY_DATA
    for (unsigned int i = 0; i < sourceVertices.size(); i++)
    {
        for (int j = 0; j < graph.vertexCount; j++)
        {
            if (i != j)
            {
                printf("%s --> %s: %f\n", vNames[sourceVertArray[i]], vNames[j], results[i * graph.vertexCount + j] );
            }
        }
    }
#endif

    std::ostringstream oss;
    oss << "\nCSV: " << graph.vertexCount << " " << generateEdgesPerVert << " " << numSources << " ";
    
    if (doGPU)
    {
        shrLog("\nrunDijkstra - Single GPU Time:        %f s\n", cutGetTimerValue(gpuTimer) / 1000.0f);
        oss << ( cutGetTimerValue(gpuTimer) / 1000.0f ) << " ";
    }

    if (doMultiGPU)
    {
        shrLog("\nrunDijkstra - Multi GPU Time:         %f s\n", cutGetTimerValue(multiGPUTimer) / 1000.0f);
        oss << cutGetTimerValue(multiGPUTimer) / 1000.0f << " ";
    }

    if (doRef)
    {
        shrLog("\nrunDijkstra - Reference (CPU):        %f s\n", cutGetTimerValue(refTimer) / 1000.0f);
        oss << (cutGetTimerValue(refTimer) / 1000.0f) << " ";
    }
    oss << "\n";
    shrLog(oss.str().c_str());

    free(sourceVertArray);
    free(results);

    cudaThreadExit();
}

///
/// Print Device info
///
void printDeviceInfo(int argc, char **argv)
{
    int deviceCount = 0;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
    {
        printf("cudaGetDeviceCount failed! CUDA Driver and Runtime version may be mismatched.\n");
        printf("\nTest FAILED!\n");
        CUT_EXIT(argc, argv);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
        printf("There is no device supporting CUDA\n");

    int dev;
    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0)
        {
            // This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("There is 1 device supporting CUDA\n");
            else
                printf("There are %d devices supporting CUDA\n", deviceCount);
        }
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    #if CUDART_VERSION >= 2020
        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
    #endif

        printf("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
        printf("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);

        printf("  Total amount of global memory:                 %u bytes\n", (unsigned int)deviceProp.totalGlobalMem);
    #if CUDART_VERSION >= 2000
        printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
        printf("  Number of cores:                               %d\n", 8 * deviceProp.multiProcessorCount);
    #endif
        printf("  Total amount of constant memory:               %u bytes\n", (unsigned int)deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %u bytes\n", (unsigned int)deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n", (unsigned int)deviceProp.memPitch);
        printf("  Texture alignment:                             %u bytes\n", (unsigned int)deviceProp.textureAlignment);
        printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    #if CUDART_VERSION >= 2000
        printf("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
    #endif
    #if CUDART_VERSION >= 2020
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
                                                                        "Default (multiple host threads can use this device simultaneously)" :
                                                                        deviceProp.computeMode == cudaComputeModeExclusive ?
                                                                        "Exclusive (only one host thread at a time can use this device)" :
                                                                        deviceProp.computeMode == cudaComputeModeProhibited ?
                                                                        "Prohibited (no host thread can use this device)" :
                                                                        "Unknown");
    #endif
    }

}
