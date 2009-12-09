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

/*
 * This application demonstrates how to use multiple GPUs with OpenCL.
 * Event Objects are used to synchronize the CPU with multiple GPUs. 
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the 
 * application. On the other side, you can still extend your desktop to screens 
 * attached to both GPUs.
 */

#include <oclUtils.h>
#include "oclDijkstraKernel.h"

///
//  Macro Options
//
//#define DO_CPU
//#define DO_GPU
//#define DO_MULTI_GPU
#define DO_MULTI_GPU_CPU

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
// Helper functions
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    // start logs 
    shrSetLogFileName ("oclDijkstra.txt");

    cl_context gpuContext;
    cl_context cpuContext;
    cl_int errNum;

    // start timer & logs
    shrLog(LOGBOTH, 0.0, "Setting up OpenCL on the Host...\n\n");
    shrDeltaT(1);

    // create the OpenCL context on available GPU devices
    gpuContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    shrLog(LOGBOTH, 0.0, "clCreateContextFromType\n\n");

    // Create an OpenCL context on available CPU devices
    cpuContext = clCreateContextFromType(0, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    shrLog(LOGBOTH, 0.0, "clCreateContextFromType");

    // Allocate memory for arrays
    GraphData graph;
#ifdef CITY_DATA
    graph.vertexCount = sizeof(vertexArray) / sizeof(int);
    graph.edgeCount = sizeof(edgeArray) / sizeof(int);
    graph.vertexArray = &vertexArray[0];
    graph.edgeArray = &edgeArray[0];
    graph.weightArray = &weightArray[0];
#else
    generateRandomGraph(&graph, 98000, 100);
#endif


    printf("Vertex Count: %d\n", graph.vertexCount);
    printf("Edge Count: %d\n", graph.edgeCount);

    std::vector<int> sourceVertices;


    for(int source = 0; source < 200; source++)
    {
        sourceVertices.push_back(source);
    }

    int *sourceVertArray = (int*) malloc(sizeof(int) * sourceVertices.size());
    std::copy(sourceVertices.begin(), sourceVertices.end(), sourceVertArray);

    float *results = (float*) malloc(sizeof(float) * sourceVertices.size() * graph.vertexCount);


    // Run Dijkstra's algorithm
    shrDeltaT(0);
#ifdef DO_CPU
    double startTimeCPU = shrDeltaT(0);
    runDijkstra(cpuContext, oclGetMaxFlopsDev(cpuContext), &graph, sourceVertArray,
                results, sourceVertices.size() );
    double endTimeCPU = shrDeltaT(0);
#endif

#ifdef DO_GPU
    double startTimeGPU = shrDeltaT(0);
    runDijkstra(gpuContext, oclGetMaxFlopsDev(gpuContext), &graph, sourceVertArray,
                results, sourceVertices.size() );
    double endTimeGPU = shrDeltaT(0);
#endif

#ifdef DO_MULTI_GPU
    double startTimeMultiGPU = shrDeltaT(0);
    runDijkstraMultiGPU(gpuContext, &graph, sourceVertArray,
                        results, sourceVertices.size() );
    double endTimeMultiGPU = shrDeltaT(0);
#endif

#ifdef DO_MULTI_GPU_CPU
    double startTimeGPUCPU = shrDeltaT(0);
    runDijkstraMultiGPUandCPU(gpuContext, cpuContext, &graph, sourceVertArray,
                              results, sourceVertices.size() );
    double endTimeGPUCPU = shrDeltaT(0);
#endif

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


#ifdef DO_CPU
    shrLog(LOGBOTH, 0.0, "\nrunDijkstra - CPU Time:               %f s\n", endTimeCPU - startTimeCPU);
#endif

#ifdef DO_GPU
    shrLog(LOGBOTH, 0.0, "\nrunDijkstra - Single GPU Time:        %f s\n", endTimeGPU - startTimeGPU);
#endif

#ifdef DO_MULTI_GPU
    shrLog(LOGBOTH, 0.0, "\nrunDijkstra - Multi GPU Time:         %f s\n", endTimeMultiGPU - startTimeMultiGPU);
#endif

#ifdef DO_MULTI_GPU_CPU
    shrLog(LOGBOTH, 0.0, "\nrunDijkstra - Multi GPU and CPU Time: %f s\n", endTimeGPUCPU - startTimeGPUCPU);
#endif

    free(sourceVertArray);
    free(results);

    clReleaseContext(gpuContext);

    // finish
    shrEXIT(argc, argv);
 }
