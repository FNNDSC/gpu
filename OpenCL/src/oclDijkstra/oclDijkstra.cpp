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
#include <pthread.h>
#include <sstream>
#include "oclDijkstraKernel.h"

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

///
//  Parse command line arguments
//
void parseCommandLineArgs(int argc, const char **argv, bool &doCPU, bool &doGPU,
                          bool &doMultiGPU, bool &doCPUGPU, bool &doRef,
                          int *sourceVerts,
                          int *generateVerts, int *generateEdgesPerVert)
{
    doCPU = shrCheckCmdLineFlag(argc, argv, "cpu");
    doGPU = shrCheckCmdLineFlag(argc, argv, "gpu");
    doMultiGPU = shrCheckCmdLineFlag(argc, argv, "multigpu");
    doCPUGPU = shrCheckCmdLineFlag(argc, argv, "cpugpu");
    doRef = shrCheckCmdLineFlag(argc, argv, "ref");
    shrGetCmdLineArgumenti(argc, argv, "sources", sourceVerts);
    shrGetCmdLineArgumenti(argc, argv, "verts", generateVerts);
    shrGetCmdLineArgumenti(argc, argv, "edges", generateEdgesPerVert);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    bool doCPU = false;
    bool doGPU = false;
    bool doMultiGPU = false;
    bool doCPUGPU = false;
    bool doRef = false;
    int numSources = 100;
    int generateVerts = 100000;
    int generateEdgesPerVert = 10;

    parseCommandLineArgs(argc, argv, doCPU, doGPU,
                         doMultiGPU, doCPUGPU, doRef,
                         &numSources, &generateVerts, &generateEdgesPerVert);
    // start logs 
    shrSetLogFileName ("oclDijkstra.txt");

    cl_platform_id platform;
    cl_context gpuContext;
    cl_context cpuContext;
    cl_int errNum;
    cl_uint gpuDeviceCount;
    cl_device_id* gpuDevices;
    cl_uint cpuDeviceCount;
    cl_device_id* cpuDevices;
    

    // start timer & logs
    shrLog("Setting up OpenCL on the Host...\n\n");
    shrDeltaT(1);

    // Get the NVIDIA platform
    errNum = oclGetPlatformID(&platform);
    oclCheckError(errNum, CL_SUCCESS);
    shrLog("clGetPlatformID...\n");

    // Get the devices
    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &gpuDeviceCount);
    oclCheckError(errNum, CL_SUCCESS);
    gpuDevices = (cl_device_id *)malloc(gpuDeviceCount * sizeof(cl_device_id) );
    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, gpuDeviceCount, gpuDevices, NULL);
    oclCheckError(errNum, CL_SUCCESS);
    shrLog("clGetDeviceIDs...\n");


    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &cpuDeviceCount);
    cpuDevices = (cl_device_id *)malloc(cpuDeviceCount * sizeof(cl_device_id) );
    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, cpuDeviceCount, cpuDevices, NULL);
//    oclCheckError(errNum, CL_SUCCESS);
    shrLog("clGetDeviceIDs...\n");




    // create the OpenCL context on available GPU devices
    gpuContext = clCreateContext(0, gpuDeviceCount, gpuDevices, NULL, NULL, &errNum);
    shrLog("clCreateContext\n\n");

    // Create an OpenCL context on available CPU devices
    cpuContext = clCreateContext(0, cpuDeviceCount, cpuDevices, NULL, NULL, &errNum);
    shrLog("clCreateContextFromType");

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


    // Run Dijkstra's algorithm
    shrDeltaT(0);
    double startTimeCPU = shrDeltaT(0);
    if (doCPU)
    {
        runDijkstra(cpuContext, oclGetMaxFlopsDev(cpuContext), &graph, sourceVertArray,
                    results, sourceVertices.size() );
    }
    double endTimeCPU = shrDeltaT(0);

    double startTimeGPU = shrDeltaT(0);
    if (doGPU)
    {
        runDijkstra(gpuContext, oclGetMaxFlopsDev(gpuContext), &graph, sourceVertArray,
                    results, sourceVertices.size() );
    }
    double endTimeGPU = shrDeltaT(0);

    double startTimeMultiGPU = shrDeltaT(0);
    if (doMultiGPU)
    {
        runDijkstraMultiGPU(gpuContext, &graph, sourceVertArray,
                            results, sourceVertices.size() );
    }
    double endTimeMultiGPU = shrDeltaT(0);

    double startTimeGPUCPU = shrDeltaT(0);
    if (doCPUGPU)
    {
        runDijkstraMultiGPUandCPU(gpuContext, cpuContext, &graph, sourceVertArray,
                                  results, sourceVertices.size() );
    }
    double endTimeGPUCPU = shrDeltaT(0);

    double startTimeRef = shrDeltaT(0);
    if (doRef)
    {
        runDijkstraRef( &graph, sourceVertArray,
                        results, sourceVertices.size() );
    }
    double endTimeRef = shrDeltaT(0);

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
    if (doCPU)
    {
        shrLog("\nrunDijkstra - CPU Time:               %f s\n", endTimeCPU - startTimeCPU);
        oss << (endTimeCPU - startTimeCPU) << " ";
    }

    if (doGPU)
    {
        shrLog("\nrunDijkstra - Single GPU Time:        %f s\n", endTimeGPU - startTimeGPU);
        oss << (endTimeGPU - startTimeGPU) << " ";
    }

    if (doMultiGPU)
    {
        shrLog("\nrunDijkstra - Multi GPU Time:         %f s\n", endTimeMultiGPU - startTimeMultiGPU);
        oss << (endTimeMultiGPU - startTimeMultiGPU) << " ";
    }

    if (doCPUGPU)
    {
        shrLog("\nrunDijkstra - Multi GPU and CPU Time: %f s\n", endTimeGPUCPU - startTimeGPUCPU);
        oss << (endTimeGPUCPU - startTimeGPUCPU) << " ";
    }

    if (doRef)
    {
        shrLog("\nrunDijkstra - Reference (CPU):        %f s\n", endTimeRef - startTimeRef);
        oss << (endTimeRef - startTimeRef) << " ";
    }
    oss << "\n";
    shrLog(oss.str().c_str());

    free(sourceVertArray);
    free(results);
    free(gpuDevices);
    free(cpuDevices);

    clReleaseContext(gpuContext);

    // finish
    //shrEXIT(argc, argv);
 }
