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


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    // start logs 
    shrSetLogFileName ("oclDijkstra.txt");

    cl_context gpuContext;
    cl_int errNum;

    // start timer & logs
    shrLog(LOGBOTH, 0.0, "Setting up OpenCL on the Host...\n\n");
    shrDeltaT(1);


    // create the OpenCL context on available GPU devices
    gpuContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    shrLog(LOGBOTH, 0.0, "clCreateContextFromType\n\n");



    // Allocate memory for arrays
    GraphData graph;
    graph.vertexCount = sizeof(vertexArray) / sizeof(int);
    graph.edgeCount = sizeof(edgeArray) / sizeof(int);
    graph.vertexArray = &vertexArray[0];
    graph.edgeArray = &edgeArray[0];
    graph.weightArray = &weightArray[0];


    printf("Vertex Count: %d\n", graph.vertexCount);
    printf("Edge Count: %d\n", graph.edgeCount);

    std::vector<int> sourceVertices;
    std::vector<int> endVertices;

    for (int k = 0; k < 1000; k++)
    {
       for(int source = 0; source < graph.vertexCount; source++)
       {
           for (int i = 0; i < graph.vertexCount; i++ )
           {
               if (i != source)
               {
                   sourceVertices.push_back(source);
                   endVertices.push_back(i);
               }
           }
       }
    }

    int *sourceVertArray = (int*) malloc(sizeof(int) * sourceVertices.size());
    std::copy(sourceVertices.begin(), sourceVertices.end(), sourceVertArray);

    int *endVertArray = (int*) malloc(sizeof(int) * endVertices.size());
    std::copy(endVertices.begin(), endVertices.end(), endVertArray);

    float *results = (float*) malloc(sizeof(float) * endVertices.size());


    // Run Dijkstra's algorithm
    runDijkstra(gpuContext, &graph, sourceVertArray, endVertArray, results, sourceVertices.size() );


    for (unsigned int i = 0; i < sourceVertices.size(); i++)
    {
       printf("%s --> %s: %f\n", vNames[sourceVertArray[i]], vNames[endVertArray[i]], results[i] );
    }


    free(sourceVertArray);
    free(endVertArray);
    free(results);

#if 0
    // OpenCL
    cl_context cxGPUContext;
    cl_device_id cdDevice;                          // GPU device
    int deviceNr[MAX_GPU_COUNT];
    cl_command_queue commandQueue[MAX_GPU_COUNT];
    cl_mem d_Data[MAX_GPU_COUNT];
    cl_mem d_Result[MAX_GPU_COUNT];
    cl_program cpProgram; 
    cl_kernel reduceKernel[MAX_GPU_COUNT];
    cl_event GPUDone[MAX_GPU_COUNT];
    cl_event GPUExecution[MAX_GPU_COUNT];
    cl_uint ciDeviceCount = 0;
    size_t programLength;
    cl_int ciErrNum;			               
    char cDeviceName [256];
    cl_mem h_DataBuffer;

    // Vars for reduction results
    float h_SumGPU[MAX_GPU_COUNT * ACCUM_N];   
    float *h_Data;
    double sumGPU;
    double sumCPU, dRelError;

    // allocate and init host buffer with with some random generated input data
    h_Data = (float *)malloc(DATA_N * sizeof(float));
    shrFillArray(h_Data, DATA_N);

    // start timer & logs 
    shrLog(LOGBOTH, 0.0, "Setting up OpenCL on the Host...\n\n"); 
    shrDeltaT(1);


    // create the OpenCL context on available GPU devices
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
    shrLog(LOGBOTH, 0.0, "clCreateContextFromType\n\n"); 


    // Find out how many GPU's to compute on all available GPUs
    size_t nDeviceBytes;
    ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
        shrCheckError(ciErrNum, CL_SUCCESS);
    ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);

    for(unsigned int i = 0; i < ciDeviceCount; ++i )
    {
        // get & log device index # and name
        deviceNr[i] = i;
        cdDevice = oclGetDev(cxGPUContext, i);
        ciErrNum = clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cDeviceName), cDeviceName, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog(LOGBOTH, 0.0, " Device %i: %s\n", i, cDeviceName);

        // create a command que
        commandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog(LOGBOTH, 0.0, "clCreateCommandQueue\n\n");
    }

    // Load the OpenCL source code from the .cl file 
    const char* source_path = shrFindFilePath("dijkstra.cl", argv[0]);
    char *source = oclLoadProgSource(source_path, "", &programLength);
    shrCheckError(source != NULL, shrTRUE);
    shrLog(LOGBOTH, 0.0, "oclLoadProgSource\n"); 

    // Create the program for all GPUs in the context
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&source, &programLength, &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);
    shrLog(LOGBOTH, 0.0, "clCreateProgramWithSource\n"); 
    
    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLog(LOGBOTH | ERRORMSG, (double)ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclSimpleMultiGPU.ptx");
        shrCheckError(ciErrNum, CL_SUCCESS); 
    }
    shrLog(LOGBOTH, 0.0, "clBuildProgram\n"); 

    // Create host buffer with page-locked memory
    h_DataBuffer = clCreateBuffer(cxGPUContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                  DATA_N * sizeof(float), h_Data, &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);
    shrLog(LOGBOTH, 0.0, "clCreateBuffer (Page-locked Host)\n\n"); 

    // Create buffers for each GPU, with data divided evenly among GPU's
    int sizePerGPU = DATA_N / ciDeviceCount;
    int workOffset[MAX_GPU_COUNT];
    int workSize[MAX_GPU_COUNT];
    workOffset[0] = 0;
    for(unsigned int i = 0; i < ciDeviceCount; ++i ) 
    {
        workSize[i] = (i != (ciDeviceCount - 1)) ? sizePerGPU : (DATA_N - workOffset[i]);        

        // Input buffer
        d_Data[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, workSize[i] * sizeof(float), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog(LOGBOTH, 0.0, "clCreateBuffer (Input)\t\tDev %i\n", i); 

        // Copy data from host to device
        ciErrNum = clEnqueueCopyBuffer(commandQueue[i], h_DataBuffer, d_Data[i], workOffset[i] * sizeof(float), 
                                      0, workSize[i] * sizeof(float), 0, NULL, NULL);        
        shrLog(LOGBOTH, 0.0, "clEnqueueCopyBuffer (Input)\tDev %i\n", i);

        // Output buffer
        d_Result[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, ACCUM_N * sizeof(float), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog(LOGBOTH, 0.0, "clCreateBuffer (Output)\t\tDev %i\n", i);
        
        // Create kernel
        reduceKernel[i] = clCreateKernel(cpProgram, "reduce", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog(LOGBOTH, 0.0, "clCreateKernel\t\t\tDev %i\n", i); 
        
        // Set the args values and check for errors
        ciErrNum |= clSetKernelArg(reduceKernel[i], 0, sizeof(cl_mem), &d_Result[i]);
        ciErrNum |= clSetKernelArg(reduceKernel[i], 1, sizeof(cl_mem), &d_Data[i]);
        ciErrNum |= clSetKernelArg(reduceKernel[i], 2, sizeof(int), &workSize[i]);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog(LOGBOTH, 0.0, "clSetKernelArg\t\t\tDev %i\n\n", i);

        workOffset[i + 1] = workOffset[i] + workSize[i];
    }

    // Set # of work items in work group and total in 1 dimensional range
    size_t localWorkSize[] = {THREAD_N};        
    size_t globalWorkSize[] = {ACCUM_N};        

    // Start timer and launch reduction kernel on each GPU, with data split between them 
    shrLog(LOGBOTH, 0.0, "Launching Kernels on GPU(s)...\n\n");
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {        
        ciErrNum = clEnqueueNDRangeKernel(commandQueue[i], reduceKernel[i], 1, 0, globalWorkSize, localWorkSize,
                                         0, NULL, &GPUExecution[i]);
        shrCheckError(ciErrNum, CL_SUCCESS);
    }
    
    // Copy result from device to host for each device
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
        ciErrNum = clEnqueueReadBuffer(commandQueue[i], d_Result[i], CL_FALSE, 0, ACCUM_N * sizeof(float), 
                            h_SumGPU + i *  ACCUM_N, 0, NULL, &GPUDone[i]);
        shrCheckError(ciErrNum, CL_SUCCESS);
    }

    // Synchronize with the GPUs and do accumulated error check
    clWaitForEvents(ciDeviceCount, GPUDone);
    shrLog(LOGBOTH, 0.0, "clWaitForEvents complete...\n\n"); 

    // Aggregate results for multiple GPU's and stop/log processing time
    sumGPU = 0;
    for(unsigned int i = 0; i < ciDeviceCount * ACCUM_N; i++)
    {
         sumGPU += h_SumGPU[i];
    }


    // Run the computation on the Host CPU and log processing time 
    shrLog(LOGBOTH, 0.0, "Launching Host/CPU C++ Computation...\n\n");
    sumCPU = 0;
    for(unsigned int i = 0; i < DATA_N; i++)
    {
        sumCPU += h_Data[i];
    }

    // Check GPU result against CPU result 
    dRelError = 100.0 * fabs(sumCPU - sumGPU) / fabs(sumCPU);
    shrLog(LOGBOTH, 0.0, "Comparing against Host/C++ computation...\n"); 
    shrLog(LOGBOTH, 0.0, " GPU sum: %f\n CPU sum: %f\n", sumGPU, sumCPU);
    shrLog(LOGBOTH, 0.0, " Relative Error % (100.0 * Error / Golden) = %f \n\n", dRelError);
    shrLog(LOGBOTH, 0.0, "TEST %s\n\n", (dRelError < 1e-4) ? "PASSED" : "FAILED !!!");

    // cleanup 
    free(source);
    free(h_Data);
    for(unsigned int i = 0; i < ciDeviceCount; ++i ) 
    {
	    clReleaseKernel(reduceKernel[i]);
        clReleaseCommandQueue(commandQueue[i]);
    }
    clReleaseProgram(cpProgram);
    clReleaseContext(cxGPUContext);

#endif

    // finish
    shrEXIT(argc, argv);
 }
