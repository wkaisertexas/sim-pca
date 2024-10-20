# Using NVBit for Trace Simulation

[NVBit Paper](https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus)
[GitHub](https://github.com/NVlabs/NVBit)

Compilation of a NVBit tool you make a tool using nvbit and then you attach this binary as an `LD_PRELOAD`.

```bash
nvcc -shared nvbit_tool.cu -lnvbit
```

Runtime usage of the NVBit tool,

```bash
LD_PRELOAD=nvbit_tool.so ./main
```

Basic NVBit example:

```cpp
/* Counter variable used to count instructions */
__managed__ long counter = 0;

/* Used to keep track of kernels already instrumented */
std::set<CUfunction> instrumented_kernels;

/* Implementation of instrumentation function */
extern "C" __device__ __noinline__ void incr_counter() {
 atomicAdd(&counter, 1);
} NVBIT_EXPORT_DEV_FUNC(incr_counter);

/* Callback triggered on CUDA driver call */
void nvbit_at_cuda_driver_call(CUcontext ctx,
int is_exit, cbid_t cbid, const char *name,
void *params, CUresult *pStatus) {

/* Return if not at the entry of a kernel launch */
if (cbid != API_CUDA_cuLaunchKernel || is_exit) return;

/* Get parameters of the kernel launch */
cuLaunchKernel_params *p = (cuLaunchKernel_params *) params;

/* Return if kernel is already instrumented */
if(!instrumented_kernels.insert(p->func).second)
return;

/* Instrument all instructions in the kernel */
for (auto &i: nvbit_get_instrs(ctx, p->func)) {
 nvbit_insert_call(i, "incr_counter", IPOINT_BEFORE);
}
}

/* Callback triggered on application termination */
void nvbit_at_term() {
cout << "Total thread instructions " << counter << "\n";
}
```

## Accel-Sim SASS Tracer

I am supposed to be able to use the Accel-Sim SASS Tracer, but there is a chance that I do not have NVBit installed.