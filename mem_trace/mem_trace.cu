#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <sys/socket.h>
#include "cache.h"

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

#define HEX(x)                                                            \
    "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x \
         << std::dec

#define CHANNEL_SIZE (1l << 20)

struct CTXstate {
    int id;
    ChannelDev* channel_dev;
    ChannelHost channel_host;
    volatile bool recv_thread_done = false;
};

pthread_mutex_t mutex;
pthread_mutex_t cuda_event_mutex;
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;
bool skip_callback_flag = false;
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;
uint64_t grid_launch_id = 0;
int sock_fd;

void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;

    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];
    ChannelHost* ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);

    // 创建 Cache 模拟器实例，您可以根据需要设置缓存大小、块大小、关联度和替换策略
   Cache cache_simulator(10, 2, 32, Cache::LRU);

    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);
    while (!ctx_state->recv_thread_done) {
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                mem_access_t* ma = (mem_access_t*)&recv_buffer[num_processed_bytes];
                for (int i = 0; i < 32; i++) {
                    uint64_t addr = ma->addrs[i];

                    // 直接调用cache模拟器
                    if (cache_simulator.read(addr)) {
                        // 处理命中
                    } else {
                        // 处理未命中，并加载到缓存
                        cache_simulator.load(addr);
                        logMissAddress("/home/cc/nvbit_release/tools/mem_trace4/misses1.log", addr); // 记录miss地址
                    }
                }
                num_processed_bytes += sizeof(mem_access_t);
            }
        }
    }

    // 打印统计信息
    uint64_t total_accesses = cache_simulator.getTotalAccesses();
    uint64_t total_hits = cache_simulator.getTotalHits();
    uint64_t total_misses = cache_simulator.getTotalMisses();
    double hit_ratio = (total_accesses > 0) ? static_cast<double>(total_hits) / total_accesses : 0.0;

    printf("Total traces: %lu\n", total_accesses);
    printf("Total hits: %lu\n", total_hits);
    printf("Total misses: %lu\n", total_misses);
    printf("Hit ratio: %.4f\n", hit_ratio);

    ctx_state->recv_thread_done = false;
    free(recv_buffer);
    return NULL;
}

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(instr_begin_interval, "INSTR_BEGIN", 0,
                "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
                "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);
    pthread_mutex_init(&cuda_event_mutex, &attr);

    // 不再需要创建管道和子进程
}

std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);
    related_functions.push_back(func);

    for (auto f : related_functions) {
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);

        if (verbose) {
            fprintf(stderr,
                "MEMTRACE: CTX %p, Inspecting CUfunction %p name %s at address "
                "0x%lx\n",
                ctx, f, nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
        }

        uint32_t cnt = 0;
        for (auto instr : instrs) {
            if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
                instr->getMemorySpace() == InstrType::MemorySpace::NONE ||
                instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT) {
                cnt++;
                continue;
            }
            if (verbose) {
                instr->printDecoded();
            }

            if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
                int opcode_id = opcode_to_id_map.size();
                opcode_to_id_map[instr->getOpcode()] = opcode_id;
                id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
            }

            int opcode_id = opcode_to_id_map[instr->getOpcode()];
            int mref_idx = 0;
            for (int i = 0; i < instr->getNumOperands(); i++) {
                const InstrType::operand_t* op = instr->getOperand(i);
                if (op->type == InstrType::OperandType::MREF) {
                    nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
                    nvbit_add_call_arg_guard_pred_val(instr);
                    nvbit_add_call_arg_const_val32(instr, opcode_id);
                    nvbit_add_call_arg_mref_addr64(instr, mref_idx);
                    nvbit_add_call_arg_launch_val64(instr, 0);
                    nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
                    mref_idx++;
                }
            }
            cnt++;
        }
    }
}

__global__ void flush_channel(ChannelDev* ch_dev) { ch_dev->flush(); }

void init_context_state(CUcontext ctx) {
    CTXstate* ctx_state = ctx_state_map[ctx];
    ctx_state->recv_thread_done = false;
    cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                                 ctx_state->channel_dev, recv_thread_fun, ctx);
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    pthread_mutex_lock(&cuda_event_mutex);

    if (skip_callback_flag) {
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }
    skip_callback_flag = true;

    if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchCooperativeKernel ||
        cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        CTXstate* ctx_state = ctx_state_map[ctx];
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
        }

        if (!is_exit) {
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);

            instrument_function_if_needed(ctx, func);

            int nregs = 0;
            CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

            int shmem_static_nbytes = 0;
            CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes,
                                             CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

            const char* func_name = nvbit_get_func_name(ctx, func);
            uint64_t pc = nvbit_get_func_addr(func);

            nvbit_set_at_launch(ctx, func, (uint64_t)&grid_launch_id);
            nvbit_enable_instrumented(ctx, func, true);

        } else {
            cudaDeviceSynchronize();
            flush_channel<<<1, 1>>>(ctx_state->channel_dev);

            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);

            grid_launch_id++;
        }
    }
    skip_callback_flag = false;
    pthread_mutex_unlock(&cuda_event_mutex);
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) == ctx_state_map.end());
    CTXstate* ctx_state = new CTXstate;
    ctx_state_map[ctx] = ctx_state;
    pthread_mutex_unlock(&mutex);
}

void nvbit_tool_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    init_context_state(ctx);
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    skip_callback_flag = true;
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    ctx_state->recv_thread_done = true;
    while (!ctx_state->recv_thread_done)
        ;

    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback_flag = false;
    delete ctx_state;
    pthread_mutex_unlock(&mutex);
}
