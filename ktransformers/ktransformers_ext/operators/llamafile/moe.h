/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_MOE_H
#define CPUINFER_OPERATOR_MOE_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>

#include "../../cpu_backend/backend.h"
#include "conversion.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"
#include "shared_mem_buffer.h"

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

struct MOEConfig {
    int expert_num;
    int routed_expert_num;
    int hidden_size;       // 7168
    int intermediate_size; // 2048
    int stride;         // 64
    int group_min_len;  // 10
    int group_max_len;  // 1024
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    ggml_type gate_type;
    ggml_type up_type;
    ggml_type down_type;
    ggml_type hidden_type;

#ifdef USE_NUMA
    int e_n_numa_nodes;
    
    int e_n_gate_proj_strides_per_expert;
    int e_n_gate_proj_strides_per_expert_per_node;
    int e_n_gate_remain_nodes;

    int e_n_up_proj_strides_per_expert;
    int e_n_up_proj_strides_per_expert_per_node; // = e_n_gate_proj_strides_per_expert_per_node
    int e_n_up_remain_nodes;

    int e_n_down_proj_strides_per_expert;
    int e_n_down_proj_strides_per_expert_per_node;
    int e_n_down_remain_nodes;
#endif

    MOEConfig() {}

    MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, void* gate_proj, void* up_proj, void* down_proj, ggml_type gate_type, ggml_type up_type, ggml_type down_type, ggml_type hidden_type)
        : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size), intermediate_size(intermediate_size), stride(stride), group_min_len(group_min_len), group_max_len(group_max_len), gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type) {
#ifdef USE_NUMA
        e_n_numa_nodes = numa_num_configured_nodes();
        if (e_n_numa_nodes <= 0) {
            printf("[MOE] No NUMA nodes configured\n");
            exit(EXIT_FAILURE);
        }

        e_n_gate_proj_strides_per_expert = intermediate_size / stride;
        int avg_gate_proj_strides_per_expert_per_node = e_n_gate_proj_strides_per_expert / e_n_numa_nodes;
        e_n_gate_remain_nodes = e_n_gate_proj_strides_per_expert % e_n_numa_nodes;
        e_n_gate_proj_strides_per_expert_per_node = (e_n_gate_remain_nodes > 0) ? avg_gate_proj_strides_per_expert_per_node + 1 : avg_gate_proj_strides_per_expert_per_node;

        e_n_up_proj_strides_per_expert = intermediate_size / stride;
        int avg_up_proj_strides_per_expert_per_node = e_n_up_proj_strides_per_expert / e_n_numa_nodes;
        e_n_up_remain_nodes = e_n_up_proj_strides_per_expert % e_n_numa_nodes;
        e_n_up_proj_strides_per_expert_per_node = (e_n_up_remain_nodes > 0) ? avg_up_proj_strides_per_expert_per_node + 1 : avg_up_proj_strides_per_expert_per_node;

        e_n_down_proj_strides_per_expert = hidden_size / stride;
        int avg_down_proj_strides_per_expert_per_node = e_n_down_proj_strides_per_expert / e_n_numa_nodes;
        e_n_down_remain_nodes = e_n_down_proj_strides_per_expert % e_n_numa_nodes;
        e_n_down_proj_strides_per_expert_per_node = (e_n_down_remain_nodes > 0) ? avg_down_proj_strides_per_expert_per_node + 1 : avg_down_proj_strides_per_expert_per_node;
#endif
    }

#ifdef USE_NUMA
    int gate_stride_numa_node_by_stride_id(int stride_id, int& num_strides) {
        if (e_n_gate_remain_nodes == 0 ||
            stride_id < e_n_gate_remain_nodes * e_n_gate_proj_strides_per_expert_per_node) {
            num_strides = e_n_gate_proj_strides_per_expert_per_node;
            return stride_id / e_n_gate_proj_strides_per_expert_per_node;
        }

        num_strides = e_n_gate_proj_strides_per_expert_per_node - 1;
        return e_n_gate_remain_nodes + (stride_id - e_n_gate_remain_nodes * e_n_gate_proj_strides_per_expert_per_node) / num_strides;
    }

    int up_stride_numa_node_by_stride_id(int stride_id, int& num_strides) {
        if (e_n_up_remain_nodes == 0 ||
            stride_id < e_n_up_remain_nodes * e_n_up_proj_strides_per_expert_per_node) {
            num_strides = e_n_up_proj_strides_per_expert_per_node;
            return stride_id / e_n_up_proj_strides_per_expert_per_node;
        }

        num_strides = e_n_up_proj_strides_per_expert_per_node - 1;
        return e_n_up_remain_nodes + (stride_id - e_n_up_remain_nodes * e_n_up_proj_strides_per_expert_per_node) / num_strides;
    }

    int down_stride_numa_node_by_stride_id(int stride_id, int& num_strides) {
        if (e_n_down_remain_nodes == 0 ||
            stride_id < e_n_down_remain_nodes * e_n_down_proj_strides_per_expert_per_node) {
            num_strides = e_n_down_proj_strides_per_expert_per_node;
            return stride_id / e_n_down_proj_strides_per_expert_per_node;
        }

        num_strides = e_n_down_proj_strides_per_expert_per_node - 1;
        return e_n_down_remain_nodes + (stride_id - e_n_down_remain_nodes * e_n_down_proj_strides_per_expert_per_node) / num_strides;
    }

    size_t gate_proj_element_size_on_numa_node(int numa_node_id) {
        if (e_n_gate_remain_nodes == 0 ||
            numa_node_id < e_n_gate_remain_nodes) {
            return (size_t)expert_num * e_n_gate_proj_strides_per_expert_per_node * stride * hidden_size;
        }
        return (size_t)expert_num * (e_n_gate_proj_strides_per_expert_per_node - 1) * stride * hidden_size;
    }

    size_t up_proj_element_size_on_numa_node(int numa_node_id) {
        if (e_n_up_remain_nodes == 0 ||
            numa_node_id < e_n_up_remain_nodes) {
            return (size_t)expert_num * e_n_up_proj_strides_per_expert_per_node * stride * hidden_size;
        }
        return (size_t)expert_num * (e_n_up_proj_strides_per_expert_per_node - 1) * stride * hidden_size;
    }

    size_t down_proj_element_size_on_numa_node(int numa_node_id) {
        if (e_n_down_remain_nodes == 0 ||
            numa_node_id < e_n_down_remain_nodes) {
            return (size_t)expert_num * e_n_down_proj_strides_per_expert_per_node * stride * intermediate_size;
        }
        return (size_t)expert_num * (e_n_down_proj_strides_per_expert_per_node - 1) * stride * intermediate_size;
    }
#endif
};

class MOE {
   public:
    MOE(MOEConfig);
    ~MOE();
    void warm_up(Backend* backend);
    void forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend);
    void forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend);
    void forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend);
#ifdef USE_NUMA
    static void* numa_alloc_huge_pages(size_t mem_size, int numa_id);
#endif

   private:
    MOEConfig config_;
    void* gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

    #ifdef USE_NUMA
    std::vector<void*> gate_proj_numa_;  // [numa_num, expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    std::vector<void*> up_proj_numa_;    // [numa_num, expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    std::vector<void*> down_proj_numa_;  // [numa_num, expert_num * hidden_size * intermediate_size ( /32 if quantized)]
    #endif

    float* s_input_fp32_;                      // [hidden_size]
    uint8_t* s_gate_input_;                    // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* s_up_input_;                      // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    std::vector<float*> s_gate_output_;        // [routed_expert_num, intermediate_size]
    std::vector<float*> s_up_output_;          // [routed_expert_num, intermediate_size]
    std::vector<float*> s_intermediate_fp32_;  // [routed_expert_num, intermediate_size]
    std::vector<uint8_t*> s_down_input_;       // [routed_expert_num, intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    std::vector<float*> s_down_output_;        // [routed_expert_num, hidden_size]
    float* s_output_fp32_;                     // [hidden_size]

    std::vector<float*> m_input_fp32_;    // [group_max_len, hidden_size]
    std::vector<uint8_t*> m_gate_input_;  // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    std::vector<uint8_t*> m_up_input_;    // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    uint8_t* m_local_gate_input_;         // [routed_expert_num * group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* m_local_up_input_;           // [routed_expert_num * group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    float* m_local_gate_output_;          // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_up_output_;            // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_intermediate_fp32_;    // [routed_expert_num * group_max_len * intermediate_size]
    uint8_t* m_local_down_input_;         // [routed_expert_num * group_max_len * intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    float* m_local_down_output_;          // [routed_expert_num * group_max_len * hidden_size]
    std::vector<float*> m_output_fp32_;   // [group_max_len, hidden_size]

    std::vector<std::vector<int>> m_local_pos_;          // [group_max_len, routed_expert_num]
    std::vector<int> m_local_num_;                       // [expert_num]
    std::vector<uint8_t*> m_local_gate_input_ptr_;       // [expert_num]
    std::vector<uint8_t*> m_local_up_input_ptr_;         // [expert_num]
    std::vector<float*> m_local_gate_output_ptr_;        // [expert_num]
    std::vector<float*> m_local_up_output_ptr_;          // [expert_num]
    std::vector<float*> m_local_intermediate_fp32_ptr_;  // [expert_num]
    std::vector<uint8_t*> m_local_down_input_ptr_;       // [expert_num]
    std::vector<float*> m_local_down_output_ptr_;        // [expert_num]
};

#endif