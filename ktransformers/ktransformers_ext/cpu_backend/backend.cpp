/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:34
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "backend.h"

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <iostream>

thread_local int Backend::numa_node = -1;
#endif

thread_local int Backend::thread_local_id = -1;

Backend::Backend(int max_thread_num) {
#ifdef USE_NUMA
    const int num_numa_node = numa_num_configured_nodes();
    if (num_numa_node <= 0) {
        printf("[Backend::Backend] No NUMA nodes configured\n");
        exit(EXIT_FAILURE);
    }
    threads_on_each_numa_node.resize(num_numa_node);

    {
        std::unique_lock<std::mutex> lock(mux);
        threads_on_each_numa_node[0].push_back(0);
    }
#endif
    max_thread_num_ = max_thread_num;
    thread_state_.resize(max_thread_num_);
    for (int i = 0; i < max_thread_num_; i++) {
        thread_state_[i].curr = std::make_unique<std::atomic<int>>();
        thread_state_[i].status =
            std::make_unique<std::atomic<ThreadStatus>>(ThreadStatus::WAITING);
    }
    workers_.resize(max_thread_num_);
    for (int i = 1; i < max_thread_num_; i++) {
        workers_[i] = std::thread(&Backend::worker_thread, this, i);
    }
#ifdef USE_NUMA
    // wait for numa_info ready
    while (numa_info_ready_cnter.load(std::memory_order_relaxed) != max_thread_num_);

    int check_thread_num = 0;
    for (int i = 0; i < num_numa_node; ++i) {
        check_thread_num += threads_on_each_numa_node[i].size();
        printf("threads on numa node [%d]: ");
        for (auto t_i : threads_on_each_numa_node[i]) {
            printf("%d, ", t_i);
        }
        printf("\n");
    }
    if (check_thread_num != max_thread_num_) {
        printf("threads_on_each_numa_node thread count mismatch with max_thread_num_: %d expected, get %d\n", max_thread_num_, check_thread_num);
        exit(EXIT_FAILURE);
    }
#endif
}

Backend::~Backend() {
    for (int i = 0; i < max_thread_num_; i++) {
        thread_state_[i].status->store(ThreadStatus::EXIT,
                                       std::memory_order_release);
    }
    for (int i = 1; i < max_thread_num_; i++) {
        if (workers_[i].joinable()) {
            workers_[i].join();
        }
    }
}

int Backend::get_thread_num() { return max_thread_num_; }

#ifdef USE_NUMA
void Backend::do_work_stealing_job_numa_aware(
    int task_num, 
    std::vector<int>& task_splits,
    std::function<void(int)> init_func,
    std::function<void(int)> compute_func,
    std::function<void(int)> finalize_func
) {
    int n_numa_node = threads_on_each_numa_node.size();
    if (n_numa_node != task_splits.size()) {
        printf("size of task_splits mismatch with numa nodes: %d expected, get %ld\n", n_numa_node, task_splits.size());
        exit(EXIT_FAILURE);
    }
    int tmp = 0;
    for (auto task_cnt : task_splits) {
        tmp += task_cnt;
    }
    if (tmp != task_num) {
        printf("sum of task_splits mismatch with task_num: %d expected, get %d\n", task_num,tmp);
        exit(EXIT_FAILURE);
    }

    init_func_ = init_func;
    compute_func_ = compute_func;
    finalize_func_ = finalize_func;

    // numa node location will be calculated based on the number of threads
    thread_num_ = max_thread_num_;

    for (int numa_node_id = 0; numa_node_id < n_numa_node; ++numa_node_id) {
        // split task_num by numa_node_id
        int task_cnt_cur_numa_node = task_splits[numa_node_id];
        // assign tasks to the threads which are running on this numa node
        int n_threads_cur_numa_node = threads_on_each_numa_node[numa_node_id].size();
        int base = task_cnt_cur_numa_node / n_threads_cur_numa_node;
        int remain = task_cnt_cur_numa_node % n_threads_cur_numa_node;

        thread_state_[threads_on_each_numa_node[numa_node_id][0]].curr->store(0, std::memory_order_relaxed);
        thread_state_[threads_on_each_numa_node[numa_node_id][0]].end = base + (0 < remain);
        for (int i = 1; i < n_threads_cur_numa_node; ++i) {
            thread_state_[threads_on_each_numa_node[numa_node_id][i]].curr->store(
                thread_state_[threads_on_each_numa_node[numa_node_id][i - 1]].end,
                std::memory_order_relaxed);
            thread_state_[threads_on_each_numa_node[numa_node_id][i]].end = 
                thread_state_[threads_on_each_numa_node[numa_node_id][i - 1]].end + base + (i < remain);
            thread_state_[threads_on_each_numa_node[numa_node_id][i]].status->store(ThreadStatus::WORKING, std::memory_order_release);
        }
    }

    thread_local_id = 0;
    process_tasks(0);
    for (int i = 1; i < thread_num_; i++) {
        while (thread_state_[i].status->load(std::memory_order_acquire) ==
               ThreadStatus::WORKING) {
        }
    }
}
#endif

void Backend::do_work_stealing_job(int task_num,
                                   std::function<void(int)> init_func,
                                   std::function<void(int)> compute_func,
                                   std::function<void(int)> finalize_func) {
    init_func_ = init_func;
    compute_func_ = compute_func;
    finalize_func_ = finalize_func;
#ifdef USE_NUMA
    // numa node location will be calculated based on the number of threads
    thread_num_ = max_thread_num_;
#else
    thread_num_ = std::min(max_thread_num_, task_num);
#endif
    int base = task_num / thread_num_;
    int remain = task_num % thread_num_;
    thread_state_[0].end = base + (0 < remain);

    // 为主线程设置 thread_local_id
    thread_local_id = 0;

    for (int i = 1; i < thread_num_; i++) {
        thread_state_[i].curr->store(thread_state_[i - 1].end,
                                     std::memory_order_relaxed);
        thread_state_[i].end = thread_state_[i - 1].end + base + (i < remain);
        thread_state_[i].status->store(ThreadStatus::WORKING,
                                       std::memory_order_release);
    }
    thread_state_[0].curr->store(0, std::memory_order_relaxed);
    thread_state_[0].status->store(ThreadStatus::WORKING,
                                   std::memory_order_release);
    process_tasks(0);
    for (int i = 1; i < thread_num_; i++) {
        while (thread_state_[i].status->load(std::memory_order_acquire) ==
               ThreadStatus::WORKING) {
        }
    }
}

void Backend::process_tasks(int thread_id) {
    
    #ifdef USE_NUMA
    if (numa_node == -1) {
        if (thread_id != 0) {
            printf("Only thread-0's numa node is not set, while get thread_id: %d\n", thread_id);
            exit(EXIT_FAILURE);
        }
        numa_node = 0;
        struct bitmask* mask = numa_bitmask_alloc(numa_num_configured_nodes());
        numa_bitmask_setbit(mask, numa_node);
        numa_bind(mask);
    }
    #endif

    if (init_func_ != nullptr) {
        init_func_(thread_id);
    }
    while (true) {
        int task_id = thread_state_[thread_id].curr->fetch_add(
            1, std::memory_order_acq_rel);
        if (task_id >= thread_state_[thread_id].end) {
            break;
        }
        compute_func_(task_id);
    }
#ifdef USE_NUMA
    // steal jobs only on the same numa node
    for (auto t_i : threads_on_each_numa_node[numa_node]) {
        if (t_i == thread_id) {
            continue;
        }
        if (thread_state_[t_i].status->load(std::memory_order_acquire) !=
            ThreadStatus::WORKING) {
            continue;
        }
        while (true) {
            int task_id = thread_state_[t_i].curr->fetch_add(
                1, std::memory_order_acq_rel);
            if (task_id >= thread_state_[t_i].end) {
                break;
            }
            compute_func_(task_id);
        }
    }
#else
    for (int t_offset = 1; t_offset < thread_num_; t_offset++) {
        int t_i = (thread_id + t_offset) % thread_num_;
        if (thread_state_[t_i].status->load(std::memory_order_acquire) !=
            ThreadStatus::WORKING) {
            continue;
        }
        while (true) {
            int task_id = thread_state_[t_i].curr->fetch_add(
                1, std::memory_order_acq_rel);
            if (task_id >= thread_state_[t_i].end) {
                break;
            }
            compute_func_(task_id);
        }
    }
#endif
    if (finalize_func_ != nullptr) {
        finalize_func_(thread_id);
    }
    thread_state_[thread_id].status->store(ThreadStatus::WAITING,
                                           std::memory_order_release);
}

void Backend::worker_thread(int thread_id) {
#ifdef USE_NUMA
    // bind numa here
    const int num_numa_node = numa_num_configured_nodes();
    if (num_numa_node <= 0) {
        printf("No NUMA nodes configured\n");
        exit(EXIT_FAILURE);
    }

    const int base_thread_cnt_per_numa = max_thread_num_ / num_numa_node;
    const int base_thread_cnt = base_thread_cnt_per_numa * num_numa_node;
    int numa_node_id = (thread_id < base_thread_cnt) ? (thread_id / base_thread_cnt_per_numa) : (thread_id - base_thread_cnt);
    printf("thread-[%d]: base_thread_cnt_per_numa=%d, base_thread_cnt=%d, numa_node_id=%d\n", thread_id, base_thread_cnt_per_numa, base_thread_cnt, numa_node_id);
    if (numa_node_id < 0 || numa_node_id >= num_numa_node) {
        printf("Invalid numa node id: %d\n", numa_node_id);
        exit(EXIT_FAILURE);
    }

    struct bitmask* mask = numa_bitmask_alloc(num_numa_node);
    numa_bitmask_setbit(mask, numa_node_id);
    numa_bind(mask);

    numa_node = numa_node_id;
    {
        std::unique_lock<std::mutex> lock(mux);
        threads_on_each_numa_node[numa_node].push_back(thread_id);
    }
    numa_info_ready_cnter.fetch_add(1, std::memory_order_relaxed);
#endif
    auto start = std::chrono::steady_clock::now();
    thread_local_id = thread_id; // 设置线程本地变量
    while (true) {
        ThreadStatus status =
            thread_state_[thread_id].status->load(std::memory_order_acquire);
        if (status == ThreadStatus::WORKING) {
            process_tasks(thread_id);
            start = std::chrono::steady_clock::now();
        } else if (status == ThreadStatus::WAITING) {
            auto now = std::chrono::steady_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                      start)
                    .count();
            if (duration > 50) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (status == ThreadStatus::EXIT) {
            return;
        }
    }
}
