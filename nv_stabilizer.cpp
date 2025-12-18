#include "processor_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vpi/VPI.h>

constexpr static int max_history_size = 10;
typedef struct {
    int dx[max_history_size];
    int dy[max_history_size];
    int curr_index;
} NvStabHistory;

typedef struct {
    uint64_t frame_count;

    const char* cfg_path_seen;
    NvStabHistory history;
    int history_index;
    int prev_cx, prev_cy;

    bool has_prev;
    bool dims_valid;

    VPIStream stream;

    VPIImage cur_img;
    VPIImage prev_img;

    VPIImage in_wrap;   // wrapper around VP_Frame memory (per-call)

} NvStabCtx;

static ProcStatus nv_stab_init(const char* config_path, void** ctx)
{
    NvStabCtx* c = (NvStabCtx*)malloc(sizeof(NvStabCtx));
    if (!c)
        return PROC_STATUS_ERR_ALLOC;

    c->frame_count = 0;
    c->cfg_path_seen = config_path ? strdup(config_path) : nullptr;
    c->history = {};
    c->prev_cx = 0;
    c->prev_cy = 0;
    c->has_prev = false;

    *ctx = c;

    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_prepare_frame(NvStabCtx* c, VP_Frame* frame)
{
    (void)c;
    (void)frame;
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_compute_flow(NvStabCtx* c)
{
    (void)c;
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_compute_transform(NvStabCtx* c)
{
    (void)c;
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_apply_transform(NvStabCtx* c)
{
    (void)c;
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_process(void* vctx, VP_Frame* frame)
{
    NvStabCtx* c = (NvStabCtx*)vctx;
    if (!c || !frame)
        return PROC_STATUS_ERR_GENERAL;

    c->frame_count++;
    printf("[nv-stabilizer] process: frame=%lu\n", c->frame_count);

    ProcStatus st;

    st = nv_stab_prepare_frame(c, frame);
    if (st != PROC_STATUS_OK)
        return st;

    st = nv_stab_compute_flow(c);
    if (st != PROC_STATUS_OK)
        return st;

    st = nv_stab_compute_transform(c);
    if (st != PROC_STATUS_OK)
        return st;

    st = nv_stab_apply_transform(c);
    if (st != PROC_STATUS_OK)
        return st;

    return PROC_STATUS_OK;
}

static void nv_stab_destroy(void* vctx)
{
    NvStabCtx* c = (NvStabCtx*)vctx;
    if (!c)
        return;

    fprintf(stderr,
        "[nv-stabilizer] destroy: total_frames=%lu\n",
        c->frame_count);

    free(c);
}

ProcStatus proc_register(ProcessorAPI* api)
{
    if (!api)
    return PROC_STATUS_ERR_GENERAL;

    api->init   = nv_stab_init;
    api->process = nv_stab_process;
    api->destroy = nv_stab_destroy;

    return PROC_STATUS_OK;
}
