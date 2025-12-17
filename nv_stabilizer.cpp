#include "processor_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr static int max_history_size = 10;
typedef struct {
    int dx[max_history_size];
    int dy[max_history_size];
    int curr_index;
} NvStabHistory;

typedef struct {
    unsigned long frame_count;
    const char* cfg_path_seen;
    NvStabHistory history;
    int history_index;
    int prev_cx, prev_cy;
    bool has_prev;
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


static ProcStatus nv_stab_process(void* vctx, VP_Frame* frame)
{
    NvStabCtx* c = (NvStabCtx*)vctx;
    if (!c || !frame)
        return PROC_STATUS_ERR_GENERAL;

    c->frame_count++;
    printf("[nv-stabilizer] process: frame=%lu\n", c->frame_count);

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
