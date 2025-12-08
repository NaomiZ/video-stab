#include "processor_api.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    unsigned long frame_count;
    const char* cfg_path_seen;
} DemoStabCtx;

static ProcStatus demo_stab_init(const VP_Config* cfg, void** ctx)
{
    DemoStabCtx* c = (DemoStabCtx*)malloc(sizeof(DemoStabCtx));
    if (!c)
        return PROC_STATUS_ERR_ALLOC;

    c->frame_count = 0;
    c->cfg_path_seen = (cfg && cfg->config_path) ? cfg->config_path : NULL;

    fprintf(stderr,
        "[demo-stabilizer] init: config_path=%s width=%d height=%d pixfmt=%d\n",
        c->cfg_path_seen ? c->cfg_path_seen : "(null)",
        cfg ? cfg->width : -1,
        cfg ? cfg->height : -1,
        cfg ? (int)cfg->pixfmt : -1);

    *ctx = c;
    return PROC_STATUS_OK;
}

static ProcStatus demo_stab_process(void* vctx,
                                     const VP_FrameIn* in,
                                     VP_FrameOut* out)
{
    DemoStabCtx* c = (DemoStabCtx*)vctx;
    if (!c || !in || !out)
        return PROC_STATUS_ERR_GENERAL;

    c->frame_count++;

    /* For now, this is a NO-OP stabilizer.
       It simply logs and leaves the frame unchanged.
       Later you will insert your real stabilizer code here. */
    constexpr int LOG_INTERVAL = 30;
    if (c->frame_count == 1 || (c->frame_count % LOG_INTERVAL) == 0) {
        fprintf(stderr,
            "[demo-stabilizer] process: frame=%lu in.data=%p\n",
            c->frame_count, (void*)in->data);
    }

    return PROC_STATUS_OK;
}

static void demo_stab_destroy(void* vctx)
{
    DemoStabCtx* c = (DemoStabCtx*)vctx;
    if (!c)
        return;

    fprintf(stderr,
        "[demo-stabilizer] destroy: total_frames=%lu\n",
        c->frame_count);

    free(c);
}

ProcStatus proc_register(ProcessorAPI* api)
{
    if (!api)
        return PROC_STATUS_ERR_GENERAL;

    api->init   = demo_stab_init;
    api->process = demo_stab_process;
    api->destroy = demo_stab_destroy;

    return PROC_STATUS_OK;
}
