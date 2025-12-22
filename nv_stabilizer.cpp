#include "processor_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vpi/VPI.h>
#include <vpi/algo/ConvertImageFormat.h>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << "line " << __LINE__ << ": ";                \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

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

    int width;
    int height;

    bool dims_valid;

    bool has_prev;

    VPIStream vpi_stream;

    VPIImage cur_img_y;
    VPIImage prev_img_y;

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
    c->cur_img_y = nullptr;
    c->prev_img_y = nullptr;
    c->in_wrap = nullptr;
    c->dims_valid = false;
    c->width = 0;
    c->height = 0;
    
    VPIStatus st;

    st = vpiStreamCreate(0, &c->vpi_stream);
    if (st != VPI_SUCCESS) {
        printf("[nv-stabilizer] vpiStreamCreate failed: %d\n", (int)st);
        free(c);
        return PROC_STATUS_ERR_GENERAL;
    }

    *ctx = c;

    return PROC_STATUS_OK;
}

static inline VPIStatus
nv_vpi_submit_copy(VPIStream stream, VPIBackend backend, VPIImage src, VPIImage dst)
{
    return vpiSubmitConvertImageFormat(
        stream,
        backend,
        src,
        dst,
        NULL
    );
}

static ProcStatus nv_stab_reset_context(NvStabCtx *c, int w, int h) {
    if (!c->dims_valid || c->width != w || c->height != h) {
        if (c->cur_img_y) {
            printf("[nv-stabilizer] prepare: before destroy\n");
            vpiImageDestroy(c->cur_img_y);  
            printf("[nv-stabilizer] prepare: after destroy\n");
            c->cur_img_y  = NULL; 
        }
        printf("[nv-stabilizer] prepare: step1.1 ended\n");

        if (c->prev_img_y) { 
            printf("[nv-stabilizer] prepare: before destroy\n");
            vpiImageDestroy(c->prev_img_y); 
            printf("[nv-stabilizer] prepare: after destroy\n");
            c->prev_img_y = NULL; 
        }
        printf("[nv-stabilizer] prepare: step1.2 ended\n");
        VPIStatus st;
        st = vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8, 0, &c->cur_img_y);
        if (st != VPI_SUCCESS){
            printf("[nv-stabilizer] prepare: vpiImageCreate failed status:%d\n", st);
            return PROC_STATUS_ERR_NOMEM;
        }
        printf("[nv-stabilizer] prepare: step1.3 ended\n");

        st = vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8, 0, &c->prev_img_y);
        printf("[nv-stabilizer] prepare: step1.4 ended\n");

        if (st != VPI_SUCCESS)
            return PROC_STATUS_ERR_NOMEM;

        c->has_prev = 0;
        printf("[nv-stabilizer] prepare: step1 ended\n");
    }
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_prepare_frame(NvStabCtx* c, VP_Frame* frame)
{
    printf("[nv-stabilizer] prepare: begin\n");
    VPIStatus st;
    if (!c || !frame)
        return PROC_STATUS_ERR_GENERAL;

    if (frame->pixfmt != PROC_PIXFMT_NV12) {
        printf("[nv-stabilizer] prepare: unsupported pixfmt\n");
        return PROC_STATUS_ERR_UNSUPPORTED;
    }

    if (!frame->data || frame->width <= 0 || frame->height <= 0 || frame->stride <= 0) {
        printf("[nv-stabilizer] prepare: invalid frame fields\n");
        return PROC_STATUS_ERR_GENERAL;
    }

    const int w = frame->width;
    const int h = frame->height;
    const int stride = frame->stride;
    printf("[nv-stabilizer] prepare: step0 ended\n");

    if(nv_stab_reset_context(c, w, h) != PROC_STATUS_OK) {
        printf("[nv-stabilizer] prepare: reset_context failed\n");
        return PROC_STATUS_ERR_GENERAL;
    }

    /* ------------------------------------------------------------------
     * 2. Destroy stale wrapper from previous call (must not persist)
     * ------------------------------------------------------------------ */
    if (c->in_wrap) {
        vpiImageDestroy(c->in_wrap);
        c->in_wrap = NULL;
    }

    printf("[nv-stabilizer] prepare: step2 ended\n");

    /* ------------------------------------------------------------------
     * 3. Wrap VP_Frame memory as NV12 pitch-linear
     *    Assumption (temporary): contiguous NV12 (Y then UV)
     * ------------------------------------------------------------------ */
    uint8_t* y  = frame->data;
    uint8_t* uv = frame->data + (size_t)stride * h;

    VPIImageData inData;
    memset(&inData, 0, sizeof(inData));

    inData.bufferType = VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR;
    inData.buffer.pitch.format = VPI_IMAGE_FORMAT_NV12;
    inData.buffer.pitch.numPlanes = 2;

    inData.buffer.pitch.planes[0].data = y;
    inData.buffer.pitch.planes[0].pitchBytes = stride;
    inData.buffer.pitch.planes[0].width = w;
    inData.buffer.pitch.planes[0].height = h;

    inData.buffer.pitch.planes[1].data = uv;
    inData.buffer.pitch.planes[1].pitchBytes = stride;
    inData.buffer.pitch.planes[1].width = w;
    inData.buffer.pitch.planes[1].height = h / 2;

    st = vpiImageCreateWrapper(&inData, NULL, 0, &c->in_wrap);
    if (st != VPI_SUCCESS) {
        c->in_wrap = NULL;
        return PROC_STATUS_ERR_GENERAL;
    }

    printf("[nv-stabilizer] prepare: step3 ended\n");

    /* ------------------------------------------------------------------
     * 4. Copy caller frame into VPI-owned cur_img_y
     * ------------------------------------------------------------------ */
    st = nv_vpi_submit_copy(c->vpi_stream,
                            VPI_BACKEND_CPU,
                            c->in_wrap,
                            c->cur_img_y);
    if (st != VPI_SUCCESS)
        return PROC_STATUS_ERR_GENERAL;

    /* ------------------------------------------------------------------
     * 5. Initialize prev_img_y on first usable frame
     * ------------------------------------------------------------------ */
    if (!c->has_prev) {
        st = nv_vpi_submit_copy(c->vpi_stream,
                                VPI_BACKEND_CPU,
                                c->cur_img_y,
                                c->prev_img_y);
        if (st != VPI_SUCCESS)
            return PROC_STATUS_ERR_GENERAL;

        c->has_prev = 1;
    }

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
