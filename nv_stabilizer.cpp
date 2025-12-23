#include "processor_api.h"
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <vpi/VPI.h>
#include <vpi/algo/ConvertImageFormat.h>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/OpticalFlowDense.h>

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
    int stride;

    bool dims_valid;

    bool has_prev;

    VPIStream vpi_stream;

    VPIImage cur_img_y;
    VPIImage prev_img_y;

    VPIImage in_wrap;   // wrapper around VP_Frame memory (per-call)

    // OFA config (tune later)
    int grid_size;
    int num_levels;
    VPIOpticalFlowQuality quality;

    // OFA resources
    bool ofa_inited;

    VPIPayload ofa_payload;

    // Pyramids: pitch-linear (tmp) + block-linear (BL)
    VPIPyramid prev_pyr_pl;
    VPIPyramid cur_pyr_pl;
    VPIPyramid prev_pyr_bl;
    VPIPyramid cur_pyr_bl;

    // Motion vectors: BL output from OFA + PL for CPU access later
    VPIImage mv_bl;
    VPIImage mv_pl;

    // Stabilization
    int estimated_dx;
    int estimated_dy;

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
    c->stride = 0;

    c->estimated_dx = 0;
    c->estimated_dy = 0;

    VPIStatus st;

    st = vpiStreamCreate(0, &c->vpi_stream);
    if (st != VPI_SUCCESS) {
        printf("[nv-stabilizer] vpiStreamCreate failed: %d\n", (int)st);
        free(c);
        return PROC_STATUS_ERR_GENERAL;
    }


    // OFA config (reduced for latency)
    c->grid_size   = 8;   // coarser grid reduces compute
    c->num_levels  = 4;   // fewer pyramid levels
    c->quality = VPI_OPTICAL_FLOW_QUALITY_LOW; // faster, less precise

    // OFA resources
    c->ofa_inited = false;

    c->ofa_payload = nullptr;

    // Pyramids: pitch-linear (tmp) + block-linear (BL)
    c->prev_pyr_pl = nullptr;
    c->cur_pyr_pl  = nullptr;
    c->prev_pyr_bl = nullptr;
    c->cur_pyr_bl  = nullptr;

    // Motion vectors: BL output from OFA + PL for CPU access later
    c->mv_bl = nullptr;   // VPI_IMAGE_FORMAT_2S16_BL
    c->mv_pl = nullptr;   // VPI_IMAGE_FORMAT_2S16

    *ctx = c;

    return PROC_STATUS_OK;
}

static void shift_plane(uint8_t* data, int width, int height, int stride, int dx, int dy) {
    uint8_t* temp = (uint8_t*)malloc(width * height);
    if (!temp) return; // error, but for simplicity

    for (int y = 0; y < height; y++) {
        int sy = y - dy;
        if (sy < 0 || sy >= height) {
            memset(temp + y * width, 0, width);
            continue;
        }
        for (int x = 0; x < width; x++) {
            int sx = x - dx;
            if (sx >= 0 && sx < width) {
                temp[y * width + x] = data[sy * stride + sx];
            } else {
                temp[y * width + x] = 0;
            }
        }
    }

    // Copy back
    for (int y = 0; y < height; y++) {
        memcpy(data + y * stride, temp + y * width, width);
    }

    free(temp);
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

static void fill_vpi_y_plane_data(VPIImageData* inData, VP_Frame* frame) {
    memset(inData, 0, sizeof(*inData));
    inData->bufferType = VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR;
    inData->buffer.pitch.format = VPI_IMAGE_FORMAT_Y8;
    inData->buffer.pitch.numPlanes = 1;
    uint8_t* y  = frame->data;

    inData->buffer.pitch.planes[0].data = y;
    inData->buffer.pitch.planes[0].pitchBytes = frame->stride;
    inData->buffer.pitch.planes[0].width = frame->width;
    inData->buffer.pitch.planes[0].height = frame->height;
}

static ProcStatus nv_stab_reset_context(NvStabCtx *c, int w, int h, int stride) {
    if (!c->dims_valid || c->width != w || c->height != h || c->stride != stride) {
        
        c->width = w;
        c->height = h;
        c->stride = stride;
        c->dims_valid = true;

        if (c->cur_img_y) {
            printf("[nv-stabilizer] prepare: before destroy\n");
            vpiImageDestroy(c->cur_img_y);  
            printf("[nv-stabilizer] prepare: after destroy\n");
            c->cur_img_y = NULL; 
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

    if(nv_stab_reset_context(c, w, h, stride) != PROC_STATUS_OK) {
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
    
    VPIImageData inData;
    fill_vpi_y_plane_data(&inData, frame);

    CHECK_STATUS(vpiImageCreateWrapper(&inData, NULL, 0, &c->in_wrap));

    printf("[nv-stabilizer] prepare: step3 ended\n");

    /* ------------------------------------------------------------------
     * 4. Copy caller frame into VPI-owned cur_img_y
     * ------------------------------------------------------------------ */
    CHECK_STATUS(nv_vpi_submit_copy(c->vpi_stream,
                            VPI_BACKEND_CPU,
                            c->in_wrap,
                            c->cur_img_y));
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

        c->has_prev = true;
    }

    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_ofa_init(NvStabCtx *c)
{
    const int w = c->width;
    const int h = c->height;

    // Formats used by OFA path (match sample structure)
    const VPIImageFormat fmt_pl = VPI_IMAGE_FORMAT_Y8;       // pitch-linear 8-bit
    const VPIImageFormat fmt_bl = VPI_IMAGE_FORMAT_Y8_ER_BL;    // block-linear 8-bit

    // Pyramid grid sizes: same grid size for each level
    std::vector<int32_t> pyrGridSize(c->num_levels, c->grid_size);

    // Create OFA payload: it operates on BL pyramids
    // FIXME: doesnt work with 640X480 because the num_levels is too high (but reducing doesn't solves this)
    CHECK_STATUS(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA, w, h, fmt_bl,
                                   pyrGridSize.data(), pyrGridSize.size(),
                                   c->quality, &c->ofa_payload));

    // Create pyramids (PL then BL)
    CHECK_STATUS(vpiPyramidCreate(w, h, fmt_pl, c->num_levels, 0.5, 0, &c->prev_pyr_pl));
    CHECK_STATUS(vpiPyramidCreate(w, h, fmt_bl, c->num_levels, 0.5, 0, &c->prev_pyr_bl));
    
    CHECK_STATUS(vpiPyramidCreate(w, h, fmt_pl, c->num_levels, 0.5, 0, &c->cur_pyr_pl));
    CHECK_STATUS(vpiPyramidCreate(w, h, fmt_bl, c->num_levels, 0.5, 0, &c->cur_pyr_bl));

    // Motion vector image dimensions aligned by grid size
    const int mvW = (w + c->grid_size - 1) / c->grid_size;
    const int mvH = (h + c->grid_size - 1) / c->grid_size;

    CHECK_STATUS(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16_BL, 0, &c->mv_bl));
    CHECK_STATUS(vpiImageCreate(mvW, mvH, VPI_IMAGE_FORMAT_2S16, 0, &c->mv_pl));

    c->ofa_inited = true;
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_compute_flow(NvStabCtx *c)
{
    if (!c || !c->prev_img_y || !c->cur_img_y) return PROC_STATUS_ERR_GENERAL;

    if (!c->ofa_inited) {
        ProcStatus ps = nv_stab_ofa_init(c);
        if (ps != PROC_STATUS_OK) return ps;
    }

    // 1) Build PL pyramids from PL Y8 images (CUDA is typical; CPU also works but slower)
    CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(c->vpi_stream, VPI_BACKEND_CUDA,
                                          c->prev_img_y, c->prev_pyr_pl,
                                          VPI_BORDER_CLAMP));

    CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(c->vpi_stream, VPI_BACKEND_CUDA,
                                          c->cur_img_y, c->cur_pyr_pl,
                                          VPI_BORDER_CLAMP));

    // 2) Convert pyramid PL -> BL using VIC (required for OFA input)
    CHECK_STATUS(vpiSubmitConvertImageFormatPyramid(c->vpi_stream, VPI_BACKEND_VIC,
                                            c->prev_pyr_pl, c->prev_pyr_bl, NULL));

    CHECK_STATUS(vpiSubmitConvertImageFormatPyramid(c->vpi_stream, VPI_BACKEND_VIC,
                                            c->cur_pyr_pl, c->cur_pyr_bl, NULL));

    // 3) OFA dense optical flow on BL pyramids -> motion vectors (BL)
    CHECK_STATUS(vpiSubmitOpticalFlowDensePyramid(c->vpi_stream, VPI_BACKEND_OFA,
                                          c->ofa_payload,
                                          c->prev_pyr_bl, c->cur_pyr_bl,
                                          c->mv_bl));

    // 4) Convert motion vectors BL -> PL so CPU can read them later (Function 3/4)
    CHECK_STATUS(vpiSubmitConvertImageFormat(c->vpi_stream, VPI_BACKEND_VIC,
                                     c->mv_bl, c->mv_pl, NULL));

    // 5) Sync (you need results to estimate transform; also ensures no lifetime issues)
    CHECK_STATUS(vpiStreamSync(c->vpi_stream));

    return PROC_STATUS_OK;
}


static ProcStatus nv_stab_compute_transform(NvStabCtx* c)
{
    if (!c->mv_pl) return PROC_STATUS_ERR_GENERAL;

    // Lock the motion vectors
    VPIImageData mvData;
    CHECK_STATUS(vpiImageLockData(c->mv_pl, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mvData));

    int mvW = (c->width + c->grid_size - 1) / c->grid_size;
    int mvH = (c->height + c->grid_size - 1) / c->grid_size;

    int16_t* mvPtr = (int16_t*)mvData.buffer.pitch.planes[0].data;
    int pitch = mvData.buffer.pitch.planes[0].pitchBytes / sizeof(int16_t); // mvW * 2

    std::vector<int> dxs, dys;
    for (int y = 0; y < mvH; y++) {
        for (int x = 0; x < mvW; x++) {
            int idx = y * pitch + x * 2;
            int16_t dx = mvPtr[idx];
            int16_t dy = mvPtr[idx + 1];
            dxs.push_back(dx);
            dys.push_back(dy);
        }
    }

    vpiImageUnlock(c->mv_pl);

    // Compute median for robustness
    if (!dxs.empty()) {
        std::sort(dxs.begin(), dxs.end());
        std::sort(dys.begin(), dys.end());
        int med_dx = dxs[dxs.size() / 2];
        int med_dy = dys[dys.size() / 2];
        // Apply inverse for stabilization
        c->estimated_dx = -med_dx;
        c->estimated_dy = -med_dy;
    } else {
        c->estimated_dx = 0;
        c->estimated_dy = 0;
    }

    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_apply_transform(NvStabCtx* c, VP_Frame* frame)
{
    // Skip transformation for first frame or zero motion
    if (!c->has_prev || (c->estimated_dx == 0 && c->estimated_dy == 0)) 
        return PROC_STATUS_OK;

    printf("[nv-stabilizer] apply_transform: dx=%d dy=%d\n", c->estimated_dx, c->estimated_dy);

    // For now, skip the actual shifting to avoid memory issues with NVMM
    // The optical flow computation is working, but applying the shift
    // to mapped NVMM memory causes allocation issues
    // TODO: Implement GPU-based warping using VPI instead of CPU shifting
    
    (void)frame; // Suppress unused warning
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
    printf("[nv-stabilizer] process: context params: w:%d h:%d\n",
           c->width, c->height);
    st = nv_stab_compute_flow(c);
    if (st != PROC_STATUS_OK)
        return st;

    st = nv_stab_compute_transform(c);
    if (st != PROC_STATUS_OK)
        return st;

    st = nv_stab_apply_transform(c, frame);
    if (st != PROC_STATUS_OK)
        return st;

    // Update for next frame: swap prev and cur
    if (c->has_prev) {
        VPIImage tmp = c->prev_img_y;
        c->prev_img_y = c->cur_img_y;
        c->cur_img_y = tmp;
    } else {
        // Copy cur to prev for first frame
        CHECK_STATUS(nv_vpi_submit_copy(c->vpi_stream, VPI_BACKEND_CPU, c->cur_img_y, c->prev_img_y));
        CHECK_STATUS(vpiStreamSync(c->vpi_stream));
        c->has_prev = true;
    }

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
    if (c->cur_img_y)  vpiImageDestroy(c->cur_img_y);
    if (c->prev_img_y) vpiImageDestroy(c->prev_img_y);
    if (c->in_wrap) vpiImageDestroy(c->in_wrap);
    if (c->vpi_stream)  vpiStreamDestroy(c->vpi_stream);

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
