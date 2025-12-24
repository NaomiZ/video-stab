#include "processor_api.h"
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <vpi/VPI.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/PerspectiveWarp.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/KLTFeatureTracker.h>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>

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

#define MOTION_HISTORY_SIZE 15

typedef struct {
    uint64_t frame_count;
    const char* cfg_path_seen;

    int width;
    int height;
    int stride;
    bool dims_valid;

    VPIStream vpi_stream;

    // VPI-owned images (Y and UV planes)
    VPIImage cur_img_y;
    VPIImage prev_img_y;
    VPIImage out_img_y;
    
    VPIImage cur_img_uv;
    VPIImage prev_img_uv;
    VPIImage out_img_uv;

    // Harris corner detection
    VPIPayload harris_payload;
    VPIArray keypoints_cur;
    VPIArray keypoints_prev;
    VPIHarrisCornerDetectorParams harris_params;

    // KLT feature tracking
    VPIPayload klt_payload;
    VPIArray tracked_features;
    VPIArray tracking_estimates;
    VPIKLTFeatureTrackerParams klt_params;

    // Motion estimation
    float affine_matrix[6];        // [a b tx c d ty]
    float smoothed_affine[6];      // Temporally filtered

    // Motion history for smoothing
    float motion_history[MOTION_HISTORY_SIZE][6];
    int history_index;
    bool history_full;

    // State
    bool has_prev_features;
    int num_tracked_points;
    int redetect_counter;

} NvStabCtx;

static ProcStatus nv_stab_init(const char* config_path, void** ctx)
{
    NvStabCtx* c = (NvStabCtx*)malloc(sizeof(NvStabCtx));
    if (!c)
        return PROC_STATUS_ERR_ALLOC;

    memset(c, 0, sizeof(NvStabCtx));
    
    c->frame_count = 0;
    c->cfg_path_seen = config_path ? strdup(config_path) : nullptr;
    c->dims_valid = false;
    c->has_prev_features = false;
    c->num_tracked_points = 0;
    c->history_index = 0;
    c->history_full = false;
    c->redetect_counter = 0;

    VPIStatus st = vpiStreamCreate(0, &c->vpi_stream);
    if (st != VPI_SUCCESS) {
        printf("[nv-stabilizer] vpiStreamCreate failed: %d\n", (int)st);
        free(c);
        return PROC_STATUS_ERR_GENERAL;
    }

    // Harris corner detection parameters
    c->harris_params.strengthThresh = 0.05f;
    c->harris_params.sensitivity = 0.08f;
    c->harris_params.minNMSDistance = 8;

    // KLT tracking parameters
    c->klt_params.numberOfIterationsScaling = 2;
    c->klt_params.nccThresholdUpdate = 0.7f;
    c->klt_params.trackingType = VPI_KLT_INVERSE_COMPOSITIONAL;

    // Initialize identity affine transform
    c->affine_matrix[0] = 1.0f; c->affine_matrix[1] = 0.0f; c->affine_matrix[2] = 0.0f;
    c->affine_matrix[3] = 0.0f; c->affine_matrix[4] = 1.0f; c->affine_matrix[5] = 0.0f;
    
    memcpy(c->smoothed_affine, c->affine_matrix, sizeof(c->affine_matrix));

    *ctx = c;
    printf("[nv-stabilizer] Initialized with Harris+KLT feature tracking\n");
    return PROC_STATUS_OK;
}

static inline VPIStatus
nv_vpi_submit_copy(VPIStream stream, VPIBackend backend, VPIImage src, VPIImage dst)
{
    return vpiSubmitConvertImageFormat(stream, backend, src, dst, NULL);
}

static void fill_vpi_y_plane_data(VPIImageData* inData, VP_Frame* frame) {
    memset(inData, 0, sizeof(*inData));
    inData->bufferType = VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR;
    inData->buffer.pitch.format = VPI_IMAGE_FORMAT_Y8;
    inData->buffer.pitch.numPlanes = 1;

    inData->buffer.pitch.planes[0].data = frame->data;
    inData->buffer.pitch.planes[0].pitchBytes = frame->stride;
    inData->buffer.pitch.planes[0].width = frame->width;
    inData->buffer.pitch.planes[0].height = frame->height;
}

static void fill_vpi_uv_plane_data(VPIImageData* uvData, VP_Frame* frame) {
    memset(uvData, 0, sizeof(*uvData));
    uvData->bufferType = VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR;
    uvData->buffer.pitch.format = VPI_IMAGE_FORMAT_Y8_ER;  // Treat UV as raw Y8 bytes
    uvData->buffer.pitch.numPlanes = 1;

    // UV starts after Y plane
    uint8_t* uv_start = frame->data + (frame->width * frame->height);
    
    uvData->buffer.pitch.planes[0].data = uv_start;
    uvData->buffer.pitch.planes[0].pitchBytes = frame->stride;  // Same stride
    uvData->buffer.pitch.planes[0].width = frame->width;
    uvData->buffer.pitch.planes[0].height = frame->height / 2;  // Half height for NV12 UV plane
}

static ProcStatus nv_stab_reset_context(NvStabCtx *c, int w, int h, int stride) {
    printf("[nv-stabilizer] reset_context check: dims_valid=%d, width=%d/%d, height=%d/%d, stride=%d/%d\n",
           c->dims_valid, c->width, w, c->height, h, c->stride, stride);
    
    if (!c->dims_valid || c->width != w || c->height != h || c->stride != stride) {
        printf("[nv-stabilizer] Resetting context for new dimensions\n");
        
        c->width = w;
        c->height = h;
        c->stride = stride;
        c->dims_valid = true;

        // Destroy old Y plane images
        if (c->cur_img_y) vpiImageDestroy(c->cur_img_y);
        if (c->prev_img_y) vpiImageDestroy(c->prev_img_y);
        if (c->out_img_y) vpiImageDestroy(c->out_img_y);
        
        // Destroy old UV plane images
        if (c->cur_img_uv) vpiImageDestroy(c->cur_img_uv);
        if (c->prev_img_uv) vpiImageDestroy(c->prev_img_uv);
        if (c->out_img_uv) vpiImageDestroy(c->out_img_uv);

        // Create Y plane images
        printf("[nv-stabilizer] Creating Y images %dx%d\n", w, h);
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8_ER, 0, &c->cur_img_y));
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8_ER, 0, &c->prev_img_y));
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8_ER, 0, &c->out_img_y));
        printf("[nv-stabilizer] Y images created: cur=%p prev=%p out=%p\n", 
               c->cur_img_y, c->prev_img_y, c->out_img_y);

        // Create UV plane images - treat as Y8 for raw interleaved UV data
        // UV plane is (w x h/2) of interleaved U,V bytes
        printf("[nv-stabilizer] Creating UV images %dx%d\n", w, h/2);
        CHECK_STATUS(vpiImageCreate(w, h/2, VPI_IMAGE_FORMAT_Y8_ER, 0, &c->cur_img_uv));
        CHECK_STATUS(vpiImageCreate(w, h/2, VPI_IMAGE_FORMAT_Y8_ER, 0, &c->prev_img_uv));
        CHECK_STATUS(vpiImageCreate(w, h/2, VPI_IMAGE_FORMAT_Y8_ER, 0, &c->out_img_uv));
        printf("[nv-stabilizer] UV images created: cur=%p prev=%p out=%p\n",
               c->cur_img_uv, c->prev_img_uv, c->out_img_uv);

        // Destroy old Harris/KLT resources
        if (c->harris_payload) vpiPayloadDestroy(c->harris_payload);
        if (c->klt_payload) vpiPayloadDestroy(c->klt_payload);
        if (c->keypoints_cur) vpiArrayDestroy(c->keypoints_cur);
        if (c->keypoints_prev) vpiArrayDestroy(c->keypoints_prev);
        if (c->tracked_features) vpiArrayDestroy(c->tracked_features);
        if (c->tracking_estimates) vpiArrayDestroy(c->tracking_estimates);

        // Create Harris corner detector
        printf("[nv-stabilizer] Creating Harris detector %dx%d\n", w, h);
        fflush(stdout);
        CHECK_STATUS(vpiCreateHarrisCornerDetector(VPI_BACKEND_CUDA, w, h, &c->harris_payload));
        printf("[nv-stabilizer] Harris created: %p\n", c->harris_payload);
        fflush(stdout);

        // Create KLT feature tracker
        CHECK_STATUS(vpiCreateKLTFeatureTracker(VPI_BACKEND_CUDA, w, h,
                                                 VPI_IMAGE_FORMAT_Y8_ER, 0, &c->klt_payload));

        // Create keypoint arrays
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &c->keypoints_cur));
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &c->keypoints_prev));
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &c->tracked_features));
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &c->tracking_estimates));

        c->has_prev_features = false;
        printf("[nv-stabilizer] Context reset: %dx%d, Harris+KLT initialized\n", w, h);
    }
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_prepare_frame(NvStabCtx* c, VP_Frame* frame)
{
    if (!c || !frame) return PROC_STATUS_ERR_GENERAL;
    if (frame->pixfmt != PROC_PIXFMT_NV12) return PROC_STATUS_ERR_UNSUPPORTED;
    if (!frame->data || frame->width <= 0 || frame->height <= 0 || frame->stride <= 0)
        return PROC_STATUS_ERR_GENERAL;

    const int w = frame->width;
    const int h = frame->height;
    const int stride = frame->stride;

    if(nv_stab_reset_context(c, w, h, stride) != PROC_STATUS_OK)
        return PROC_STATUS_ERR_GENERAL;

    printf("[nv-stabilizer] prepare_frame: w=%d h=%d stride=%d, cur_img_y=%p\n", w, h, stride, c->cur_img_y);

    // Copy Y plane data
    printf("[nv-stabilizer] Copying Y plane\n");
    fflush(stdout);
    {
        VPIImageData imgData;
        CHECK_STATUS(vpiImageLockData(c->cur_img_y, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData));
        
        uint8_t* src = (uint8_t*)frame->data;
        uint8_t* dst = (uint8_t*)imgData.buffer.pitch.planes[0].data;
        int y_size = frame->width * frame->height;
        memcpy(dst, src, y_size);
        
        vpiImageUnlock(c->cur_img_y);
    }
    printf("[nv-stabilizer] Y plane copied\n");
    fflush(stdout);

    // Copy UV plane data - DISABLED for now due to buffer access issues
    // TODO: Investigate proper VPI UV plane buffer access
    /*
    {
        VPIImageData imgData;
        CHECK_STATUS(vpiImageLockData(c->cur_img_uv, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData));
        
        uint8_t* src = (uint8_t*)frame->data + (frame->width * frame->height);
        uint8_t* dst = (uint8_t*)imgData.buffer.pitch.planes[0].data;
        int uv_size = (frame->width * frame->height) / 2;  // UV plane is half the Y plane size
        memcpy(dst, src, uv_size);
        
        vpiImageUnlock(c->cur_img_uv);
    }
    */

    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_detect_features(NvStabCtx *c)
{
    // DISABLED Harris detection for debugging
    printf("[nv-stabilizer] detect_features SKIPPED (Harris disabled for debugging)\n");
    return PROC_STATUS_OK;
}

/*
static ProcStatus nv_stab_detect_features_ORIG(NvStabCtx *c)
{
    VPIStream s = c->vpi_stream;
    VPIPayload p = c->harris_payload;
    VPIImage img = c->cur_img_y;
    VPIArray arr = c->keypoints_cur;
    VPIHarrisCornerDetectorParams* params = &c->harris_params;
    
    printf("[nv-stabilizer] detect_features: stream=%p, payload=%p, image=%p, array=%p, params=%p\n",
           s, p, img, arr, params);
    fflush(stdout);
    
    // Run Harris corner detector on current Y image (scores output can be NULL)
    printf("[nv-stabilizer] Calling vpiSubmitHarrisCornerDetector\n");
    fflush(stdout);
    
    CHECK_STATUS(vpiSubmitHarrisCornerDetector(s, VPI_BACKEND_CUDA, p, img, arr, NULL, params));
    
    CHECK_STATUS(vpiStreamSync(c->vpi_stream));

    // Check how many features were detected
    VPIArrayData kpData;
    CHECK_STATUS(vpiArrayLockData(c->keypoints_cur, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &kpData));
    int num_features = *kpData.buffer.aos.sizePointer;
    vpiArrayUnlock(c->keypoints_cur);

    printf("[nv-stabilizer] Harris detected %d features\n", num_features);
    
    if (num_features < 20) {
        printf("[nv-stabilizer] WARNING: Too few features detected (%d < 20)\n", num_features);
    }

    return PROC_STATUS_OK;
}
*/

static ProcStatus nv_stab_track_features(NvStabCtx *c)
{
    // DISABLED KLT tracking for debugging - just fake it
    c->num_tracked_points = 5;  // Fake count to allow warp
    c->has_prev_features = true;
    printf("[nv-stabilizer] KLT tracking SKIPPED (disabled for debugging)\n");
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_estimate_motion(NvStabCtx* c)
{
    if (c->num_tracked_points < 3) {
        // Not enough points for affine estimation, use identity
        c->affine_matrix[0] = 1.0f; c->affine_matrix[1] = 0.0f; c->affine_matrix[2] = 0.0f;
        c->affine_matrix[3] = 0.0f; c->affine_matrix[4] = 1.0f; c->affine_matrix[5] = 0.0f;
        return PROC_STATUS_OK;
    }

    // Lock tracked features to get correspondences
    VPIArrayData prevData, curData;
    CHECK_STATUS(vpiArrayLockData(c->keypoints_prev, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &prevData));
    CHECK_STATUS(vpiArrayLockData(c->tracked_features, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &curData));

    VPIKeypointF32* prevPts = (VPIKeypointF32*)prevData.buffer.aos.data;
    VPIKeypointF32* curPts = (VPIKeypointF32*)curData.buffer.aos.data;
    int numPts = std::min(*prevData.buffer.aos.sizePointer, c->num_tracked_points);

    // Simple median-based translation estimation (robust to outliers)
    std::vector<float> dxs, dys;
    for (int i = 0; i < numPts; i++) {
        dxs.push_back(curPts[i].x - prevPts[i].x);
        dys.push_back(curPts[i].y - prevPts[i].y);
    }

    vpiArrayUnlock(c->tracked_features);
    vpiArrayUnlock(c->keypoints_prev);

    if (dxs.empty()) {
        c->affine_matrix[0] = 1.0f; c->affine_matrix[1] = 0.0f; c->affine_matrix[2] = 0.0f;
        c->affine_matrix[3] = 0.0f; c->affine_matrix[4] = 1.0f; c->affine_matrix[5] = 0.0f;
        return PROC_STATUS_OK;
    }

    std::sort(dxs.begin(), dxs.end());
    std::sort(dys.begin(), dys.end());
    float med_dx = dxs[dxs.size() / 2];
    float med_dy = dys[dys.size() / 2];

    // Build affine transform (translation only for now)
    // Inverse for stabilization: move frame opposite to camera motion
    c->affine_matrix[0] = 1.0f;
    c->affine_matrix[1] = 0.0f;
    c->affine_matrix[2] = -med_dx;  // tx (inverse)
    c->affine_matrix[3] = 0.0f;
    c->affine_matrix[4] = 1.0f;
    c->affine_matrix[5] = -med_dy;  // ty (inverse)

    printf("[nv-stabilizer] Motion: dx=%.2f dy=%.2f (from %d points)\n", 
           med_dx, med_dy, numPts);

    return PROC_STATUS_OK;
}

static void nv_stab_smooth_motion(NvStabCtx* c)
{
    // Add current motion to history
    memcpy(c->motion_history[c->history_index], c->affine_matrix, sizeof(c->affine_matrix));
    c->history_index = (c->history_index + 1) % MOTION_HISTORY_SIZE;
    if (c->history_index == 0) c->history_full = true;

    // Compute exponential moving average
    const float alpha = 0.3f;  // Smoothing factor (0 = max smooth, 1 = no smooth)
    
    if (!c->history_full && c->history_index == 0) {
        // First frame, no smoothing yet
        memcpy(c->smoothed_affine, c->affine_matrix, sizeof(c->affine_matrix));
    } else {
        // Smooth each component
        for (int i = 0; i < 6; i++) {
            c->smoothed_affine[i] = alpha * c->affine_matrix[i] + 
                                   (1.0f - alpha) * c->smoothed_affine[i];
        }
    }

    printf("[nv-stabilizer] Smoothed: tx=%.2f ty=%.2f\n", 
           c->smoothed_affine[2], c->smoothed_affine[5]);
}

static ProcStatus nv_stab_apply_stabilization(NvStabCtx* c, VP_Frame* frame)
{
    // DISABLED perspective warp for debugging - just copy Y plane back unchanged
    // Copy Y plane data back to frame
    {
        VPIImageData imgData;
        CHECK_STATUS(vpiImageLockData(c->cur_img_y, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData));
        
        uint8_t* src = (uint8_t*)imgData.buffer.pitch.planes[0].data;
        uint8_t* dst = (uint8_t*)frame->data;
        int y_size = frame->width * frame->height;
        memcpy(dst, src, y_size);
        
        vpiImageUnlock(c->cur_img_y);
    }
    
    printf("[nv-stabilizer] apply_stabilization SKIPPED (warp disabled for debugging)\n");
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_process(void* vctx, VP_Frame* frame)
{
    NvStabCtx* c = (NvStabCtx*)vctx;
    if (!c || !frame)
        return PROC_STATUS_ERR_GENERAL;

    c->frame_count++;
    printf("[nv-stabilizer] === Frame %lu === Starting process\n", c->frame_count);
    fflush(stdout);

    ProcStatus st;

    // 1. Prepare frame: copy Y and UV to VPI-owned images
    printf("[nv-stabilizer] Calling prepare_frame\n");
    fflush(stdout);
    st = nv_stab_prepare_frame(c, frame);
    printf("[nv-stabilizer] prepare_frame returned: %d\n", st);
    fflush(stdout);
    if (st != PROC_STATUS_OK) return st;

    // 2. Detect or track features
    printf("[nv-stabilizer] Checking redetect: frame_count=%lu, redetect_counter=%d, num_tracked_points=%d\n",
           c->frame_count, c->redetect_counter, c->num_tracked_points);
    fflush(stdout);
    
    if (c->frame_count == 1 || c->redetect_counter >= 30 || c->num_tracked_points < 20) {
        printf("[nv-stabilizer] === Calling detect_features ===\n");
        fflush(stdout);
        st = nv_stab_detect_features(c);
        if (st != PROC_STATUS_OK) return st;
        c->redetect_counter = 0;
    }

    st = nv_stab_track_features(c);
    if (st != PROC_STATUS_OK) return st;
    c->redetect_counter++;

    // 3. Estimate affine motion
    st = nv_stab_estimate_motion(c);
    if (st != PROC_STATUS_OK) return st;

    // 4. Smooth motion trajectory
    nv_stab_smooth_motion(c);

    // 5. Apply stabilization warp to Y and UV planes
    if (c->has_prev_features && c->num_tracked_points >= 3) {
        st = nv_stab_apply_stabilization(c, frame);
        if (st != PROC_STATUS_OK) return st;
    }

    // 6. Update for next frame: swap images
    VPIImage tmp_y = c->prev_img_y;
    c->prev_img_y = c->cur_img_y;
    c->cur_img_y = tmp_y;

    VPIImage tmp_uv = c->prev_img_uv;
    c->prev_img_uv = c->cur_img_uv;
    c->cur_img_uv = tmp_uv;

    // Swap keypoints
    VPIArray tmp_kp = c->keypoints_prev;
    c->keypoints_prev = c->keypoints_cur;
    c->keypoints_cur = tmp_kp;

    return PROC_STATUS_OK;
}

static void nv_stab_destroy(void* vctx)
{
    NvStabCtx* c = (NvStabCtx*)vctx;
    if (!c) return;

    printf("[nv-stabilizer] Destroy: processed %lu frames\n", c->frame_count);

    // Destroy Y plane images
    if (c->cur_img_y) vpiImageDestroy(c->cur_img_y);
    if (c->prev_img_y) vpiImageDestroy(c->prev_img_y);
    if (c->out_img_y) vpiImageDestroy(c->out_img_y);

    // Destroy UV plane images
    if (c->cur_img_uv) vpiImageDestroy(c->cur_img_uv);
    if (c->prev_img_uv) vpiImageDestroy(c->prev_img_uv);
    if (c->out_img_uv) vpiImageDestroy(c->out_img_uv);

    // Destroy Harris/KLT resources
    if (c->harris_payload) vpiPayloadDestroy(c->harris_payload);
    if (c->klt_payload) vpiPayloadDestroy(c->klt_payload);
    if (c->keypoints_cur) vpiArrayDestroy(c->keypoints_cur);
    if (c->keypoints_prev) vpiArrayDestroy(c->keypoints_prev);
    if (c->tracked_features) vpiArrayDestroy(c->tracked_features);
    if (c->tracking_estimates) vpiArrayDestroy(c->tracking_estimates);

    if (c->vpi_stream) vpiStreamDestroy(c->vpi_stream);
    if (c->cfg_path_seen) free((void*)c->cfg_path_seen);

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
