#include "processor_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr static int max_history_size = 10;
typedef struct {
    int dx[max_history_size];
    int dy[max_history_size];
    int curr_index;
} BasicStabHistory;
typedef struct {
    unsigned long frame_count;
    const char* cfg_path_seen;
    BasicStabHistory history;
    int history_index;
    int prev_cx, prev_cy;
    bool has_prev;
} BasicStabCtx;

static ProcStatus basic_stab_init(const char* config_path, void** ctx)
{
    BasicStabCtx* c = (BasicStabCtx*)malloc(sizeof(BasicStabCtx));
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


static void apply_shift_rgb(
        const unsigned char* src,
        unsigned char* dst,
        int w, int h, int stride,
        int dx, int dy)
{
    printf("[basic-stabilizer] apply_shift_rgb: dx=%d dy=%d\n", dx, dy);
    for (int y = 0; y < h; ++y) {
        int sy = y - dy;              // shifted source Y
        if (sy < 0 || sy >= h) {
            // Entire row out of bounds → fill with black
            memset(dst + y * stride, 0, stride);
            continue;
        }

        for (int x = 0; x < w; ++x) {
            int sx = x - dx;          // shifted source X
            unsigned char* pix_dst = dst + (y * stride + x * 3);

            if (sx < 0 || sx >= w) {
                // Fill with black when out of bounds
                pix_dst[0] = pix_dst[1] = pix_dst[2] = 0;
                continue;
            }

            const unsigned char* pix_src = src + (sy * stride + sx * 3);
            pix_dst[0] = pix_src[0];
            pix_dst[1] = pix_src[1];
            pix_dst[2] = pix_src[2];
        }
    }
}

static void apply_shift(ProcPixelFormat pixfmt, const unsigned char* src,
        unsigned char* dst, int w, int h, int stride, int dx, int dy) {
            printf("[basic-stabilizer] apply_shift: pixfmt=%d dx=%d dy=%d\n",
                pixfmt, dx, dy);
            switch (pixfmt) {
                case ProcPixelFormat::PROC_PIXFMT_RGB:
                    apply_shift_rgb(src, dst, w, h, stride, dx, dy);
                    break;
                // Add cases for other pixel formats as needed
                default:
                    fprintf(stderr, "[basic-stabilizer] apply_shift: unsupported pixel format\n");
                    break;
            }
        }

        
        static void basic_stab_destroy(void* vctx)
{
    BasicStabCtx* c = (BasicStabCtx*)vctx;
    if (!c)
        return;

    fprintf(stderr,
        "[basic-stabilizer] destroy: total_frames=%lu\n",
        c->frame_count);

        free(c);
    }
    
    static ProcStatus basic_stab_process(void* vctx,
                                         const VP_FrameIn* in,
                                         VP_FrameOut* out)
    {
        BasicStabCtx* c = (BasicStabCtx*)vctx;
        if (!c || !in || !out)
        return PROC_STATUS_ERR_GENERAL;
        
        c->frame_count++;
        printf("[basic-stabilizer] process: frame=%lu\n", c->frame_count);
    
        // Simple centroid-based motion estimation
        int cx = 0, cy = 0;
        int w = in->width;
        int h = in->height;
        int stride = in->stride;
        printf("[basic-stabilizer] process: frame size w=%d h=%d stride=%d\n", w, h, stride);
        const unsigned char* data = (const unsigned char*)in->data;
        unsigned long sum_x = 0, sum_y = 0, sum_val = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                unsigned char val = data[y * stride + x];
                sum_x += x * val;
                sum_y += y * val;
                sum_val += val;
            }
        }
        if (sum_val > 0) {
            cx = sum_x / sum_val;
            cy = sum_y / sum_val;
        }
    
        int dx = 0, dy = 0;
        if (c->has_prev) {
            dx = cx - c->prev_cx;
            dy = cy - c->prev_cy;
            c->history.dx[c->history.curr_index] = dx;
            c->history.dy[c->history.curr_index] = dy;
            c->history.curr_index = (c->history.curr_index + 1) % max_history_size;
            printf("[basic-stabilizer] process: history updated current_index=%d\n", c->history.curr_index);
        }
        else {
            c->has_prev = true;
        }
        printf("[basic-stabilizer] process: estimated motion dx=%d dy=%d\n", dx, dy);
        c->prev_cx = cx;
        c->prev_cy = cy;
        
        printf("[basic-stabilizer] process: motion dx=%d dy=%d\n", dx, dy);
    
        // Smooth motion (moving average)
        printf("[basic-stabilizer] process: smoothing\n");
        int avg_dx = 0, avg_dy = 0;
        for (int v : c->history.dx) avg_dx += v;
        for (int v : c->history.dy) avg_dy += v;
        if(c->frame_count < max_history_size) {
            avg_dx /= (int)c->frame_count;
            avg_dy /= (int)c->frame_count;
        }  
        else{
            avg_dx /= max_history_size;
            avg_dy /= max_history_size;
        }
    
        // Apply stabilization: shift/crop the frame
        // For demonstration, just copy input to output and log the shift
        printf("[basic-stabilizer] process: applying stabilization\n");
        memcpy(out->data, in->data, in->stride * h); // assumes single channel, same size
        printf("[basic-stabilizer] process: frame=%lu dx=%d dy=%d avg_dx=%d avg_dy=%d\n",
               c->frame_count, dx, dy, avg_dx, avg_dy);
        out->width = w;
        out->height = h;
        // out->pixfmt = in->pixfmt; // Removed: struct does not have pixfmt
    
        if (c->frame_count == 1 || (c->frame_count % 30) == 0) {
            fprintf(stderr,
                "[basic-stabilizer] process: frame=%lu dx=%d dy=%d avg_dx=%d avg_dy=%d\n",
                c->frame_count, dx, dy, avg_dx, avg_dy);
        }
    
        apply_shift(in->pixfmt,
            in->data, out->data,
            w, h, stride,
            avg_dx, avg_dy);
        return PROC_STATUS_OK;
    }

    ProcStatus proc_register(ProcessorAPI* api)
    {
        if (!api)
        return PROC_STATUS_ERR_GENERAL;

    api->init   = basic_stab_init;
    api->process = basic_stab_process;
    api->destroy = basic_stab_destroy;

    return PROC_STATUS_OK;
}
