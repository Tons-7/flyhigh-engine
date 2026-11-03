/* FlyHigh Engine: C ABI (crates/flyhigh-capi, docs/api-c.md). Link against flyhigh (.dll / .so / .dylib / .a).
 *
 * The engine's layers behind opaque handles:
 *   FhEngine     the device (fh_engine_*): one per GPU, sessions and streams share it
 *   FhModel      weights and graph (fh_model_*): CPU data, several may live beside one engine
 *   FhSession    the plans for one frame size (fh_session_*): two frames in, one frame at phase t out
 *   FhStream     frames with timestamps in, intervals out (fh_stream_*): the ring, the pair classifier, cadence
 *   FhPacer      what to present on this display tick and what to generate (fh_pacer_*): pure logic, no clock
 *   FhGenerator  the CPU convenience (fh_generator_*): source frames in, the frames to display out, in order
 * Frames cross this ABI as RGBA8 bytes (width * height * 4); GPU textures come in through the interop calls
 * (fh_session_import_d3d11, fh_stream_import_d3d11).
 * Single-threaded use per handle. Errors: a call returns NULL or -1 and fh_last_error() holds the message.
 * Options: always start from the fh_*_options_default() struct and change what you need.
 */
#ifndef FLYHIGH_RUNTIME_H
#define FLYHIGH_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct FhEngine FhEngine;
typedef struct FhModel FhModel;
typedef struct FhSession FhSession;
typedef struct FhStream FhStream;
typedef struct FhPacer FhPacer;
typedef struct FhGenerator FhGenerator;

/* message of the last failed call on this thread (empty string if none) */
const char*  fh_last_error(void);

/* ---- Engine: the device ---- */
typedef struct FhEngineOptions {
    uint32_t backend;           /* 0 automatic (default), 1 Vulkan, 2 DX12, 3 Metal, 4 OpenGL */
    uint32_t tensor_cores;      /* 1 = cooperative-matrix kernels where the GPU has them (default), 0 = portable only */
    uint32_t autotune;          /* 1 = time conv kernels per shape when a session is built (default), 0 = fixed */
    const char* adapter;        /* NULL = the high-performance GPU (default); "#1", "10de:2803", "4060", "low-power" */
    uint32_t precision;         /* 0 automatic (default: f16 where the GPU has shader-f16, else f32), 1 f16, 2 f32 */
    const char* tune_cache;     /* NULL (default) or a directory: autotune results persist there, timed once per GPU / driver */
    uint32_t compact_memory;    /* 1 = small allocator blocks (default): ~190 MB reserved at 1080p instead of 512 MB */
    const char* dxc_path;       /* Windows DX12: directory or path of dxcompiler.dll (dxil.dll beside it). NULL (default) =
                                   beside this library, then the DLL search path; without it FXC: slow compiles, no f16 */
} FhEngineOptions;

FhEngineOptions fh_engine_options_default(void);
FhEngine*    fh_engine_create(const FhEngineOptions* options);               /* NULL options = the defaults */
void         fh_engine_destroy(FhEngine* engine);                             /* after its sessions and streams */
/* JSON {"name","class":"tc-f16"|"portable-f16"|"portable-f32","tensor_cores","timestamps","bgra_storage",
 *       "shader_compiler":"dxc"|"fxc"|null (DX12 only)}; bytes or -1 */
int          fh_engine_info(const FhEngine* engine, char* buf, size_t cap);
/* JSON: the device, kernels compiled so far, autotune entries, tensor bytes; bytes written or -1 */
int          fh_engine_report_json(const FhEngine* engine, char* buf, size_t cap);

/* ---- Model: weights and graph ---- */
FhModel*     fh_model_load(const char* dir);                /* graph.json, flyhighfg.json, flyhighfg.bin in dir */
FhModel*     fh_model_from_bytes(const uint8_t* graph_json, size_t graph_len, const uint8_t* manifest_json, size_t manifest_len,
                                 const uint8_t* weights, size_t weights_len);
void         fh_model_destroy(FhModel* model);
/* JSON {"name","train_step","graph_ops","tensors","render_sampling","format_version"}; bytes written or -1 */
int          fh_model_info(const FhModel* model, char* buf, size_t cap);

/* ---- Session: the plans for one frame size ---- */
typedef struct FhSessionOptions {
    uint32_t width, height;     /* frame size (>= 64 x 64) */
    uint32_t internal_w, internal_h;   /* network resolution; 0,0 = automatic (480-line rung above 960 px, else the frame) */
    uint32_t k;                 /* fine-flow iterations 1..3 (default 1) */
    uint32_t repair;            /* periodic-structure repair: 0 off, 1 safe (default), 2 aggressive (drawn content) */
    uint32_t render;            /* 0 = the model's sampler (default), 1 = bilinear, 2 = bicubic */
    uint32_t keep_static;       /* 1 = static protection: pixels unchanged for static_frames frames (HUD, logos) are kept */
    uint32_t static_frames;     /* unchanged source frames before a pixel is protected (default 8) */
    uint32_t static_tolerance;  /* largest change, 0..255 levels, still counted as unchanged (default 4) */
    uint32_t fast_render;       /* 1 = render from the source textures (same quality, cheaper per generated frame) */
    uint32_t linear_t;          /* 1 = with fast_render: the t-dependent network once per pair, phases scaled linearly */
    uint32_t keep_maps;         /* 1 = keep the repair maps after the pair (the Stream's cadence; raises repair to 1) */
    uint32_t protection_mask;   /* 1 = the host paints pixels never interpolated (fh_session_write_mask), with or without keep_static */
    const char* tune;           /* NULL or the autotune table text for this device and size */
} FhSessionOptions;

FhSessionOptions fh_session_options_default(uint32_t width, uint32_t height);
FhSession*   fh_session_create(const FhEngine* engine, const FhModel* model, const FhSessionOptions* options);
void         fh_session_destroy(FhSession* session);
int          fh_session_add_input(FhSession* session);                        /* the new slot's index, or -1 */
int          fh_session_add_output(FhSession* session);
int          fh_session_write_rgba(const FhSession* session, uint32_t slot, const uint8_t* rgba, size_t len);
int          fh_session_write_mask(const FhSession* session, const uint8_t* mask, size_t len);   /* width * height bytes, 0 interpolates; from the next pair */
/* waits for the GPU; bytes written (width * height * 4) or -1 */
int          fh_session_read_rgba(const FhSession* session, uint32_t slot, uint8_t* buf, size_t cap);
int          fh_session_pair(const FhSession* session, uint32_t a, uint32_t b);    /* the motion of the interval (a, b) */
int          fh_session_mid(const FhSession* session, float t, uint32_t out);      /* phase t of the last pair into out */
int          fh_session_generate(const FhSession* session, uint32_t a, uint32_t b, float t, uint32_t out);
/* GPU time in ms: 1 = measured (ms filled), 0 = no timestamp queries on this device, -1 = error */
int          fh_session_pair_timed(const FhSession* session, uint32_t a, uint32_t b, double* ms);
int          fh_session_mid_timed(const FhSession* session, float t, uint32_t out, double* ms);
int          fh_session_warm_up(FhSession* session);                          /* compile every kernel before the first frame */
/* Windows: a D3D11 texture shared through an NT handle (D3D11_RESOURCE_MISC_SHARED_NTHANDLE) becomes an input slot of the
 * session without a copy (Vulkan engine with VK_KHR_external_memory_win32); the slot, or -1 (elsewhere always -1) */
int          fh_session_import_d3d11(FhSession* session, const FhEngine* engine, intptr_t handle, uint32_t width, uint32_t height);
int          fh_session_report_json(const FhSession* session, char* buf, size_t cap);

/* ---- Stream: frames with timestamps in, intervals out (ARCHITECTURE.md) ---- */
typedef struct FhStreamOptions {
    uint32_t mode;              /* 0 realtime (default), 1 video (one frame of lookahead) */
    uint32_t cadence;           /* video: 1 = re-time elements animated on 2s (anime) */
    double   scene_cut;         /* mean-difference threshold (0..1) above which a pair is a cut; 0 = never (default 0.08) */
    uint32_t held_frames;       /* 1 = detect repeated frames (default 0) */
    double   held_mean, held_max;   /* their thresholds; 0 = the defaults 0.0033 / 0.06 */
    uint32_t inputs, outputs;   /* slots created up front (default 4 / 4) */
} FhStreamOptions;

typedef struct FhClassification {
    uint32_t kind;              /* 0 motion, 1 scene cut, 2 held frame */
    float    mean, max;         /* mean and largest per-cell difference, 0..1 */
    uint32_t black;             /* 1 = the frames are identical */
    double   motion;            /* the Pacer's motion units (0..765) */
} FhClassification;

typedef struct FhPush {
    uint32_t ready;             /* 1 = an interval can be generated now */
    int32_t  base_slot;         /* the real frame that starts it (present it unless fh_stream_real_replaced) */
    uint32_t stats_pending;     /* 1 = fh_stream_resolve_stats before generating (video + cadence) */
    double   start_pts, end_pts;
} FhPush;

FhStreamOptions fh_stream_options_default(void);
FhStream*    fh_stream_create(const FhEngine* engine, const FhModel* model, const FhSessionOptions* session, const FhStreamOptions* options);
void         fh_stream_destroy(FhStream* stream);
int          fh_stream_add_input(FhStream* stream, const FhEngine* engine);
int          fh_stream_add_output(FhStream* stream, const FhEngine* engine);
/* the slot to write the next frame into: never one the stream still reads nor one of the n slots in in_use; -1 = add inputs */
int          fh_stream_next_input(FhStream* stream, const uint32_t* in_use, size_t n);
int          fh_stream_write_rgba(const FhStream* stream, const FhEngine* engine, uint32_t slot, const uint8_t* rgba, size_t len);
int          fh_stream_write_mask(const FhStream* stream, const FhEngine* engine, const uint8_t* mask, size_t len);
/* the pair (previous, slot) before pushing it, waiting for the GPU: 1 = out filled, 0 = take it as motion, -1 = error */
int          fh_stream_classify(FhStream* stream, const FhEngine* engine, uint32_t previous, uint32_t slot, FhClassification* out);
/* the newest frame is in slot with timestamp pts (the host's units, usually media ms): runs its pair; 0 / -1 */
int          fh_stream_push(FhStream* stream, const FhEngine* engine, uint32_t slot, double pts, FhPush* out);
int          fh_stream_resolve_stats(FhStream* stream, const FhEngine* engine);
int          fh_stream_real_replaced(const FhStream* stream);
/* phase t of the current interval into out: 1 = written, 0 = the source frame is enough, -1 = error */
int          fh_stream_generate(FhStream* stream, const FhEngine* engine, float t, uint32_t out);
int          fh_stream_read_rgba(const FhStream* stream, const FhEngine* engine, uint32_t out, uint8_t* buf, size_t cap);
int          fh_stream_warm_up(FhStream* stream, const FhEngine* engine);
int          fh_stream_import_d3d11(FhStream* stream, const FhEngine* engine, intptr_t handle, uint32_t width, uint32_t height);   /* as fh_session_import_d3d11 */
int          fh_stream_reset(FhStream* stream);                               /* seek, decoder gap */
/* camera warp (ARCHITECTURE.md): mouse counts at t, on the clock of push's pts (then arrival times in host ms); the
 * first call starts it. warp: the frame about to be presented at now (input slot, or output slot when output is 1)
 * turned by the mouse motion since, into output slot out: 1 = present out, 0 = off, present the frame as it is */
int          fh_stream_mouse(FhStream* stream, const FhEngine* engine, double t, double dx, double dy);
int          fh_stream_set_mouse_captured(FhStream* stream, uint32_t captured);
int          fh_stream_warp(FhStream* stream, const FhEngine* engine, uint32_t output, uint32_t slot, uint32_t out, double now);
int          fh_stream_report_json(const FhStream* stream, char* buf, size_t cap);

/* ---- Pacer: the engine's scheduler for a host with its own display loop (ARCHITECTURE.md) ----
 * Pure logic, no clock: every call takes `now`, the host's monotonic milliseconds. Per display tick call
 * fh_pacer_tick, present what it says, generate the requests (fh_pacer_requests) through the stream and report
 * each with fh_pacer_generated. Per source frame, once its kind is known, call fh_pacer_source_frame, generate the
 * requests it returns, then push the frame. Frames are the host's own ids (slot indices). */
typedef struct FhPacerOptions {
    uint32_t mode;              /* 0 auto (default), 1 a fixed factor, 2 the display's rate, 3 target_fps */
    uint32_t factor;            /* mode 1: presentations per source frame, 1..6 */
    double   target_fps;        /* mode 3 */
    uint32_t max_factor;        /* ceiling of auto, 1..6 (default 6) */
    double   fps_limit;         /* auto capped at this rate, 0 = none */
    uint32_t lookahead;         /* 1 = the generated interval is one frame back (video mode of the stream) */
    uint32_t limit_to_display;  /* 1 (default) = the refresh rate is a ceiling; 0 = generate above it */
    uint32_t fill_display;      /* mode 2: 1 = fill the whole panel rate instead of leaving the headroom */
    double   display_headroom;  /* share of the refresh rate the planner may use (0.97) */
    double   compute_budget;    /* share of the source interval the GPU may spend per interval (0.9) */
    double   delay_floor_ms, delay_max_ms;        /* presentation delay window (60, 180) */
    double   lateness_intervals;                  /* how late a frame may still be shown, in output intervals (3) */
    uint32_t queue_capacity;                      /* pending presentations (24) */
    double   pressure_tau_ms;                     /* drop pressure memory (300) */
    double   penalty_up, penalty_burst, penalty_min_gap_ms;   /* AIMD thresholds (1.2, 3, 500) */
    uint32_t penalty_max;                         /* steps auto may lose (3) */
    double   recovery_after_ms;                   /* clean time before auto recovers a step (6000) */
    double   stall_window_ms, stall_rate_limit;   /* drops within the window after a host stall are exempt (20, 4 per 2 s) */
    double   settle_ms;                           /* drops exempt after fh_pacer_mode_change (1200) */
    uint32_t motion_adaptive;                     /* 1 (default) = fewer mids on fast scenes */
    double   min_present_interval_ms;             /* the host's present spacing limit (0; Firefox 7) */
    double   playback_rate;                       /* media seconds per wall second (1) */
    double   mid_cost_seed_ms, pair_cost_seed_ms; /* assumed GPU costs until measured (10, 4) */
    uint32_t gpu_timestamps;                      /* 1 (default) = the host measures GPU time, probes make sense */
    uint32_t low_latency;                         /* 1 = games: the least presentation delay that still fits the work
                                                     (10 to 20 ms at 60 fps instead of the video floor of 60) */
} FhPacerOptions;

typedef struct FhSourceFrame {
    uint32_t id;                /* the host's id of the frame (a slot index) */
    double   pts_ms;            /* media or capture time */
    uint32_t kind;              /* 0 motion, 1 scene cut, 2 held frame */
    uint32_t has_motion;        /* 1 = `motion` is set (the pair's mean difference in 0..765 units) */
    double   motion;
    uint32_t skipped;           /* frames the host knows it missed before this one */
} FhSourceFrame;

typedef struct FhRequest {
    uint64_t id;
    uint32_t previous, current; /* the pair: phase t of (previous, current) */
    float    t;
    double   due_ms;            /* wanted on screen at */
    uint32_t retime;            /* 1 = the interval's real slot re-timed (t 0); the stream may say the source is enough: fh_pacer_skipped */
    uint32_t probe;             /* 1 = run whatever the budget says, so the cost sample teaches the controller */
} FhRequest;

typedef struct FhTick {
    uint32_t has_present;       /* 1 = show frame_kind / frame_id now */
    uint32_t frame_kind;        /* 0 = one of the host's source frames, 1 = an output slot */
    uint32_t frame_id;
    double   due_ms, late_ms;
    uint32_t mid;               /* 1 = a generated frame */
    uint32_t fallback;          /* 1 = the generated frame was not there in time; this is its anchor */
    uint32_t factor;            /* the factor in force */
    uint32_t dropped;           /* frames skipped on this tick */
    uint32_t requests;          /* frames to generate now: fh_pacer_requests */
} FhTick;

FhPacerOptions fh_pacer_options_default(void);
FhPacer*     fh_pacer_create(const FhPacerOptions* options);                 /* NULL options = the defaults */
void         fh_pacer_destroy(FhPacer* pacer);
int          fh_pacer_set_options(FhPacer* pacer, double now, const FhPacerOptions* options);   /* live; 0 / -1 */
int          fh_pacer_tick(FhPacer* pacer, double now, FhTick* out);          /* 0 / -1 */
/* returns how many requests must be generated before the pair changes (fh_pacer_requests), or -1 */
int          fh_pacer_source_frame(FhPacer* pacer, double now, const FhSourceFrame* frame);
/* copies the requests of the last tick / source_frame into buf (at most cap); returns how many there are in all */
int          fh_pacer_requests(const FhPacer* pacer, FhRequest* buf, size_t cap);
int          fh_pacer_generated(FhPacer* pacer, double now, uint64_t request, uint32_t output);   /* generated into output slot */
int          fh_pacer_skipped(FhPacer* pacer, uint64_t request);             /* the source frame is enough */
/* stage 0 pair, 1 mid, 2 present; 1 = taken, 0 = ignored inside the settle window (measure again), -1 = error */
int          fh_pacer_cost_sample(FhPacer* pacer, double now, uint32_t stage, double ms);
int          fh_pacer_host_stall(FhPacer* pacer, double now, double ms);
int          fh_pacer_mode_change(FhPacer* pacer, double now);               /* fullscreen, resize: a settle window */
int          fh_pacer_reset(FhPacer* pacer, double now);                     /* seek, new source */
int          fh_pacer_report_json(const FhPacer* pacer, char* buf, size_t cap);   /* bytes written or -1 */
int          fh_pacer_record(FhPacer* pacer, const char* host);                /* record the calls from now on as trace v1 */
int          fh_pacer_trace_json(const FhPacer* pacer, char* buf, size_t cap);    /* the recording; bytes written or -1 */

/* ---- Generator: the CPU convenience ---- */
typedef struct FhOptions {
    uint32_t width, height;     /* frame size (>= 64 x 64) */
    uint32_t k;                 /* fine-flow iterations 1..3 (default 1; 0 = default) */
    uint32_t internal_w, internal_h;   /* network resolution; 0,0 = automatic (480-line rung above 960 px, else the frame) */
    uint32_t mode;              /* 0 realtime (default), 1 video (one frame of lookahead) */
    uint32_t repair;            /* periodic-structure repair: 0 off, 1 safe (default), 2 aggressive (drawn content) */
    uint32_t cadence;           /* video mode only: 1 = re-time elements animated on 2s (default 0) */
    uint32_t render_sampling;   /* 0 = the model's (default), 1 = bilinear, 2 = bicubic */
    uint32_t keep_static;       /* 1 = static protection: pixels unchanged for static_frames frames (HUD, logos) are kept */
    uint32_t static_frames;     /* unchanged source frames before a pixel is protected (default 8; 0 = default) */
    uint32_t static_tolerance;  /* largest change, 0..255 levels, still counted as unchanged (default 4) */
    uint32_t fast_render;       /* 1 = render from the source textures (same quality, cheaper per generated frame) */
    uint32_t linear_t;          /* 1 = with fast_render: the t-dependent network once per pair, phases scaled linearly */
} FhOptions;

FhOptions    fh_options_default(uint32_t width, uint32_t height);
FhGenerator* fh_generator_create(const FhEngine* engine, const FhModel* model, const FhOptions* options);
void         fh_generator_destroy(FhGenerator* generator);
/* pushes until the first push that queues frames: 1 (realtime: the source frame itself) or 3 (video: lookahead) */
int          fh_generator_warmup(const FhGenerator* generator);
/* push one source frame; returns the number of output frames now queued (pop them before the next push), or -1 */
int          fh_generator_push(FhGenerator* generator, const uint8_t* rgba, size_t len, uint32_t factor);
/* pop the next output frame into rgba (cap >= width*height*4); 1 = frame written (phase / real filled, either may be
 * NULL), 0 = queue empty, -1 = error */
int          fh_generator_pop(FhGenerator* generator, uint8_t* rgba, size_t cap, float* phase, int* real);

#ifdef __cplusplus
}
#endif
#endif
