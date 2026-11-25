// Narrow C interface to the upstream 64-wide, noncausal, FP16 Turing kernel.
#include "flash_fwd_launch_template.h"

extern "C" int run(half_t* q, half_t* k, half_t* v, half_t* output,
                   float* logsumexp, int batch, int length, int heads) {
    Flash_fwd_params p{};
    p.q_ptr = q;
    p.k_ptr = k;
    p.v_ptr = v;
    p.o_ptr = output;
    p.l_ptr = logsumexp;
    p.b = batch;
    p.seqlen_q = p.seqlen_k = length;
    p.h = p.h_k = heads;
    p.h_h_k_ratio = 1;
    p.d = 64;
    p.softmax_scale = 0.125f;
    run_mha_fwd_hdim64<false>(p);
    return cudaGetLastError();
}
