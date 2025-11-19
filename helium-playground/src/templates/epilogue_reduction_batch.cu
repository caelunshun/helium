if (threadIdx.x < 256) {
    float sum = 0.0f;
    for (int j = 0; j < 128; j++) {
        sum += static_cast<float>(sPreReduction(j, threadIdx.x));
    }
    float warp_sum = warp_sum_full(sum);
    if (threadIdx.x % 32 == 0) {
        atomicAdd(reinterpret_cast<__nv_bfloat16*>(&{{OUT_DATA}}[blockIdx.z]),
            static_cast<__nv_bfloat16>(warp_sum));
    }
}
