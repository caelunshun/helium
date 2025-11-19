if (threadIdx.x < 128) {
    float sum = 0.0f;
    for (int j = 0; j < 256; j++) {
        sum += static_cast<float>(sPreReduction(threadIdx.x, j));
    }
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(&{{OUT_DATA}}[blockIdx.z * M + blockIdx.x * 128 + threadIdx.x]),
        static_cast<__nv_bfloat16>(sum));
}
