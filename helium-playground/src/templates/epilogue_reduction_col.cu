if (threadIdx.x < 256) {
    float sum = 0.0f;
    for (int j = 0; j < 128; j++) {
        sum += static_cast<float>(sPreReduction(j, threadIdx.x));
    }
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(&{{OUT_DATA}}[blockIdx.z * N + blockIdx.y * 256 + threadIdx.x]),
        static_cast<__nv_bfloat16>(sum));
}
