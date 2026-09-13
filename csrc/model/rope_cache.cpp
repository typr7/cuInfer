#include <cmath>
#include <utility>
#include <vector>

#include "model/rope_cache.h"


namespace cuinfer
{

RopeCache RopeCache::create(const ModelConfig &config)
{
    constexpr float k2Pi = 6.28318530717958647692f;

    const int rotary_dim = config.head_dim;
    const int half_rotary_dim = rotary_dim / 2;

    std::vector<float> inv_freq(half_rotary_dim);
    for (int i = 0; i < half_rotary_dim; i++) {
        inv_freq[i] = 1.0f / std::pow(
            config.rope_theta,
            static_cast<float>(2 * i) / rotary_dim
        );
    }

    if (config.rope_scaling) {
        const RopeScaling& scaling = *config.rope_scaling;
        const float low_freq_wavelen
            = scaling.original_max_position_embeddings / scaling.low_freq_factor;
        const float high_freq_wavelen
            = scaling.original_max_position_embeddings / scaling.high_freq_factor;

        for (float& freq : inv_freq) {
            const float wavelen = k2Pi / freq;
            if (wavelen > low_freq_wavelen) {
                freq /= scaling.factor;
            } else if (wavelen >= high_freq_wavelen) {
                const float smooth = (
                    scaling.original_max_position_embeddings / wavelen
                    - scaling.low_freq_factor
                ) / (scaling.high_freq_factor - scaling.low_freq_factor);
                freq = (1.0f - smooth) * freq / scaling.factor + smooth * freq;
            }
        }
    }

    std::vector<float> host_data(
        static_cast<std::size_t>(config.max_model_len) * half_rotary_dim * 2
    );
    for (int pos = 0; pos < config.max_model_len; pos++) {
        for (int i = 0; i < half_rotary_dim; i++) {
            const float angle = pos * inv_freq[i];
            const std::size_t offset
                = (static_cast<std::size_t>(pos) * half_rotary_dim + i) * 2;
            host_data[offset] = std::cos(angle);
            host_data[offset + 1] = std::sin(angle);
        }
    }

    const std::size_t byte_size = host_data.size() * sizeof(float);
    CudaDeviceBuffer data(byte_size);
    data.upload(host_data.data(), byte_size);

    void* device_ptr = data.data();
    return RopeCache{
        .data = std::move(data),
        .view = make_tensor<3>(
            device_ptr,
            DataType::kFp32,
            {config.max_model_len, half_rotary_dim, 2}
        )
    };
}

}
