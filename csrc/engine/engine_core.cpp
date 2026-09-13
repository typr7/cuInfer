#include <cstdint>
#include <format>
#include <iterator>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <msgpack.hpp>
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "engine/engine_core.h"
#include "common/logger.h"
#include "executor/model_runner.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"


namespace cuinfer
{

namespace
{

Request to_request(EngineCoreRequest request)
{
    return Request{
        .id = std::move(request.request_id),
        .max_output_tokens = request.max_output_tokens,
        .token_ids = std::move(request.token_ids),
        .sample_params = SampleParams{
            .temperature = request.temperature,
            .top_k = request.top_k,
            .top_p = request.top_p
        }
    };
}

RequestType request_type(const std::vector<zmq::message_t>& frames)
{
    if (frames.empty() || frames.front().size() != sizeof(std::uint8_t)) {
        throw ProtocolError("missing request type frame");
    }
    return static_cast<RequestType>(*frames.front().data<std::uint8_t>());
}

template <typename T>
T unpack_payload(const std::vector<zmq::message_t>& frames)
{
    if (frames.size() != 2) {
        throw ProtocolError(std::format("expected 2 frames, got {}", frames.size()));
    }
    const zmq::message_t& payload = frames.back();
    try {
        auto handle = msgpack::unpack(payload.data<const char>(), payload.size());
        return handle.get().as<T>();
    } catch (const std::exception& e) {
        throw ProtocolError(std::format("invalid payload: {}", e.what()));
    }
}

class EngineCore
{
public:
    EngineCore(const Config& config, zmq::context_t& context, const Addresses& addresses)
        : input_socket_(context, zmq::socket_type::pull),
          output_socket_(context, zmq::socket_type::push),
          model_runner_(config),
          scheduler_(
              config,
              model_runner_.model_config(),
              initialize_kv_cache(model_runner_, config)
          )
    {
        input_socket_.set(zmq::sockopt::linger, 0);
        input_socket_.connect(addresses.input_address);

        output_socket_.set(zmq::sockopt::linger, 4000);
        output_socket_.connect(addresses.output_address);
    }

    void run()
    {
        send(OutputType::kReady);
        Logger::info("EngineCore ready.");

        // block to wait input when scheduler has no request
        while (process_inputs(!scheduler_.has_requests())) {
            std::vector<ScheduledRequest> scheduled = scheduler_.schedule();
            model_runner_.run_model(scheduled);
            std::vector<SampledToken> sampled = model_runner_.finish();
            EngineCoreOutputs outputs = scheduler_.update(sampled);
            if (!outputs.empty()) {
                send(OutputType::kOutputs, outputs);
            }
        }
    }

private:
    static KVCacheConfig initialize_kv_cache(ModelRunner& model_runner, const Config& config)
    {
        std::size_t byte_size =
            model_runner.profile_available_kv_cache_memory(config.gpu_memory_utilization);

        // The block layout, including the layer count, is owned by the model runner.
        std::size_t block_byte_size = model_runner.kv_cache_block_bytes();
        int num_blocks = static_cast<int>(byte_size / block_byte_size);

        // if device memory cannot contain at least single request's kv cache, refuse to start
        const int max_model_len = model_runner.model_config().max_model_len;
        if (num_blocks * config.block_size < max_model_len) {
            throw std::runtime_error(std::format(
                "kv cache holds {} tokens but max_model_len is {}; raise gpu_memory_utilization",
                num_blocks * config.block_size,
                max_model_len
            ));
        }

        model_runner.allocate_kv_cache(num_blocks);

        return KVCacheConfig{
            .num_block_slots = config.block_size,
            .num_blocks = num_blocks
        };
    }

    bool process_inputs(bool block)
    {
        std::vector<zmq::message_t> frames;
        auto flags = block ? zmq::recv_flags::none : zmq::recv_flags::dontwait;
        while (zmq::recv_multipart(input_socket_, std::back_inserter(frames), flags)) {
            switch (request_type(frames)) {
                case RequestType::kAdd:
                    scheduler_.add_request(to_request(unpack_payload<EngineCoreRequest>(frames)));
                    break;
                case RequestType::kAbort:
                    scheduler_.abort_requests(unpack_payload<std::vector<std::string>>(frames));
                    break;
                case RequestType::kShutdown:
                    return false;
                default:
                    throw ProtocolError("unknown request type");
            }
            frames.clear();
            flags = zmq::recv_flags::dontwait;
        }
        return true;
    }

    void send(OutputType type)
    {
        send_frame(type, zmq::send_flags::none);
    }

    template <typename T>
    void send(OutputType type, const T& payload)
    {
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, payload);
        send_frame(type, zmq::send_flags::sndmore);
        send_frame(zmq::buffer(buffer.data(), buffer.size()), zmq::send_flags::none);
    }

    void send_frame(OutputType type, zmq::send_flags flags)
    {
        auto byte = static_cast<std::uint8_t>(type);
        send_frame(zmq::buffer(&byte, sizeof(byte)), flags);
    }

    void send_frame(zmq::const_buffer frame, zmq::send_flags flags)
    {
        // A blocking send only fails by throwing; the EAGAIN result is for dontwait.
        std::ignore = output_socket_.send(frame, flags);
    }

private:
    zmq::socket_t input_socket_;
    zmq::socket_t output_socket_;

    ModelRunner model_runner_;
    Scheduler scheduler_;
};

}

EngineCoreShutdownReason run_engine_core(const Config& config, const Addresses& addresses)
{
    try {
        Logger::info(std::format(
            "EngineCore started with config:\n"
                "\tmodel_path={}\n"
                "\tgpu_memory_utilization={}\n"
                "\tblock_size={}\n"
                "\tmax_num_scheduled_tokens={}\n"
                "\tmax_num_seqs",
            config.model_path,
            config.gpu_memory_utilization,
            config.block_size,
            config.max_num_scheduled_tokens,
            config.max_num_seqs
        ));

        zmq::context_t context;
        EngineCore engine_core(config, context, addresses);
        engine_core.run();
    } catch (const std::exception& e) {
        Logger::error(std::format("EngineCore died: {}", e.what()));
        return EngineCoreShutdownReason::kEngineCoreDead;
    }

    Logger::info("EngineCore shutdown.");
    return EngineCoreShutdownReason::kShutdown;
}

}
