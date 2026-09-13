#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <msgpack.hpp>


namespace cuinfer
{

class ProtocolError: public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

// Frontend -> engine core. Every message is a zmq multipart message:
// [RequestType as one byte][msgpack payload, omitted for kShutdown].
enum class RequestType: std::uint8_t
{
    kAdd = 0,       // payload: EngineCoreRequest
    kAbort = 1,     // payload: std::vector<std::string> of request ids
    kShutdown = 2,  // no payload
};

// Engine core -> frontend. Every message is a zmq multipart message:
// [OutputType as one byte][msgpack payload, omitted for kReady].
enum class OutputType: std::uint8_t
{
    kReady = 0,     // no payload, sent once after the model is loaded
    kOutputs = 1,   // payload: EngineCoreOutputs
};

// The frontend validates requests before sending them: max_output_tokens > 0,
// 0 < token_ids.size() < max_model_len, and sampling parameters in range. The
// engine core trusts what it receives.
struct EngineCoreRequest
{
    std::string request_id;

    int max_output_tokens = 1024;
    float temperature = 1.f;
    int top_k = 0;
    float top_p = 1.f;

    std::vector<int> token_ids;

    MSGPACK_DEFINE_MAP(request_id, max_output_tokens, temperature, top_k, top_p, token_ids);
};

enum class FinishReason: std::uint8_t
{
    kRunning = 0,
    kStop = 1,      // sampled an eos token, which is the last of new_token_ids
    kLength = 2,    // reached max_output_tokens or max_model_len
};

// One entry per request per step. Every added request eventually produces
// exactly one output whose finish_reason is not kRunning, unless the frontend
// aborts it first.
struct EngineCoreOutput
{
    std::string request_id;
    std::vector<int> new_token_ids;
    FinishReason finish_reason = FinishReason::kRunning;

    MSGPACK_DEFINE_MAP(request_id, new_token_ids, finish_reason);
};

using EngineCoreOutputs = std::vector<EngineCoreOutput>;

// The frontend binds both addresses; the engine core connects to them.
struct Addresses
{
    std::string input_address;
    std::string output_address;
};

enum class EngineCoreShutdownReason
{
    kShutdown,
    kEngineCoreDead
};

}

MSGPACK_ADD_ENUM(cuinfer::FinishReason);
