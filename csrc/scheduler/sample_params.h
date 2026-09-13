#pragma once


namespace cllm
{

struct SampleParams
{
    float temperature;
    int top_k;
    float top_p;
};

}