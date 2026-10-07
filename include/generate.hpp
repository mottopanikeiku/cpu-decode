#pragma once
#include "decode.hpp"

namespace decode {
struct LookupOptions {
    size_t draft_tokens = 0;
    size_t max_ngram = 4;
    size_t prefill_batch = 8;
};
struct Generation {
    std::vector<int> tokens;
    size_t drafted = 0, accepted = 0, verification_batches = 0;
    size_t forward_tokens = 0, ordinary_steps = 0;
    Json json() const;
};
struct GenerationTiming {
    double prefill_seconds = 0, decode_seconds = 0;
};
// Search earlier occurrences of the longest suffix first, most recent first.
void lookup_draft(const std::vector<int>& context, size_t max_ngram, size_t limit, std::vector<int>& output);
Generation generate(Engine& engine, const std::vector<int>& prompt, size_t steps,
                    LookupOptions options = {}, GenerationTiming* timing = nullptr);
} // namespace decode
