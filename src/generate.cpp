#include "generate.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace decode {
namespace {
int maximum(const float* row, size_t size) {
    return int(std::max_element(row, row + size) - row);
}
}
Json Generation::json() const {
    return {{"drafted_tokens", drafted}, {"accepted_tokens", accepted},
            {"acceptance_rate", drafted ? Json(double(accepted) / drafted) : Json(nullptr)},
            {"verification_batches", verification_batches}, {"forward_tokens", forward_tokens},
            {"ordinary_steps", ordinary_steps}};
}
void lookup_draft(const std::vector<int>& context, size_t max_ngram, size_t limit, std::vector<int>& output) {
    output.clear();
    if (!limit || context.size() < 2) return;
    for (size_t n = std::min(max_ngram, context.size() - 1); n > 0; --n) {
        size_t suffix = context.size() - n;
        for (size_t end = suffix; end >= n; --end) {
            size_t begin = end - n;
            if (std::equal(context.begin() + begin, context.begin() + end, context.begin() + suffix)) {
                size_t count = std::min(limit, context.size() - end);
                output.assign(context.begin() + end, context.begin() + end + count);
                return;
            }
        }
    }
}
Generation generate(Engine& engine, const std::vector<int>& prompt, size_t steps, LookupOptions options, GenerationTiming* timing) {
    if (prompt.empty() || !options.prefill_batch || !options.max_ngram || options.draft_tokens > 64)
        throw std::runtime_error("invalid generation options");
    for (int token : prompt) if (token < 0 || size_t(token) >= engine.vocab_size())
        throw std::runtime_error("token ID outside vocabulary");
    engine.reset();
    using Clock = std::chrono::steady_clock;
    auto prefill_start = timing ? Clock::now() : Clock::time_point{};
    int next = 0;
    std::vector<int> input, draft;
    input.reserve(std::max(options.prefill_batch, options.draft_tokens + 1));
    draft.reserve(options.draft_tokens);
    for (size_t start = 0; start < prompt.size(); start += options.prefill_batch) {
        size_t end = std::min(prompt.size(), start + options.prefill_batch);
        bool final = end == prompt.size();
        if (options.prefill_batch == 1) {
            const auto& logits = engine.step(prompt[start], final);
            if (final) next = maximum(logits.data(), engine.vocab_size());
        } else {
            input.assign(prompt.begin() + start, prompt.begin() + end);
            const auto& logits = engine.batch(input, final ? input.size() - 1 : input.size());
            if (final) next = maximum(logits.data(), engine.vocab_size());
        }
    }
    auto decode_start = timing ? Clock::now() : Clock::time_point{};
    if (timing) timing->prefill_seconds = std::chrono::duration<double>(decode_start - prefill_start).count();
    Generation result;
    result.tokens.reserve(steps);
    std::vector<int> history = prompt;
    history.reserve(prompt.size() + steps);
    while (result.tokens.size() < steps) {
        result.tokens.push_back(next);
        history.push_back(next);
        size_t remaining = steps - result.tokens.size();
        if (!remaining) break;
        // One known token is consumed along with drafts. The final row predicts
        // a bonus token; it is emitted on the next iteration, not re-forwarded.
        lookup_draft(history, options.max_ngram,
                     std::min(options.draft_tokens, remaining - 1), draft);
        if (draft.empty()) {
            next = maximum(engine.step(next, true).data(), engine.vocab_size());
            ++result.ordinary_steps; ++result.forward_tokens;
            continue;
        }
        input.clear(); input.push_back(next);
        input.insert(input.end(), draft.begin(), draft.end());
        size_t before = engine.position();
        const auto& logits = engine.batch(input);
        ++result.verification_batches;
        result.forward_tokens += input.size();
        result.drafted += draft.size();
        size_t accepted = 0;
        for (; accepted < draft.size(); ++accepted) {
            int prediction = maximum(logits.data() + accepted * engine.vocab_size(), engine.vocab_size());
            if (prediction != draft[accepted]) { next = prediction; break; }
            result.tokens.push_back(draft[accepted]); history.push_back(draft[accepted]);
        }
        result.accepted += accepted;
        if (accepted == draft.size())
            next = maximum(logits.data() + accepted * engine.vocab_size(), engine.vocab_size());
        // Drop every rejected draft's KV entries. The mismatch prediction has
        // not been consumed; the next loop processes it exactly once.
        engine.rewind(before + accepted + 1);
    }
    if (timing) timing->decode_seconds = std::chrono::duration<double>(Clock::now() - decode_start).count();
    return result;
}
} // namespace decode
