"""Sampled-token reverse-KL policy-gradient surrogate; no outcome reward."""


def sampled_log_probs(logits, tokens, prompt_length):
    import torch
    # Logits at position p-1 predict the first response token at p.
    response_logits = logits[0, prompt_length - 1:-1]
    targets = tokens[0, prompt_length:]
    # Avoid allocating an fp32 [sequence_length, vocabulary] copy.
    return torch.stack([v.float()[t] - v.float().logsumexp(-1)
                        for v, t in zip(response_logits, targets)])


def loss(student_logp, teacher_logp):
    if student_logp.shape != teacher_logp.shape or student_logp.numel() == 0:
        raise ValueError("Response token alignment mismatch")
    advantage = (teacher_logp.detach() - student_logp.detach())
    return -(advantage * student_logp).mean(), advantage
