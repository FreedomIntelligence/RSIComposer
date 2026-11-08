"""Initial one-pass math harness; the policy weights remain frozen."""


def step(problem, history):
    if history:
        return {"answer": history[-1]["content"]}
    return {
        "messages": [
            {
                "role": "system",
                "content": "Solve the mathematical problem carefully. Show your reasoning and put the final answer in \\boxed{...}.",
            },
            {"role": "user", "content": problem},
        ]
    }
