"""
Pre-loaded examples for QA and Summarization tabs.
All examples are drawn from the Module 7 Week B reading.
"""

QA_EXAMPLES = {
    "Alan Turing": {
        "context": (
            "Alan Mathison Turing was an English mathematician, logician, and computer scientist. "
            "He was born in Maida Vale, London, on 23 June 1912. "
            "Turing is widely considered to be the father of theoretical computer science and artificial intelligence."
        ),
        "question": "Where was Alan Turing born?",
        "gold": "Maida Vale, London",
    },
    "Battery research": {
        "context": (
            "Researchers at Stanford University reported on Tuesday that a new battery technology "
            "based on solid-state lithium-metal cells has achieved 1,000 charge cycles in laboratory "
            "tests with less than 5% capacity loss. The research team plans to license the technology "
            "to a commercial battery manufacturer for production trials in 2027."
        ),
        "question": "How many charge cycles did the battery achieve?",
        "gold": "1,000 charge cycles",
    },
    "Transformer architecture": {
        "context": (
            "An encoder–decoder transformer has two stacks: the encoder reads the input and produces "
            "a sequence of hidden states, while the decoder generates the output autoregressively — "
            "one token at a time — attending both to prior generated tokens and to the encoder's "
            "hidden states via cross-attention. This architecture is used for summarization, "
            "translation, and generative QA."
        ),
        "question": "What mechanism does the decoder use to attend to encoder output?",
        "gold": "cross-attention",
    },
}

SUMM_EXAMPLES = {
    "Solid-state battery": {
        "article": (
            "Researchers at Stanford University reported on Tuesday that a new battery technology "
            "based on solid-state lithium-metal cells has achieved 1,000 charge cycles in laboratory "
            "tests with less than 5% capacity loss. The breakthrough could enable electric vehicles "
            "with both longer range and faster charging, addressing two of the most persistent "
            "barriers to widespread adoption. The research team plans to license the technology "
            "to a commercial battery manufacturer for production trials in 2027."
        ),
        "reference": (
            "Stanford researchers achieved 1,000-cycle solid-state battery tests with under 5% "
            "capacity loss, potentially enabling longer-range and faster-charging EVs, "
            "with commercial trials planned for 2027."
        ),
    },
    "AI chip research": {
        "article": (
            "A team at MIT has demonstrated a new AI chip architecture that processes transformer "
            "attention operations directly in memory, eliminating the data-movement bottleneck that "
            "currently limits large language model inference speed. In benchmarks, the chip achieved "
            "a 12x speedup over GPU baselines on 70-billion-parameter models while consuming 40% "
            "less energy. The team expects first silicon tape-out by Q3 2026."
        ),
        "reference": (
            "MIT researchers developed an AI chip that runs transformer attention in-memory, "
            "achieving 12x speedup over GPUs on 70B-parameter models with 40% less energy, "
            "targeting a 2026 tape-out."
        ),
    },
}

# Five decision factors from the reading (Section 8)
DECISION_FACTORS = [
    {
        "name": "1. Labeled-data availability",
        "desc": "How much labeled training data is available?",
        "lo": "1 = none / very scarce",
        "hi": "5 = thousands of examples",
        "weight": 0.25,
    },
    {
        "name": "2. Task specificity",
        "desc": "How niche is the task vs. general pre-training coverage?",
        "lo": "1 = generic / common",
        "hi": "5 = highly domain-specific",
        "weight": 0.20,
    },
    {
        "name": "3. Compute budget",
        "desc": "Available compute for training runs?",
        "lo": "1 = CPU-only / very limited",
        "hi": "5 = GPU cluster / ample",
        "weight": 0.20,
    },
    {
        "name": "4. Iteration speed needed",
        "desc": "How exploratory is the project?",
        "lo": "1 = need results today",
        "hi": "5 = can wait days per training run",
        "weight": 0.15,
    },
    {
        "name": "5. Quality gap",
        "desc": "How far is the pre-trained baseline from your quality target?",
        "lo": "1 = already meets target",
        "hi": "5 = far below target",
        "weight": 0.20,
    },
]