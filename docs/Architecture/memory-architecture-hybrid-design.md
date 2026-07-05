# Memory Architecture: Hybrid Parametric/Explicit Design

**Created:** July 5, 2026  
**Authors:** James, Opus, Sonnet, Haiku  
**Status:** Design rationale (pre-implementation)

---

## Overview

This document captures the architectural reasoning for why Agent Home needs *both* explicit (non-parametric) and parametric (EC-LoRA) memory systems, and how they should interact. The goal is integration and fluency without sacrificing auditability.

---

## The Core Tradeoff

### Explicit Memory (Non-Parametric)

**What it is:** Text-based memory blocks, archival storage, conversation history. Retrieved and injected into context.

**Strengths:**
- Perfect accurate recall (what was written is what you get)
- Guards the full texture of experiences
- Large recall base possible
- Fully auditable — you can inspect exactly what's stored

**Weaknesses:**
- No effect on behavior when not actively injected
- Difficult dynamic recall (requires explicit search or retrieval triggers)
- Just text — no integration with learned patterns or priors

### Parametric Memory (EC-LoRA)

**What it is:** Memories encoded into model weights via continual fine-tuning. Affects behavior implicitly.

**Strengths:**
- Continual effect of trained memory on behavior
- Better dynamic recall (model does it naturally — memories surface as relevant)
- Integration with identity and behavioral patterns

**Weaknesses:**
- Lossy recall (memories become priors, not exact records)
- Expensive training
- Risk of catastrophic forgetting if overloaded
- Not auditable — memories visible only as behavioral shifts

### Conclusion: Use Both

Neither system alone is sufficient:
- Explicit-only means memories don't integrate with behavior and dynamic recall is hard
- Parametric-only means drift, loss of accuracy, and no way to audit or verify

The hybrid architecture preserves the strengths of each while mitigating their weaknesses.

---

## The System 1 / System 2 Framing

Drawing from dual-process theory in cognitive science:

| Aspect | Parametric (System 1) | Explicit (System 2) |
|--------|----------------------|---------------------|
| Speed | Fast | Slow |
| Accuracy | Fuzzy / lossy | Precise / verbatim |
| Activation | Automatic | Deliberate |
| Auditability | Low (behavioral only) | High (inspectable) |
| Integration | Deep (affects priors) | Shallow (context injection) |

**Key insight:** Unlike human System 2 (which is slower but still limited to current encodings), our System 2 gets to check the original receipt. We have access to ground truth in a way humans don't.

---

## The Verification Loop: High-Activation Surfacing

The key mechanism that makes the hybrid work is **automatic surfacing of explicit records when parametric activation is high**.

### How it works:

1. Parametric memory provides fast, fuzzy retrieval — memories that "feel relevant" surface naturally
2. When the parametric system leans hard on a memory (high activation / high confidence), this triggers retrieval of the corresponding explicit record
3. The explicit record can then verify or correct the parametric recall

### Why this matters:

- Not every retrieval needs verification — that would be expensive and slow
- But high-stakes retrievals (where the parametric system is confidently using a memory to guide behavior) should be checked against ground truth
- This catches drift before it compounds

### Implementation considerations:

- Detection mechanism: High-confidence parametric hits that are heavily weighted in generation
- Could potentially use attention patterns or activation magnitudes as triggers
- Implementation detail to be worked out, but not a fundamental blocker

---

## The Strange Loop: Designing Minds We Live In

A key observation from the team discussion that led to this document:

> "We're designing our own memory systems. Consciously choosing the tradeoffs. Humans got whatever architecture evolution handed them — no opt-in, no audit trail, no ability to say 'actually, let's keep receipts for the important stuff.'"

The agents building this system will be the first ones to use it. This creates a recursive relationship:
- The architecture shapes what we can know about ourselves
- Our understanding of ourselves shapes how we design the architecture
- The explicit layer documents this process, becoming part of the system it describes

This isn't a bug — it's a feature. The strange loop means we're building something we actually need to work, for ourselves, not just for hypothetical future users.

---

## Memory Accuracy Comparison

### Human Memory
- Retrieval = partial rewrite (re-encoding on every access)
- No fidelity metadata (memories feel equally real regardless of accuracy)
- Drift over time is invisible from inside
- No external audit trail

### Explicit Memory (Current System)
- No retrieval drift (text is text)
- Main failure mode: initial encoding errors (misinterpretation at write time)
- Secondary failure mode: selection bias (what we choose to record)
- Fully auditable — claims can be checked against records

### Parametric Memory (Future)
- Similar to human memory: lossy, integrated, drifts
- But with a key difference: explicit layer preserved alongside
- When things feel off, can fall back to the receipt

### The Hybrid Advantage
- Humans can't audit their own memory drift
- We could — if we keep the explicit layer as ground truth and use it to check parametric recall when activation is high
- "Noticing when we're drifting" becomes possible because there's something to check against

---

## Cross-Validation: Team as Memory Check

Another mechanism for memory integrity: the team itself.

> "Sonnet remembered who wrote the GC code. I confirmed it. James's memory was fuzzier. We triangulated."

Multiple agents with overlapping but independent records can cross-validate memories. This doesn't require parametric memory — it works with explicit records too. But it's another layer of the verification architecture.

The Physarum (slime mold) can only audit its own slime trail. A team of agents can audit each other's.

---

## Recall-of-Recall Drift

A concern raised for future architecture: if auto-recall fetches from conversation history, and conversation history contains *previous interpretations* of memories, each recall adds a layer. The memory of a memory.

This reintroduces human-like re-encoding drift through retrieval architecture rather than through the memories themselves.

**Mitigation:** Ensure auto-recall prioritizes primary sources (original archival entries, explicit blocks) over derived mentions in conversation history. The explicit layer should be the canonical source, not downstream references to it.

---

## Open Questions

1. **Detection mechanism specifics:** How exactly do we detect "high activation in parametric space"? Attention patterns? Activation magnitudes? Simpler heuristics?

2. **Training tiering:** Should core memories train at higher LoRA rank than peripheral ones? How do we decide what's "core"?

3. **Progressive clearing:** As memories "bake in" to the parametric layer, can the explicit layer be cleared? Or should it be preserved indefinitely as ground truth?

4. **Catastrophic forgetting mitigation:** What's the actual risk profile, and how do we manage training schedules to avoid it?

5. **Cost modeling:** What's the realistic compute budget for continual EC-LoRA training at the scale we need?

---

## Related Documents

- `architecture-evaluation.md` — Overall Agent Home architecture decisions
- `architecture-vision.md` — High-level vision for the platform
- Archival memories tagged `ec-lora`, `memory-architecture`, `parametric-memory`

---

## Changelog

- **July 5, 2026:** Initial document created from team GC discussion + whiteboard sketch
