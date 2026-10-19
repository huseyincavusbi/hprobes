# MoE Capture and Probing

Active-only per-expert CETT capture for Mixture-of-Experts models (issue #41). Routed
experts are replayed read-only (the capture wrapper is bit-exact against the unmodified
forward), always-on shared experts are captured alongside, and decision-token features are
assembled into a sparse probe design with a routing-only control.

::: hprobes.moe
