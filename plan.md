1. Add an execution plan for performance optimization:
   * Replace `return F.relu(x) ** 2` with `r = F.relu(x); return r * r` in `gpu/model.py`.
   * Explain the mathematical rationale.
   * Include the pre-commit step.
