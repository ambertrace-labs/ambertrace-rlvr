---
name: Feature request
about: Suggest an improvement or a new capability
title: ""
labels: enhancement
assignees: ""
---

## Problem

What are you trying to do, and where does the current library fall short?

## Proposal

What you'd like to see. If it's a new domain, note that domains are meant to be
a **config + a parser, not a fork** — describe the rules and the report shape
rather than a code change where possible.

## Alternatives considered

Other approaches you weighed, and why this one.

## Scope check

- [ ] This keeps the reward runtime **read-only** (authoring stays in the `ambertraceai` SDK)
- [ ] This preserves **fail-closed** and **bounded/monotonic** reward behaviour
