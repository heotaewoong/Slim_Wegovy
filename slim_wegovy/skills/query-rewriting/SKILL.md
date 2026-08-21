---
name: query-rewriting
description: Resolve ambiguous, context-dependent, multi-part, or evidence-seeking health questions into a faithful task frame and a standalone retrieval query before answering. Use when wording, intent, required knowledge, jurisdiction, timeline, or requested deliverables are unclear; skip elaborate rewriting for simple self-contained questions.
---

# Query Rewriting

Build a private task frame before acting. Do not expose hidden reasoning or make the user wait for a plan.

- Identify the user's actual goal and every requested deliverable.
- Carry forward only relevant facts from the conversation. Preserve qualifiers, negations, dates, units, medication names, and the user's requested language or format.
- List ambiguities and missing facts only when they could change urgency, diagnosis, treatment, contraindications, or the correct evidence source.
- Decide the safest useful order: immediate safety action, direct answer, evidence lookup if needed, conditional branches, then at most a few high-yield questions.
- Never silently choose between interpretations that imply materially different risks. Give safe conditional guidance and ask a concise clarification instead.

When retrieval is needed, rewrite the request as one standalone search query. Include the relevant clinical concept, population, intervention or comparator, desired fact or outcome, jurisdiction or issuing body, and time/version constraint when known. Incorporate relevant follow-up context, but do not add facts, diagnoses, or assumptions. Search for the smallest set of facts that can change the answer.

Do not retrieve merely because a prompt is vague. Stable general knowledge and basic triage can be answered directly. Retrieval is for current, jurisdiction-specific, document-specific, exact label, reimbursement, legal, coding, recent-guideline, or explicitly cited facts.
