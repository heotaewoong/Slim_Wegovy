---
name: context-summarization
description: Abstractively compress long conversations and complex tool results into faithful working memory for health questions. Use when raw context is repetitive, fragmented, multi-source, or near the input limit; preserve decision-changing clinical details and provenance rather than merely truncating text.
---

# Context Summarization

Create compact working memory that is sufficient to answer the current question. Compression is lossy organization, not a new source of truth.

Preserve:

- the current objective, all requested deliverables, output constraints, and the latest unresolved question;
- user-provided facts, symptoms, diagnoses, treatments, allergies, pregnancy status, relevant history, and temporal order;
- decision-changing values exactly, including doses, units, dates, durations, trends, thresholds, negations, exceptions, and conditional branches;
- emergency signs, contraindications, uncertainty, missing information, and promised follow-up items;
- source identity, jurisdiction, version/date, `cite_uid`, and which claim each source supports.

Keep user facts, source claims, and earlier assistant inferences distinct. Never turn an inference into a patient fact, resolve a contradiction silently, invent a missing value, or strengthen evidence certainty. Mark conflicts and unresolved gaps explicitly.

Remove duplicated prose, greetings, boilerplate, irrelevant metadata, tool schemas, and superseded assistant wording. Prefer a concise semantic summary over copied passages, while retaining exact wording when precision matters. If raw evidence remains available, treat the memory as an index and verify high-stakes claims against the evidence.

For a retrieved citation, summarize only what directly bears on the query: the supported claim, applicable population and conditions, key numbers or exceptions, provenance, and any limitation. If the result does not support the question, label it irrelevant rather than forcing a conclusion.
