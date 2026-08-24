# RE-call prior art gap report

This report identifies research candidates. It does not assert that a capability has never been implemented.

## Capability status

### representation.raw_episodes

Group: `representation`. Status: **established**.

Definition: The system preserves source episodes or turns as retrievable memory.

Systems with reviewed claims: `sys_letta`, `sys_memmachine`, `sys_quipu`.

### representation.structured_facts

Group: `representation`. Status: **emerging**.

Definition: The system represents memory as structured facts or attributes rather than only raw text.

Systems with reviewed claims: `sys_mem0`.

### representation.knowledge_graph

Group: `representation`. Status: **established**.

Definition: The system represents entities and relations as a graph that can be queried.

Systems with reviewed claims: `sys_graphiti`, `sys_mem0`, `sys_neo4j_agent_memory`.

### write_path.extraction

Group: `write_path`. Status: **established**.

Definition: The system extracts persistent memory from conversations or agent traces.

Systems with reviewed claims: `sys_langmem`, `sys_mem0`.

### write_path.consolidation

Group: `write_path`. Status: **established**.

Definition: The system merges, compresses, or organizes multiple memories during maintenance.

Systems with reviewed claims: `sys_mindmemos`, `sys_neo4j_agent_memory`.

### write_path.agent_authored

Group: `write_path`. Status: **established**.

Definition: The agent can deliberately decide what to write or update in persistent memory.

Systems with reviewed claims: `sys_agemem`, `sys_amem`, `sys_letta`, `sys_memory_r1`.

### retrieval.dense

Group: `retrieval`. Status: **emerging**.

Definition: The system retrieves memory using vector similarity.

Systems with reviewed claims: `sys_mem0`.

### retrieval.lexical

Group: `retrieval`. Status: **unverified_gap**.

Definition: The system retrieves memory using lexical or full text search.

No reviewed system claim exists for this capability.

### retrieval.graph_traversal

Group: `retrieval`. Status: **established**.

Definition: The system expands retrieval through graph relations or traversals.

Systems with reviewed claims: `sys_graphiti`, `sys_hipporag`.

### retrieval.query_time_context

Group: `retrieval`. Status: **emerging**.

Definition: The system constructs or expands the memory context at query time using the retrieved evidence.

Systems with reviewed claims: `sys_memmachine`.

### validity_and_revision.supersession

Group: `validity_and_revision`. Status: **emerging**.

Definition: The system represents that a newer memory replaces an older memory.

Systems with reviewed claims: `sys_recall`.

### validity_and_revision.contradiction

Group: `validity_and_revision`. Status: **emerging**.

Definition: The system detects or represents conflicting memories.

Systems with reviewed claims: `sys_langmem`, `sys_mindmemos`.

### validity_and_revision.rollback

Group: `validity_and_revision`. Status: **emerging**.

Definition: The system can restore or select an earlier memory state.

Systems with reviewed claims: `sys_chronomem`.

### time.valid_time

Group: `time`. Status: **established**.

Definition: The system records when a fact was true in the world.

Systems with reviewed claims: `sys_graphiti`, `sys_neo4j_agent_memory`, `sys_quipu`.

### time.ingestion_time

Group: `time`. Status: **emerging**.

Definition: The system records when it learned or ingested a fact.

Systems with reviewed claims: `sys_graphiti`.

### time.point_in_time

Group: `time`. Status: **unverified_gap**.

Definition: The system can answer what was known or valid at a requested historical time.

No reviewed system claim exists for this capability.

### provenance.source_attribution

Group: `provenance`. Status: **established**.

Definition: A returned memory can identify the source evidence supporting it.

Systems with reviewed claims: `sys_neo4j_agent_memory`, `sys_recall`.

### provenance.transformation_lineage

Group: `provenance`. Status: **emerging**.

Definition: Derived memories link back through transformations to their source evidence.

Systems with reviewed claims: `sys_quipu`, `sys_recall`.

### provenance.evidence_receipts

Group: `provenance`. Status: **emerging**.

Definition: The system returns a machine readable record of evidence used for a memory result.

Systems with reviewed claims: `sys_quipu`.

### uncertainty.abstention

Group: `uncertainty`. Status: **emerging**.

Definition: The system can return an explicit refusal when memory support is insufficient.

Systems with reviewed claims: `sys_recall`.

### uncertainty.entailment

Group: `uncertainty`. Status: **unverified_gap**.

Definition: The system checks whether retrieved evidence actually supports the requested claim.

No reviewed system claim exists for this capability.

### uncertainty.calibration

Group: `uncertainty`. Status: **unverified_gap**.

Definition: The system calibrates a trust or confidence decision against labeled examples.

No reviewed system claim exists for this capability.

### authority_and_scope.user_consent

Group: `authority_and_scope`. Status: **unverified_gap**.

Definition: The system makes persistent memory subject to an explicit user consent or control decision.

No reviewed system claim exists for this capability.

### authority_and_scope.tenant_scope

Group: `authority_and_scope`. Status: **established**.

Definition: The system enforces tenant boundaries on memory storage and retrieval.

Systems with reviewed claims: `sys_quipu`, `sys_recall`.

### authority_and_scope.multi_agent_handoff

Group: `authority_and_scope`. Status: **unverified_gap**.

Definition: Memory can be intentionally shared or handed from one agent to another with scope semantics.

No reviewed system claim exists for this capability.

### deletion_and_forgetting.source_deletion

Group: `deletion_and_forgetting`. Status: **emerging**.

Definition: A source memory can be deleted from the system.

Systems with reviewed claims: `sys_memory_r1`.

### deletion_and_forgetting.derived_propagation

Group: `deletion_and_forgetting`. Status: **established**.

Definition: Deleting a source removes or invalidates derived memories, graph edges, embeddings, and caches that depend on it.

Systems with reviewed claims: `sys_dependency_rollback`, `sys_quipu`.

### deletion_and_forgetting.selective_forgetting

Group: `deletion_and_forgetting`. Status: **established**.

Definition: The system can forget selected memories without clearing unrelated memory.

Systems with reviewed claims: `sys_agemem`, `sys_memorybank`.

### action_feedback.outcome_storage

Group: `action_feedback`. Status: **emerging**.

Definition: The system stores observed results of agent actions as reusable memory.

Systems with reviewed claims: `sys_merit`, `sys_neo4j_agent_memory`.

### action_feedback.outcome_linked_revision

Group: `action_feedback`. Status: **unverified_gap**.

Definition: An observed action result updates the memory claim that influenced that action.

No reviewed system claim exists for this capability.

### action_feedback.policy_revision

Group: `action_feedback`. Status: **emerging**.

Definition: Remembered outcomes modify future agent behavior or policy selection.

Systems with reviewed claims: `sys_reflexion`.

### security.prompt_injection_resistance

Group: `security`. Status: **unverified_gap**.

Definition: The system limits memory content from changing agent instructions or tool authority through injection.

No reviewed system claim exists for this capability.

### security.memory_poisoning_resistance

Group: `security`. Status: **unverified_gap**.

Definition: The system detects or limits malicious or misleading persistent memories.

No reviewed system claim exists for this capability.

### security.egress_control

Group: `security`. Status: **emerging**.

Definition: The system controls which memory fields can leave the memory boundary for a model or tool.

Systems with reviewed claims: `sys_amp`.

### evaluation.temporal_reasoning

Group: `evaluation`. Status: **emerging**.

Definition: The system is evaluated on changing facts and historical validity.

Systems with reviewed claims: `sys_longmemeval`.

### evaluation.conflict_resolution

Group: `evaluation`. Status: **unverified_gap**.

Definition: The system is evaluated on contradictory memories or conflicting claims.

Systems with reviewed claims: `sys_letta`.

### evaluation.action_impact

Group: `evaluation`. Status: **emerging**.

Definition: The system is evaluated by whether memory improves downstream agent actions or outcomes.

Systems with reviewed claims: `sys_longmemeval_v2`.

### evaluation.continual_learning

Group: `evaluation`. Status: **emerging**.

Definition: The system is evaluated over an ordered sequence of tasks or episodes where memory must support transfer and forgetting.

Systems with reviewed claims: `sys_agent_memory_bench`.

### evaluation.memory_security

Group: `evaluation`. Status: **established**.

Definition: The system or benchmark is evaluated on persistent memory poisoning, prompt injection, downstream consequences, or selective repair.

Systems with reviewed claims: `sys_bad_memory`, `sys_memsecbench`, `sys_mpbench`.

## RE-call research hypothesis

The current hypothesis is a combination of evidence backed claims, explicit validity and supersession, reversible provenance lineage, authority and scope enforcement, deletion propagation through derived artifacts, abstention based on support and conflict, and action outcome feedback into future belief state.

The matrix must establish the evidence boundary before this combination is described as novel.

## Target combination analysis

This section reports coverage of the configured RE-call hypothesis. It is not a novelty claim. A missing cell means that this corpus has not accepted evidence for that capability in that system.

Target capabilities: `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision`.

| System | Combination status | Verified support | Partial support | Missing evidence | Conflicting evidence |
| --- | --- | --- | --- | --- | --- |
| `sys_agemem` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_agent_memory_bench` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_amem` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_amp` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_bad_memory` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_chronomem` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_dependency_rollback` | `partial_combination` | `deletion_and_forgetting.derived_propagation` | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_graphiti` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_hipporag` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_langmem` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_letta` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_longmemeval` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_longmemeval_v2` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_mem0` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_memmachine` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_memory_r1` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_memorybank` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_memsecbench` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_merit` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_mindmemos` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_mpbench` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_neo4j_agent_memory` | `partial_combination` | `provenance.source_attribution` | none | `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_quipu` | `partial_combination` | `provenance.transformation_lineage`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation` | none | `provenance.source_attribution`, `validity_and_revision.supersession`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
| `sys_recall` | `partial_combination` | `provenance.source_attribution`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `uncertainty.abstention` | `provenance.transformation_lineage` | `deletion_and_forgetting.derived_propagation`, `action_feedback.outcome_linked_revision` | none|
| `sys_reflexion` | `unverified_combination` | none | none | `provenance.source_attribution`, `provenance.transformation_lineage`, `validity_and_revision.supersession`, `authority_and_scope.tenant_scope`, `deletion_and_forgetting.derived_propagation`, `uncertainty.abstention`, `action_feedback.outcome_linked_revision` | none|
