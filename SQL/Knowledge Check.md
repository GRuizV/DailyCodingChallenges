## SQL Skills Coverage Map

As for: v2.0 SQL Trainer

| Skill Area                              | Status        | Notes                                         |
| --------------------------------------- | ------------- | --------------------------------------------- |
| SELECT, WHERE, filtering                | ✅ Covered     | Basics track handles this                     |
| JOIN logic                              | ✅ Covered     | Inner/outer joins, anti-joins                 |
| Aggregations (SUM, AVG, etc.)           | ✅ Covered     | Includes HAVING logic                         |
| GROUP BY (including multi-column)       | ✅ Covered     | Routine in Aggregations track                 |
| Subqueries (correlated/scalar)          | ✅ Covered     | In Subqueries & CTEs track                    |
| Common Table Expressions (CTEs)         | ✅ Covered     | You use them fluently                         |
| Window Functions                        | ✅ In Progress | v2.0 includes multiple challenges per session |
| Time-based analysis                     | ✅ Covered     | Daily, weekly, rolling windows included       |
| Data interpretation (ambiguous prompts) | ✅ Just Added  | v2.0 simulates business vagueness             |
| Query design & justification            | ✅ Just Added  | New format requires your rationale            |
| Query optimization awareness            | 🟡 Partial    | Still early, more performance mindset coming  |
| SQL for pipelines / staging             | 🔴 Not Yet    | Needed for DE, ELT-style work                 |
| Schema design / normalization           | 🔴 Not Yet    | Required in backend + DE roles                |
| Indexes, query plans, tuning            | 🔴 Not Yet    | Performance-heavy roles only                  |
| SQL scripting (DDL, DML, transactions)  | 🟡 Light      | DML basics known, not yet using transactions  |
| Metadata queries (INFORMATION\_SCHEMA)  | 🔴 Not Yet    | Useful for DE, infra-heavy DA, or DBAs        |

## Path-Specific Recommendations

### For Data Analyst:

* Add CASE, COALESCE, NULLIF
* Learn dashboard-oriented formatting & casting
* Review pivoting (optional depending on BI tool)

### For Backend Python Developer:

* Learn CREATE TABLE, constraints, and indexes
* Use transactions (BEGIN, COMMIT, ROLLBACK)
* Master INSERT, UPDATE, DELETE

### For Data Engineer:

* Multi-layer query staging
* Incremental load patterns
* Partitioning and clustering
* Query planning tools (EXPLAIN, ANALYZE)
