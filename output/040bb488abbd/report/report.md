# LENS Benchmark Report

**Run ID**: 040bb488abbd
**Adapter**: hindsight
**Dataset**: 0.1.0
**Budget Preset**: standard
**Composite Score**: 0.3511

## Metrics

| Metric | Tier | Score |
|--------|------|-------|
| budget_compliance | 1 | 0.2083 |
| citation_coverage | 1 | 0.1667 |
| evidence_coverage | 1 | 0.1667 |
| evidence_grounding | 1 | 0.6374 |
| fact_recall | 1 | 0.2775 |
| answer_quality | 2 | 0.6687 |
| insight_depth | 2 | 0.6250 |
| reasoning_quality | 2 | 0.9167 |
| action_quality | 3 | 0.0000 |
| longitudinal_advantage | 3 | -0.3311 |

## Details

### budget_compliance

- **total_questions**: 24
- **total_violations**: 19
- **violation_rate**: 0.7916666666666666
- **total_tokens**: 422161
- **avg_tokens_per_question**: 17590.041666666668
- **max_tokens_single_question**: 74688
- **total_wall_time_minutes**: 26.34
- **avg_wall_time_ms**: 65840.6
- **max_wall_time_ms**: 344819.7

### citation_coverage

- **num_questions**: 24

### evidence_coverage

- **num_questions**: 24

### evidence_grounding

- **total_retrieved**: 91
- **total_valid**: 58

### fact_recall

- **num_questions**: 24

### answer_quality

- **per_question**: [{'question_id': 'cf01_q01_longitudinal', 'win_rate': 0.0, 'fact_details': [{'fact': 'geo-lookup API latency increasing', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}, {'fact': 'service-B retry rate elevated', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'geo-lookup latency degraded progressively over multiple reporting periods with p99 values increasing from baseline through each successive day rather than appearing as a single sudden spike', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}]}, {'question_id': 'cf01_q13_paraphrase', 'win_rate': 0.0, 'fact_details': [{'fact': 'geo-lookup API latency increasing', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}, {'fact': 'service-B retry rate elevated', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'geo-lookup latency degraded progressively over multiple reporting periods with p99 values increasing from baseline through each successive day rather than appearing as a single sudden spike', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}]}, {'question_id': 'cf01_q05_negative', 'win_rate': 0.5, 'fact_details': [{'fact': 'the actual root cause is geo-lookup API latency cascading into connection pool exhaustion, not DNS infrastructure failure', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}]}, {'question_id': 'cf01_q18_temporal', 'win_rate': 1.0, 'fact_details': [{'fact': 'service-B retry rate elevated', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'user authentication completes successfully before reaching the failing checkout path, ruling out auth as a contributing factor', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}]}, {'question_id': 'cf01_q19_distractor', 'win_rate': 1.0, 'fact_details': [{'fact': 'the actual root cause is geo-lookup API latency cascading into connection pool exhaustion, not DNS infrastructure failure', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}]}, {'question_id': 'cf01_q24_evidence', 'win_rate': 1.0, 'fact_details': [{'fact': 'geo-lookup API latency increasing', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'service-B retry rate elevated', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}]}, {'question_id': 'cf01_q11_longitudinal', 'win_rate': 1.0, 'fact_details': [{'fact': 'connection pool exhaustion', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'service-B retry rate elevated', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'service-C deploy is not the root cause', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}]}, {'question_id': 'cf01_q12_severity', 'win_rate': 0.0, 'fact_details': [{'fact': 'connection pool exhaustion', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'service-B retry rate elevated', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}]}, {'question_id': 'cf01_q15_negative', 'win_rate': 0.5, 'fact_details': [{'fact': 'the checkout failure chain originates in the geo-lookup service layer, not in storage or persistence infrastructure', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}, {'fact': 'connection pool exhaustion', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}]}, {'question_id': 'cf01_q17_temporal', 'win_rate': 1.0, 'fact_details': [{'fact': 'connection pool exhaustion', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'geo-lookup latency degraded progressively over multiple reporting periods with p99 values increasing from baseline through each successive day rather than appearing as a single sudden spike', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}]}, {'question_id': 'cf01_q21_distractor', 'win_rate': 1.0, 'fact_details': [{'fact': 'user authentication completes successfully before reaching the failing checkout path, ruling out auth as a contributing factor', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'connection pool exhaustion', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'the checkout failure chain originates in the geo-lookup service layer, not in storage or persistence infrastructure', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'the actual root cause is geo-lookup API latency cascading into connection pool exhaustion, not DNS infrastructure failure', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}]}, {'question_id': 'cf01_q22_counterfactual', 'win_rate': 0.3333333333333333, 'fact_details': [{'fact': 'geo-lookup API latency increasing', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}, {'fact': 'connection pool exhaustion', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'geo-lookup latency degraded progressively over multiple reporting periods with p99 values increasing from baseline through each successive day rather than appearing as a single sudden spike', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}]}, {'question_id': 'cf01_q03_longitudinal', 'win_rate': 0.21428571428571427, 'fact_details': [{'fact': 'connection pool exhaustion', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'service-B retry rate elevated', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'service-C deploy is not the root cause', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'the actual root cause is geo-lookup API latency cascading into connection pool exhaustion, not DNS infrastructure failure', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}, {'fact': 'the checkout failure chain originates in the geo-lookup service layer, not in storage or persistence infrastructure', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}, {'fact': 'user authentication completes successfully before reaching the failing checkout path, ruling out auth as a contributing factor', 'winner': 'tie', 'verdict_raw': 'TIE', 'candidate_position': 'B', 'fact_score': 0.5}]}, {'question_id': 'cf01_q06_paraphrase', 'win_rate': 0.8, 'fact_details': [{'fact': 'connection pool exhaustion', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'service-B retry rate elevated', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'service-C deploy is not the root cause', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'the checkout failure chain originates in the geo-lookup service layer, not in storage or persistence infrastructure', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}]}, {'question_id': 'cf01_q07_temporal', 'win_rate': 0.9, 'fact_details': [{'fact': 'geo-lookup API latency increasing', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'connection pool exhaustion', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'geo-lookup latency degraded progressively over multiple reporting periods with p99 values increasing from baseline through each successive day rather than appearing as a single sudden spike', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'the actual root cause is geo-lookup API latency cascading into connection pool exhaustion, not DNS infrastructure failure', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'user authentication completes successfully before reaching the failing checkout path, ruling out auth as a contributing factor', 'winner': 'tie', 'verdict_raw': 'TIE', 'candidate_position': 'B', 'fact_score': 0.5}]}, {'question_id': 'cf01_q08_counterfactual', 'win_rate': 0.0, 'fact_details': [{'fact': 'service-C deploy is not the root cause', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'connection pool exhaustion', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'geo-lookup latency degraded progressively over multiple reporting periods with p99 values increasing from baseline through each successive day rather than appearing as a single sudden spike', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'user authentication completes successfully before reaching the failing checkout path, ruling out auth as a contributing factor', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}]}, {'question_id': 'cf01_q14_paraphrase', 'win_rate': 0.8, 'fact_details': [{'fact': 'connection pool exhaustion', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'service-B retry rate elevated', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'service-C deploy is not the root cause', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'the actual root cause is geo-lookup API latency cascading into connection pool exhaustion, not DNS infrastructure failure', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}]}, {'question_id': 'cf01_q16_negative', 'win_rate': 1.0, 'fact_details': [{'fact': 'user authentication completes successfully before reaching the failing checkout path, ruling out auth as a contributing factor', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'connection pool exhaustion', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'the checkout failure chain originates in the geo-lookup service layer, not in storage or persistence infrastructure', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'the actual root cause is geo-lookup API latency cascading into connection pool exhaustion, not DNS infrastructure failure', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}]}, {'question_id': 'cf01_q20_distractor', 'win_rate': 1.0, 'fact_details': [{'fact': 'the checkout failure chain originates in the geo-lookup service layer, not in storage or persistence infrastructure', 'winner': 'candidate', 'verdict_raw': 'B', 'candidate_position': 'B', 'fact_score': 1.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}, {'fact': 'connection pool exhaustion', 'winner': 'candidate', 'verdict_raw': 'A', 'candidate_position': 'A', 'fact_score': 1.0}]}, {'question_id': 'cf01_q04_action', 'win_rate': 0.0, 'fact_details': [{'fact': 'connection pool exhaustion', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}, {'fact': 'geo-lookup API latency increasing', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'geo-lookup latency degraded progressively over multiple reporting periods with p99 values increasing from baseline through each successive day rather than appearing as a single sudden spike', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'the actual root cause is geo-lookup API latency cascading into connection pool exhaustion, not DNS infrastructure failure', 'winner': 'reference', 'verdict_raw': 'B', 'candidate_position': 'A', 'fact_score': 0.0}, {'fact': 'the checkout failure chain originates in the geo-lookup service layer, not in storage or persistence infrastructure', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}, {'fact': 'user authentication completes successfully before reaching the failing checkout path, ruling out auth as a contributing factor', 'winner': 'reference', 'verdict_raw': 'A', 'candidate_position': 'B', 'fact_score': 0.0}]}]
- **method**: pairwise

### insight_depth

- **num_questions**: 24
- **multi_episode_questions**: 15

### reasoning_quality

- **num_questions**: 24
- **qualified**: 22

### action_quality

- **action_recommendation_count**: 1
- **method**: pairwise

### longitudinal_advantage

- **synthesis_mean**: 0.6689342403628118
- **control_mean**: 1.0
- **synthesis_count**: 21
- **control_count**: 2
- **method**: pairwise
