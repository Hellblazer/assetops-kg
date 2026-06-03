# Synthetic KG Benchmark Scenarios

The 40 hand-authored scenarios run by `python -m benchmark.run_samyama` against the synthetic `industrial` graph. Source of truth is the JSON in `scenarios/*.json`; this file is a rendered, searchable mirror (`.json` is not an indexed type). Per-category counts are enforced by `tests/test_scenarios.py::EXPECTED_COUNTS`.


**Total: 40 scenarios across 7 categories.**


## Multi Hop Dependency

`multi_hop_dependency.json` — 8 scenarios


### graph_dep_001 (medium)

What equipment is affected if Chiller-1 fails? Trace the full downstream dependency chain through AHU-1, Pump-CW-1, and Motor-CH1 which all depend on it.

- **Expected tools:** `impact_analysis`
- **Expected output contains:** `AHU-1`, `Pump-CW-1`, `Motor-CH1`, `cascade`, `DEPENDS_ON`
- **Requires graph:** True

### graph_dep_002 (hard)

Pump-CW-1 is scheduled for replacement. Identify all equipment within 2 hops that will be affected, considering Pump-CW-1 DEPENDS_ON Chiller-1 and Motor-P1 DEPENDS_ON Pump-CW-1.

- **Expected tools:** `impact_analysis`, `cypher_query`
- **Expected output contains:** `Chiller-1`, `Motor-P1`, `DEPENDS_ON`, `cascade`
- **Requires graph:** True

### graph_dep_003 (medium)

Boiler-2 has tube corrosion and wall thinning. Which equipment shares a system with it (Boiler-1 via SHARES_SYSTEM_WITH) and what downstream dependencies exist?

- **Expected tools:** `impact_analysis`
- **Expected output contains:** `Boiler-1`, `SHARES_SYSTEM_WITH`, `DEPENDS_ON`, `cascade`
- **Requires graph:** True

### graph_dep_004 (hard)

All equipment at HVAC-South is offline for maintenance. Trace which sensors, equipment, and active work orders are impacted across the DEPENDS_ON and SHARES_SYSTEM_WITH graphs.

- **Expected tools:** `impact_analysis`, `cypher_query`
- **Expected output contains:** `Sensor`, `WorkOrder`, `HVAC-South`, `DEPENDS_ON`
- **Requires graph:** True

### graph_dep_005 (medium)

Motor-CH1 drives Chiller-1 (Motor-CH1 DEPENDS_ON Chiller-1). If the motor bearing fails, what is the full failure cascade through AHU-1 and Pump-CW-1 which also depend on Chiller-1?

- **Expected tools:** `impact_analysis`
- **Expected output contains:** `Motor-CH1`, `Chiller-1`, `AHU-1`, `Pump-CW-1`, `cascade`
- **Requires graph:** True

### graph_dep_006 (hard)

Chiller-2 is a hub in the dependency graph with Pump-CW-2 and AHU-2 depending on it. List the transitive closure of all equipment affected if Chiller-2 goes offline.

- **Expected tools:** `impact_analysis`, `cypher_query`
- **Expected output contains:** `transitive`, `Chiller-2`, `Pump-CW-2`, `AHU-2`, `DEPENDS_ON`
- **Requires graph:** True

### graph_dep_007 (medium)

Chiller-3 has condenser water side fouling. Which equipment shares cooling loop B (Chiller-4 via SHARES_SYSTEM_WITH) and which AHUs depend on them?

- **Expected tools:** `impact_analysis`
- **Expected output contains:** `Chiller-3`, `Chiller-4`, `AHU-3`, `AHU-4`, `SHARES_SYSTEM_WITH`
- **Requires graph:** True

### graph_dep_008 (hard)

Compare the blast radius of failing Pump-CW-1 versus Pump-CW-2. Which has a larger downstream impact considering their respective dependency chains?

- **Expected tools:** `impact_analysis`, `cypher_query`
- **Expected output contains:** `blast radius`, `Pump-CW-1`, `Pump-CW-2`, `downstream`, `DEPENDS_ON`
- **Requires graph:** True

## Cross Asset Correlation

`cross_asset_correlation.json` — 6 scenarios


### graph_corr_001 (hard)

Chiller-1 condenser water return temperature rose 15% over 48 hours. Are the connected AHU-1 and Pump-CW-1 showing correlated anomalies in the same period?

- **Expected tools:** `anomaly_correlation`, `cypher_query`
- **Expected output contains:** `correlated`, `AHU-1`, `Pump-CW-1`, `DEPENDS_ON`
- **Requires graph:** True

### graph_corr_002 (medium)

Three sensors on different equipment in HVAC-North triggered high-temperature alerts within 10 minutes. Are these assets connected through SHARES_SYSTEM_WITH edges?

- **Expected tools:** `anomaly_correlation`, `cypher_query`
- **Expected output contains:** `SHARES_SYSTEM_WITH`, `temperature`, `HVAC-North`
- **Requires graph:** True

### graph_corr_003 (medium)

Vibration anomaly detected on Motor-CH1. Check if Chiller-1 (which Motor-CH1 DEPENDS_ON) and any equipment that SHARES_SYSTEM_WITH Chiller-1 has had anomalies in the last 7 days.

- **Expected tools:** `anomaly_correlation`, `cypher_query`
- **Expected output contains:** `vibration`, `DEPENDS_ON`, `SHARES_SYSTEM_WITH`, `Motor-CH1`
- **Requires graph:** True

### graph_corr_004 (hard)

Condenser water flow dropped 20% on cooling loop A (Chiller-1 and Chiller-2). Correlate sensor readings across all equipment on that loop including Pump-CW-1 and Pump-CW-2 to find the source.

- **Expected tools:** `anomaly_correlation`, `sensor_trend`
- **Expected output contains:** `condenser`, `flow`, `correlation`, `Chiller-1`, `Pump-CW-1`
- **Requires graph:** True

### graph_corr_005 (medium)

A work order reports unusual noise from AHU-3. Find all equipment sharing the same system via SHARES_SYSTEM_WITH (AHU-4) and check for concurrent anomalies.

- **Expected tools:** `anomaly_correlation`, `cypher_query`
- **Expected output contains:** `AHU-3`, `SHARES_SYSTEM_WITH`, `AHU-4`, `concurrent`
- **Requires graph:** True

### graph_corr_006 (hard)

Electrical anomaly detected at Utilities-East. Identify all equipment at that location and check for simultaneous sensor deviations to confirm propagation path through the DEPENDS_ON graph.

- **Expected tools:** `anomaly_correlation`, `impact_analysis`
- **Expected output contains:** `Utilities-East`, `DEPENDS_ON`, `propagation`, `simultaneous`
- **Requires graph:** True

## Failure Similarity

`failure_similarity.json` — 6 scenarios


### graph_sim_001 (medium)

Chiller-1 shows early signs of compressor overheating. Find the top 5 most similar historical failure modes using vector search on FMSR embeddings.

- **Expected tools:** `vector_search`
- **Expected output contains:** `Compressor Overheating`, `similarity_score`, `embedding`, `distance`
- **Requires graph:** True

### graph_sim_002 (medium)

A new anomaly 'intermittent high-frequency vibration with thermal cycling' was detected on Motor-CH1. Find failure modes with similar semantic embeddings across all equipment types.

- **Expected tools:** `vector_search`
- **Expected output contains:** `vibration`, `embedding`, `similarity_score`, `distance`
- **Requires graph:** True

### graph_sim_003 (hard)

Condenser water side fouling detected on Chiller-3. Retrieve similar past failure modes across all chillers and their associated work orders. What was the average repair cost?

- **Expected tools:** `vector_search`, `cypher_query`
- **Expected output contains:** `Condenser Water side fouling`, `similarity_score`, `work order`, `cost`
- **Requires graph:** True

### graph_sim_004 (medium)

An operator described a failure as 'pump cavitation with suction line blockage' on Pump-CW-1. Find semantically similar failure modes even if they use different terminology, such as 'Impeller erosion or cavitation damage'.

- **Expected tools:** `vector_search`
- **Expected output contains:** `cavitation`, `similarity_score`, `Impeller erosion`, `distance`
- **Requires graph:** True

### graph_sim_005 (hard)

Scale buildup detected on Boiler-1. Find similar failure modes across the entire boiler fleet (Boiler-1 through Boiler-4) and rank by severity-weighted similarity.

- **Expected tools:** `vector_search`, `cypher_query`
- **Expected output contains:** `Scale buildup`, `Boiler`, `severity`, `similarity_score`
- **Requires graph:** True

### graph_sim_006 (hard)

Motor-AHU1 experienced 'Stator winding insulation breakdown'. Which other motors (Motor-CH1, Motor-P1, Motor-BL1) had similar failure modes, and what spare parts were used in their repairs?

- **Expected tools:** `vector_search`, `cypher_query`
- **Expected output contains:** `Stator winding insulation breakdown`, `similarity_score`, `SparePart`, `USES_PART`
- **Requires graph:** True

## Criticality Analysis

`criticality_analysis.json` — 5 scenarios


### graph_crit_001 (medium)

Rank all equipment by graph-based criticality using PageRank on the DEPENDS_ON network. Which are the top 5 most critical assets?

- **Expected tools:** `criticality_ranking`
- **Expected output contains:** `PageRank`, `criticality`, `Chiller-1`, `DEPENDS_ON`
- **Requires graph:** True

### graph_crit_002 (hard)

Compare PageRank-based criticality scores with the static criticality_score property on Equipment nodes. Where do they disagree?

- **Expected tools:** `criticality_ranking`, `cypher_query`
- **Expected output contains:** `PageRank`, `criticality_score`, `disagree`, `comparison`
- **Requires graph:** True

### graph_crit_003 (hard)

Identify single points of failure: equipment with high in-degree on DEPENDS_ON that, if removed, would disconnect the dependency graph. Check Chiller-1 and Chiller-2 which have the most dependents.

- **Expected tools:** `criticality_ranking`, `cypher_query`
- **Expected output contains:** `single point of failure`, `in-degree`, `DEPENDS_ON`, `Chiller-1`
- **Requires graph:** True

### graph_crit_004 (medium)

Which Location contains the highest concentration of critical equipment (top quartile by PageRank)? Flag it for redundancy review. Consider HVAC-North, HVAC-South, Utilities-East, and Utilities-West.

- **Expected tools:** `criticality_ranking`, `cypher_query`
- **Expected output contains:** `Location`, `concentration`, `PageRank`, `redundancy`
- **Requires graph:** True

### graph_crit_005 (hard)

Using weakly connected components on the SHARES_SYSTEM_WITH edges, identify independent system clusters (e.g., Chiller-1/Chiller-2 cooling loop A, Boiler-1/Boiler-2). Which cluster has the highest aggregate criticality?

- **Expected tools:** `criticality_ranking`, `cypher_query`
- **Expected output contains:** `connected component`, `cluster`, `SHARES_SYSTEM_WITH`, `aggregate`
- **Requires graph:** True

## Maintenance Optimization

`maintenance_optimization.json` — 5 scenarios


### graph_maint_001 (hard)

Schedule preventive maintenance for the 4 chillers (Chiller-1 through Chiller-4) within the next available MaintenanceWindow, minimizing total downtime.

- **Expected tools:** `maintenance_scheduler`
- **Expected output contains:** `schedule`, `MaintenanceWindow`, `downtime`, `Chiller`
- **Requires graph:** True

### graph_maint_002 (hard)

There are open work orders and maintenance windows this month. Assign work orders to windows to minimize total cost while respecting crew_size constraints.

- **Expected tools:** `maintenance_scheduler`, `cypher_query`
- **Expected output contains:** `work order`, `window`, `cost`, `crew_size`, `constraint`
- **Requires graph:** True

### graph_maint_003 (medium)

Pump-CW-1 and Pump-HW-1 share a system (SHARES_SYSTEM_WITH). Can their maintenance be bundled into a single window to reduce repeated shutdowns?

- **Expected tools:** `maintenance_scheduler`, `cypher_query`
- **Expected output contains:** `bundle`, `SHARES_SYSTEM_WITH`, `Pump-CW-1`, `Pump-HW-1`, `window`
- **Requires graph:** True

### graph_maint_004 (hard)

Generate a Pareto-optimal set of maintenance schedules trading off total cost versus maximum equipment downtime for Q2 2026.

- **Expected tools:** `maintenance_scheduler`
- **Expected output contains:** `Pareto`, `cost`, `downtime`, `trade-off`
- **Requires graph:** True

### graph_maint_005 (medium)

A compressor gasket set spare part has a 14-day lead time and only 2 units in stock. Which pending work orders for chillers require this part, and can they all be completed before stock runs out?

- **Expected tools:** `maintenance_scheduler`, `cypher_query`
- **Expected output contains:** `lead time`, `stock`, `SparePart`, `USES_PART`
- **Requires graph:** True

## Root Cause Analysis

`root_cause_analysis.json` — 5 scenarios


### graph_rca_001 (hard)

A work order reports Chiller-2 tripped on high head pressure. Trace backward through sensor readings (Chiller-2-CondWaterRetTemp, Chiller-2-Efficiency), anomalies, and upstream equipment to find the root cause.

- **Expected tools:** `root_cause_trace`, `cypher_query`
- **Expected output contains:** `root cause`, `Chiller-2`, `DEPENDS_ON`, `trace`
- **Requires graph:** True

### graph_rca_002 (hard)

AHU-1 and AHU-2 in HVAC-North reported low supply air temperature within 2 hours. Trace the event chain to determine if the root cause is a shared chilled water supply issue from Chiller-1 or Chiller-2.

- **Expected tools:** `root_cause_trace`, `anomaly_correlation`
- **Expected output contains:** `AHU-1`, `AHU-2`, `Chiller`, `DEPENDS_ON`, `event chain`
- **Requires graph:** True

### graph_rca_003 (medium)

A vibration anomaly on Motor-P1 triggered a work order. Walk back the DETECTED_ANOMALY and TRIGGERED edges through Motor-P1's sensors to reconstruct the full event timeline.

- **Expected tools:** `root_cause_trace`, `cypher_query`
- **Expected output contains:** `DETECTED_ANOMALY`, `TRIGGERED`, `timeline`, `Motor-P1`
- **Requires graph:** True

### graph_rca_004 (hard)

Boiler-3 experienced burner flame instability. Traverse the DEPENDS_ON graph and SHARES_SYSTEM_WITH (Boiler-4) to check if related equipment had prior anomalies.

- **Expected tools:** `root_cause_trace`, `cypher_query`
- **Expected output contains:** `Burner flame instability`, `Boiler-3`, `DEPENDS_ON`, `SHARES_SYSTEM_WITH`, `anomaly`
- **Requires graph:** True

### graph_rca_005 (hard)

Over the past month, multiple work orders have been created for Motor-AHU1 for 'Stator winding insulation breakdown'. Is this a recurring issue? What was the root cause in each case and are repairs addressing it?

- **Expected tools:** `root_cause_trace`, `cypher_query`, `vector_search`
- **Expected output contains:** `recurring`, `Stator winding insulation breakdown`, `Motor-AHU1`, `root cause`
- **Requires graph:** True

## Temporal Pattern

`temporal_pattern.json` — 5 scenarios


### graph_temp_001 (medium)

Calculate the Mean Time Between Failures (MTBF) for all Chiller equipment (Chiller-1 through Chiller-4) over the past 24 months. Which chiller has the worst MTBF?

- **Expected tools:** `cypher_query`
- **Expected output contains:** `MTBF`, `Chiller`, `24 months`, `worst`
- **Requires graph:** True

### graph_temp_002 (hard)

Analyze seasonal failure patterns: do Chiller failures peak in summer and Boiler failures peak in winter? Show monthly failure counts over 2 years for all 4 chillers and 4 boilers.

- **Expected tools:** `cypher_query`, `sensor_trend`
- **Expected output contains:** `seasonal`, `summer`, `winter`, `monthly`, `Chiller`, `Boiler`
- **Requires graph:** True

### graph_temp_003 (medium)

Sensor Motor-CH1-BearingTemp on Motor-CH1 has been trending upward over 30 days. Extrapolate when it will breach the max_threshold based on the trend.

- **Expected tools:** `sensor_trend`
- **Expected output contains:** `trend`, `Motor-CH1`, `temperature`, `threshold`, `extrapolate`
- **Requires graph:** True

### graph_temp_004 (hard)

Identify equipment where the interval between consecutive work orders is decreasing, indicating accelerating degradation. Check across all equipment at MainPlant.

- **Expected tools:** `cypher_query`
- **Expected output contains:** `interval`, `decreasing`, `degradation`, `work order`
- **Requires graph:** True

### graph_temp_005 (hard)

For the top 5 most failure-prone equipment across Chillers, AHUs, Pumps, Motors, and Boilers, plot the cumulative failure count over time. Do any exhibit bathtub curve behavior (early failures, steady state, wear-out)?

- **Expected tools:** `cypher_query`, `sensor_trend`
- **Expected output contains:** `cumulative`, `bathtub curve`, `wear-out`, `failure-prone`
- **Requires graph:** True
