# TASK 2: NETWORK DIAGRAM (AON) AND SLACK ANALYSIS

## Activity Network Parameters

| Activity ID | Activity Name | Duration (Days) | Predecessors | ES | EF | LS | LF | Total Slack | Free Slack | Critical? |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1.1 | Team Formation | 2 | \- | 0 | 2 | 0 | 2 | 0 | 0 | Yes |
| 1.2 | Charter Approval | 5 | 1.1 | 2 | 7 | 2 | 7 | 0 | 0 | Yes |
| 2.1 | Requirements Analysis | 7 | 1.2 | 7 | 14 | 7 | 14 | 0 | 0 | Yes |
| 2.2 | Scope Definition | 7 | 1.2 | 7 | 14 | 7 | 14 | 0 | 0 | Yes\* |
| 2.3 | Gantt Chart Creation | 14 | 2.1, 2.2 | 14 | 28 | 14 | 28 | 0 | 0 | Yes |
| 3.1 | Data Schema Design | 7 | 2.3 | 28 | 35 | 28 | 35 | 0 | 0 | Yes |
| 3.2 | API Specification | 7 | 3.1 | 35 | 42 | 35 | 42 | 0 | 0 | Yes |
| 3.3 | Agent Logic Design | 7 | 3.2 | 42 | 49 | 42 | 49 | 0 | 0 | Yes |
| 4.1 | Data Generator | 7 | 3.1 | 35 | 42 | 42 | 49 | 7 | 0 | No |
| 4.2 | Agent Core Development | 14 | 4.1, 3.3 | 49 | 63 | 49 | 63 | 0 | 0 | Yes |
| 4.3 | API & Dashboard | 14 | 3.2 | 42 | 56 | 49 | 63 | 7 | 7 | No |
| 5.1 | Unit Testing | 5 | 4.2, 4.3 | 63 | 68 | 63 | 68 | 0 | 0 | Yes |
| 5.2 | Integration Testing | 5 | 5.1 | 68 | 73 | 68 | 73 | 0 | 0 | Yes |
| 5.3 | A/B Testing | 4 | 5.2 | 73 | 77 | 73 | 77 | 0 | 0 | Yes |
| 6.1 | Final Report | 4 | 5.3 | 77 | 81 | 77 | 81 | 0 | 0 | Yes |
| 6.2 | Presentation | 3 | 6.1 | 81 | 84 | 81 | 84 | 0 | 0 | Yes |

\*Both 2.1 and 2.2 are on the critical path as they are parallel predecessors to 2.3

## Slack Analysis Summary

**Critical Path Activities (Total Slack \= 0):** 1.1 → 1.2 → 2.1/2.2 → 2.3 → 3.1 → 3.2 → 3.3 → 4.2 → 5.1 → 5.2 → 5.3 → 6.1 → 6.2

**Non-Critical Activities:**

- **4.1 (Data Generator)**: Total Slack \= 7 days, Free Slack \= 0 days  
    
  - Can be delayed up to 7 days without impacting project completion  
  - Cannot be delayed without delaying successor 4.2 (Free Slack \= 0\)


- **4.3 (API & Dashboard)**: Total Slack \= 7 days, Free Slack \= 7 days  
    
  - Can be delayed up to 7 days without impacting project completion  
  - Can be delayed 7 days without affecting the immediate successor 5.1 (Free Slack \= 7\)

## Parallel Paths Analysis

**Path 1 (Critical):** 1.1 → 1.2 → 2.1 → 2.3 → 3.1 → 3.2 → 3.3 → 4.2 → 5.1 → 5.2 → 5.3 → 6.1 → 6.2 Duration: 84 days

**Path 2 (Near-Critical):** 1.1 → 1.2 → 2.2 → 2.3 → 3.1 → 3.2 → 3.3 → 4.2 → 5.1 → 5.2 → 5.3 → 6.1 → 6.2 Duration: 84 days (both 2.1 and 2.2 merge at 2.3)

**Path 3 (Sub-Critical):** 3.1 → 4.1 → 4.2 → 5.1 → 5.2 → 5.3 → 6.1 → 6.2 Duration: 49 days from day 28 (7 days slack due to 4.1 starting later)

**Path 4 (Sub-Critical):** 3.2 → 4.3 → 5.1 → 5.2 → 5.3 → 6.1 → 6.2 Duration: 40 days from day 35 (7 days slack available in 4.3)

## Key Findings

1. **Project Duration**: 84 days total (approximately 12 weeks)  
     
2. **Critical Activities**: 14 out of 16 activities are on the critical path \- the project has a very tight schedule  
     
3. **Merge Points**:  
     
   - Day 14: Activities 2.1 and 2.2 merge into 2.3  
   - Day 49: Activities 4.1 and 3.3 merge into 4.2  
   - Day 63: Activities 4.2 and 4.3 merge into 5.1

   

4. **Parallel Execution Windows**:  
     
   - Days 7-14: Requirements Analysis (2.1) parallel with Scope Definition (2.2)  
   - Days 35-49: Data Generator (4.1) can run in parallel with API work  
   - Days 42-56: API & Dashboard (4.3) parallel with Agent Core path

   

5. **Risk Concentration**:  
     
   - 87.5% of activities are critical \- minimal schedule buffer  
   - Only 7 days of float are available in the implementation phase  
   - Any delay in critical activities directly impacts project completion

## Calculation Methodology

**Forward Pass (ES, EF):**

- ES \= Maximum EF of all predecessors  
- EF \= ES \+ Duration

**Backwards Pass (LF, LS):**

- LF \= Minimum LS of all successors  
- LS \= LF \- Duration

**Slack Calculations:**

- Total Slack (TS) \= LF \- EF \= LS \- ES  
- Free Slack (FS) \= Minimum ES of successors \- EF

