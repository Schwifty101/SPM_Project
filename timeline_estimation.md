# TASK 1: TIMELINE ESTIMATION AND SCHEDULING OF WBS ITEMS

## Timeline Estimation Table

| Task ID | Task Name | Duration (Days) | Start Date | Finish Date | Predecessors | Responsible Team Member | Justification |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1.0 | Project Initiation | 7 | 22-Sep-2025 | 28-Sep-2025 | \- | Soban Ahmed | Initial setup requires stakeholder alignment, charter drafting, and an approval cycle |
| 1.1 | Team Formation | 2 | 22-Sep-2025 | 23-Sep-2025 | \- | Soban Ahmed | Role assignment based on technical skills |
| 1.2 | Charter Approval | 5 | 24-Sep-2025 | 28-Sep-2025 | 1.1 | Soban Ahmed | Supervisor review and sign-off process |
| 2.0 | Planning Phase | 21 | 29-Sep-2025 | 19-Oct-2025 | 1.0 | All Team | Detailed planning requires parallel work streams |
| 2.1 | Requirements Analysis | 7 | 29-Sep-2025 | 5-Oct-2025 | 1.2 | Abdul Moeiz | Data source identification, incentive rule definition, stakeholder needs assessment |
| 2.2 | Scope Definition | 7 | 29-Sep-2025 | 5-Oct-2025 | 1.2 | Soban Ahmed | Parallel with 2.1, boundary setting and success criteria establishment |
| 2.3 | Gantt Chart Creation | 14 | 6-Oct-2025 | 19-Oct-2025 | 2.1, 2.2 | Muhammad Uzair | Dependencies mapping and timeline validation require completed requirements |
| 3.0 | Design & Architecture | 21 | 20-Oct-2025 | 9-Nov-2025 | 2.0 | All Team | Complex system design with multiple integrated components |
| 3.1 | Data Schema Design | 7 | 20-Oct-2025 | 26-Oct-2025 | 2.3 | Abdul Moeiz | Synthetic dataset structure, customer behaviour attributes, transaction logs |
| 3.2 | API Specification | 7 | 27-Oct-2025 | 2-Nov-2025 | 3.1 | Muhammad Uzair | RESTful endpoints, request/response formats, and authentication protocols |
| 3.3 | Agent Logic Design | 7 | 3-Nov-2025 | 9-Nov-2025 | 3.2 | Soban Ahmed | Reward selection algorithms, optimisation logic, and decision trees |
| 4.0 | Implementation | 28 | 10-Nov-2025 | 7-Dec-2025 | 3.0 | All Team | Core development phase with parallel coding streams |
| 4.1 | Data Generator | 7 | 10-Nov-2025 | 16-Nov-2025 | 3.1 | Abdul Moeiz | Python script for synthetic data creation with realistic distributions |
| 4.2 | Agent Core Development | 14 | 17-Nov-2025 | 30-Nov-2025 | 4.1, 3.3 | Soban Ahmed, Uzair | ML model, adaptive logic, bandit algorithms \- the most complex component |
| 4.3 | API & Dashboard | 14 | 17-Nov-2025 | 30-Nov-2025 | 3.2 | Muhammad Uzair | Parallel with 4.2; Flask/FastAPI backend, React dashboard |
| 5.0 | Testing & Validation | 14 | 1-Dec-2025 | 14-Dec-2025 | 4.0 | All Team | Comprehensive testing with simulation runs |
| 5.1 | Unit Testing | 5 | 1-Dec-2025 | 5-Dec-2025 | 4.2, 4.3 | Muhammad Uzair | Individual module validation, edge case testing |
| 5.2 | Integration Testing | 5 | 6-Dec-2025 | 10-Dec-2025 | 5.1 | Abdul Moeiz | End-to-end workflow, API connectivity, database interaction |
| 5.3 | A/B Testing | 4 | 11-Dec-2025 | 14-Dec-2025 | 5.2 | Soban Ahmed | Simulate control vs. optimised incentive scenarios, measure uplift |
| 6.0 | Documentation & Closing | 7 | 15-Dec-2025 | 21-Dec-2025 | 5.0 | All Team | Final deliverables compilation |
| 6.1 | Final Report | 4 | 15-Dec-2025 | 18-Dec-2025 | 5.3 | Abdul Moeiz, Soban | Technical documentation, results analysis, lessons learned |
| 6.2 | Presentation | 3 | 19-Dec-2025 | 21-Dec-2025 | 6.1 | All Team | Demo preparation, slide deck, rehearsal |

## Duration Justification for Major Work Packages

**Planning Phase (21 days)**: Requirements gathering for AI-driven systems requires extensive data source identification, stakeholder interviews, and rule definition. Parallel work streams allow requirements analysis and scope definition to occur concurrently, followed by detailed Gantt chart creation that depends on both.

**Design & Architecture (21 days)**: A Sequential design approach is necessary. Data schema must be finalised before API specification (7 days each). Agent logic design builds on the API spec. Three-week duration accounts for design iterations and peer reviews.

**Implementation (28 days)**: Longest phase due to coding complexity. Data generator (7 days) creates a foundation. Agent core and dashboard run parallel for 14 days \- agent core involves ML model training, hyperparameter tuning, and adaptive logic implementation. Dashboard requires UI/UX design and API integration.

**Testing & Validation (14 days)**: Sequential testing layers. Unit testing validates individual components (5 days), integration testing ensures system cohesion (5 days), and A/B testing simulates real-world scenarios and requires sufficient run time for statistical significance (4 days).

**Documentation (7 days)**: Final report compilation (4 days) includes technical documentation, code comments, architecture diagrams, test results, and lessons learned. Presentation preparation (3 days) overlaps with final report review.

## Critical Path Analysis Preview

Critical Path: 1.2 → 2.1 → 2.3 → 3.1 → 3.2 → 3.3 → 4.1 → 4.2 → 5.1 → 5.2 → 5.3 → 6.1 → 6.2

Total Project Duration: 91 days (Sep 22 \- Dec 21, 2025\)  
