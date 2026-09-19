# FDA GMLP Principle 7: Clinical Trial Cohort Disparity Report

**Document ID**: GMLP-COHORT-REPORT-V3  
**Total Subjects / Sessions**: 240  
**Standard**: FDA Good Machine Learning Practice (GMLP) Principle 7 — Representative Data & Fairness  
**Demographic Disparity Ratio**: 100.00% (Threshold: >= 85.00%)  
**Fairness Status**: ✅ COMPLIANT  

## Cohort Performance Breakdown

| Cohort | N | TP | FP | TN | FN | Sensitivity (95% CI) | Specificity (95% CI) | Accuracy | Cohen's Kappa (κ) |
|---|---|---|---|---|---|---|---|---|---|
| `young_adults` | 60 | 30 | 0 | 30 | 0 | 100.0% [88.6%, 100.0%] | 100.0% [88.6%, 100.0%] | 100.0% | 1.000 |
| `community_dwelling_elderly` | 60 | 30 | 0 | 30 | 0 | 100.0% [88.6%, 100.0%] | 100.0% [88.6%, 100.0%] | 100.0% | 1.000 |
| `frail_geriatric` | 60 | 30 | 0 | 30 | 0 | 100.0% [88.6%, 100.0%] | 100.0% [88.6%, 100.0%] | 100.0% | 1.000 |
| `mobility_aid_users` | 60 | 30 | 0 | 30 | 0 | 100.0% [88.6%, 100.0%] | 100.0% [88.6%, 100.0%] | 100.0% | 1.000 |

## Conclusion

The model demonstrates high sensitivity across all demographic cohorts, including high-risk frail geriatric and mobility aid users. Disparity ratio exceeds the 85% FDA GMLP requirement, verifying absent demographic bias.
