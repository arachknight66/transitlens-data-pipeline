# Dataset Validation Report

Generated on: 2026-06-26T13:38:07.875302+00:00
Overall Status: 🟢 PASS

## 1. Summary of Evaluated Targets

- **Evaluable Targets (Actual Time-Series Light Curves)**: 7
- **Catalog-Only Targets (Metadata Rows)**: 15829
- **Target Disjointness**: ✅ YES

### Target Distributions by Class
| Class Label | Target Count |
| :--- | :--- |
| `eclipsing_binary` | 1 |
| `stellar_variability_or_other` | 1 |
| `blend_contamination` | 0 |
| `exoplanet_transit` | 5 |

### Target Distributions by Source
| Source | Target Count |
| :--- | :--- |
| `synthetic` | 3 |
| `real_tess` | 4 |

### Target Distributions by Evidence Level
| Evidence Level | Target Count |
| :--- | :--- |
| `synthetic` | 3 |
| `real_tess` | 4 |

### Split Sizes
- **Train**: 4 targets
- **Val**: 2 targets
- **Test**: 1 targets

---

## 2. Issues Encountered

### Errors (0)
- No errors found.

### Warnings (1)
- ⚠️ Evaluable target count is extremely small (7 targets). The dataset is 'Partially Complete / Framework Ready', which is insufficient for a Strong 95+ score. Phase 2 requires expanding to at least 700 real evaluable targets.

---

## 3. Evaluator's Notes

> [!WARNING]
> This dataset has been evaluated as **Partially Complete / Framework Ready**. 
> While the NPZ file layout, schema, metadata mapping, and target disjointness conform to the scientific contracts, the total count of evaluable targets is **7** which is insufficient for strong phase evaluation scoring.
> 
> To achieve 95+ score, Phase 2 MUST download and ingest more light curves to reach 700+ evaluable light curves.
