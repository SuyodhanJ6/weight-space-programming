# Audit report

```json
{
  "static_findings": [],
  "runtime": {
    "grad_mode_disabled": true,
    "optimizer_instances_alive": 0,
    "params_require_grad": [],
    "activations_with_graph": [],
    "splits_read_by_score_model": [
      "train",
      "val"
    ]
  },
  "consistency": {
    "baseline_hash_ok": true,
    "baseline_rebuilt_from_code_matches": true,
    "accepted_hash_ok": true,
    "baseline_file_read_only": true,
    "iterations_audited": 9,
    "replays": [
      {
        "id": "iter001",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter002",
        "decision": "accepted",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter003",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter004",
        "decision": "accepted",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter005",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter006",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter007",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter008",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter009",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      }
    ],
    "rejected_preserved": true,
    "no_test_metrics_in_iteration_history": true,
    "problems": []
  },
  "passed": true
}
```
