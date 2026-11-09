# Audit report

```json
{
  "experiment": "wsp-004",
  "static_findings": [],
  "mutation_findings": [],
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
    "iterations_audited": 3,
    "replays": [
      {
        "id": "iter001",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter002",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter003",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      }
    ],
    "rejected_preserved": true,
    "no_test_metrics_in_iteration_history": true,
    "problems": []
  },
  "weight_updates": {
    "iterations_replayed": 3,
    "problems": [],
    "trail": [
      {
        "id": "iter001",
        "decision": "rejected",
        "parent": "checkpoints/wsp-004/iter000_initial.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv2.weight",
            "(slice(48, 96, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            20736
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            1536
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "conv2.weight": 20736,
          "head.weight": 1536,
          "head.bias": 1
        },
        "candidate_sha256": "917a9669be50c410a6484239c4e5c89c425831b487831fd3f3081211b27f797c",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter002",
        "decision": "rejected",
        "parent": "checkpoints/wsp-004/iter000_initial.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv2.weight",
            "(slice(48, 96, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            520
          ],
          [
            "modify_parameter_group",
            "conv2.bias",
            "(slice(48, 96, None),)",
            48
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            1536
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "conv2.weight": 520,
          "conv2.bias": 48,
          "head.weight": 1536,
          "head.bias": 1
        },
        "candidate_sha256": "715ddefe63f17032119030a24fc122bb128952ff7f6ba178256e65e2ef2df783",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter003",
        "decision": "rejected",
        "parent": "checkpoints/wsp-004/iter000_initial.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv1.weight",
            "(44, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            75
          ],
          [
            "modify_parameter_group",
            "conv1.weight",
            "(45, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            75
          ],
          [
            "modify_parameter_group",
            "conv1.weight",
            "(46, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            75
          ],
          [
            "modify_parameter_group",
            "conv1.weight",
            "(47, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            75
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            1536
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "conv1.weight": 300,
          "head.weight": 1536,
          "head.bias": 1
        },
        "candidate_sha256": "e58e6b9d6621d54391c05095caf1b0138578cf4bf2790e56adf4279c1b14b861",
        "replay_matches": true,
        "all_changes_logged": true
      }
    ]
  },
  "wsp002": {
    "test_sealed_mode": "0o0",
    "test_cache_absent_before_final": true,
    "train_val_images_unchanged": true,
    "test_split_counts": {
      "train": 8000,
      "val": 3000,
      "test": 3884
    },
    "previous_experiment_artifacts_changed": [],
    "wsp001_artifacts_changed": []
  },
  "passed": true
}
```
