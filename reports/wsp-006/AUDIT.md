# Audit report

```json
{
  "experiment": "wsp-006",
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
    "iterations_audited": 19,
    "replays": [
      {
        "id": "iter001",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter002",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter003",
        "decision": "accepted",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter004",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter005",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter006",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter007",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter008",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter009",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter010",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter011",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter012",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter013",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter014",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter015",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter016",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter017",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter018",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      },
      {
        "id": "iter019",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true,
        "reproduced_up_to_signed_zero_only": false
      }
    ],
    "warnings": [],
    "rejected_preserved": true,
    "no_test_metrics_in_iteration_history": true,
    "problems": []
  },
  "weight_updates": {
    "iterations_replayed": 19,
    "problems": [],
    "trail": [
      {
        "id": "iter001",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter000_initial.pt",
        "operations": [
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "c789fb2f7b340708f648d89c50bf154ff33e75bf6345519fd3291eb5875deea4",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter002",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter000_initial.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv2.bias",
            "(slice(None, None, None),)",
            32
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "conv2.bias": 32,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "f26576b6a4aa1a46ea876bca2fd7333467fe948999cc1300bd11854c4f0fabc0",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter003",
        "decision": "accepted",
        "parent": "checkpoints/wsp-006/iter000_initial.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv2.weight",
            "(24, 12, slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(25, 13, slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(26, 14, slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(27, 15, slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(28, 12, slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(29, 13, slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(30, 14, slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(31, 15, slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "conv2.weight": 72,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "f10262fca1090d3a37f8e623be2f36f81743d8dd35244e3d33b713dec6194da0",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter004",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "replace_parameter",
            "conv3.weight",
            "all",
            1296
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "conv3.weight": 1296,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "e3578cfb6b9edd44d2c3d0b4e0575204445c868bb17c226cf2bb01284ac163f8",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter005",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "04b159c463e2232e318f8ba04bcf4e8ca2f02e7e80caef125b9989dcfdef4641",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter006",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "0b89b9fbb74bcceffd4fb3baaea14b223ec3d6e3b23e6327da6aad470ae60bf7",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter007",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "3a47d8784a4f39e5a1287622e4b531f11b927947e24527438c982e0e7b685d28",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter008",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv3.bias",
            "(slice(None, None, None),)",
            32
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "conv3.bias": 32,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "7e72398d76cab50d10bea5e699b0874cd72bc4f6d912d1a0a3c21670a72182c4",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter009",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "a29c0b1044ffebb3646c6b6860753ae53439fea98f16b0c2d6128509caf2a039",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter010",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "19a1e294e245c0f964bc734c03a14c98c441a10936611d1c920e2c45a411215d",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter011",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv1.weight",
            "(slice(12, 16, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            275
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "conv1.weight": 250,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "7e038539b8c871da95bd7f0c57ba5fa89de0942cc5b625f23120c42198c08db2",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter012",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv1.weight",
            "(slice(8, 12, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            300
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
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
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "a0c1f3dfb39f5f0bb3ea982326fab10d05169b08496269e4248c5bec166859c3",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter013",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv1.weight",
            "(8, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            75
          ],
          [
            "modify_parameter_group",
            "conv1.weight",
            "(9, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            75
          ],
          [
            "modify_parameter_group",
            "conv1.weight",
            "(10, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            75
          ],
          [
            "modify_parameter_group",
            "conv1.weight",
            "(11, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            75
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
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
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "ccbaf35a40b8846e2429d4e645495ac4bc99b4da5e935c1ed561dd08688a3ef6",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter014",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv2.weight",
            "(slice(16, 24, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            24
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "conv2.weight": 24,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "e22641a3a48068b2ebe1719d03a3e70ee0e75ca7426d15560cdd072c210bd01d",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter015",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv3.weight",
            "(slice(24, 32, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            576
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "conv3.weight": 576,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "472887ed8567f49f47a97930257b50ca455acaa58319d60fabba777a0537829b",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter016",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "d76aeff38c92c3e7237b5a8dae1fa0e086bf861d86d526829bab92f73537ba55",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter017",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "3d7f98dabbbcf3d0197af908226fa56fce91a4870ecbd98a14b75c9f85c80584",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter018",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv3.weight",
            "(24, slice(24, 32, None), slice(None, None, None), slice(None, None, None))",
            72
          ],
          [
            "modify_parameter_group",
            "conv3.weight",
            "(25, slice(24, 32, None), slice(None, None, None), slice(None, None, None))",
            72
          ],
          [
            "modify_parameter_group",
            "conv3.weight",
            "(26, slice(24, 32, None), slice(None, None, None), slice(None, None, None))",
            72
          ],
          [
            "modify_parameter_group",
            "conv3.weight",
            "(27, slice(24, 32, None), slice(None, None, None), slice(None, None, None))",
            72
          ],
          [
            "modify_parameter_group",
            "conv3.weight",
            "(28, slice(24, 32, None), slice(None, None, None), slice(None, None, None))",
            72
          ],
          [
            "modify_parameter_group",
            "conv3.weight",
            "(29, slice(24, 32, None), slice(None, None, None), slice(None, None, None))",
            72
          ],
          [
            "modify_parameter_group",
            "conv3.weight",
            "(30, slice(24, 32, None), slice(None, None, None), slice(None, None, None))",
            72
          ],
          [
            "modify_parameter_group",
            "conv3.weight",
            "(31, slice(24, 32, None), slice(None, None, None), slice(None, None, None))",
            72
          ],
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "conv3.weight": 576,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "87fee252d6323d314e20a9157506786981b8ea6a341ffb8e4c86f1125d1c4e82",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter019",
        "decision": "rejected",
        "parent": "checkpoints/wsp-006/iter003.pt",
        "operations": [
          [
            "replace_parameter",
            "head.weight",
            "all",
            512
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            1
          ]
        ],
        "changed_per_tensor": {
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "04b159c463e2232e318f8ba04bcf4e8ca2f02e7e80caef125b9989dcfdef4641",
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
      "train": 12,
      "val": 4,
      "test": 4
    },
    "previous_experiment_artifacts_changed": [],
    "wsp001_artifacts_changed": []
  },
  "passed": true
}
```
