# Audit report

```json
{
  "experiment": "wsp-005",
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
        "decision": "rejected",
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
        "reproduced_from_parent_and_diff": false,
        "reproduced_up_to_signed_zero_only": true
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
    "warnings": [
      "iter011: reproduces exactly except sign-of-zero (-0.0 vs +0.0) entries omitted from the logged diff (value-equal; logger made bit-exact afterwards)"
    ],
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
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "361be9b7aff41228518e7997d3edc3d5a0ed932a55e4b0ccd3e8176119b4cc97",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter002",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "e7e7ba61a507564213d8a5d8a425ee75abc81401507a377c0965f52364867593",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter003",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "0610c564a2d48d232bd3384ed64cc4bdfcc55faeb24e698d15a443b698980449",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter004",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "b0453d0fbbebd43dff65c4e60b7bb41cf857c5b09a8dd61a962b2b9bc95d8eac",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter005",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "5a4a5844b19b866e0d16c95dc7fdff06729779f4a6c792b1f8d152eabffc684f",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter006",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "16c946fe780be5300ff45b563809ab9e7ac076d70892b2a1714ecafee125a80c",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter007",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "8997ebaba51943b37545bd8c9dab55f4713dd126e81a10c5c9cbac1d545fc263",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter008",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "232c4c6d2a336fb343185b045cdbb82879ccbd20d8800a7cf780cc50be7b72f5",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter009",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "210822e44efd780071a910e1b2beedb383f033e6dd0f43dbbd24b77d3e28d829",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter010",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "08ef01b3e0d61ff44870bde12341de269d95ec26581f33633ef060dcd4ec4ab8",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter011",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
            0
          ],
          [
            "replace_parameter",
            "head.bias",
            "all",
            0
          ]
        ],
        "changed_per_tensor": {
          "conv1.weight": 250
        },
        "candidate_sha256": "a8e21afa20e28667ed1336cbdf75d7d63b65edcfe106bf92bc94cbf30d0cc995",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter012",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "86da344c8f7a8165e1a42b474a8e5ec2726e7800b277660d7769ba12cbebd3d6",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter013",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "007ba70733d8f253088c86aca515e63bd5bcd3053f331f2bfd3f86d80695fa3c",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter014",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "69ae26a170a06ee92230679ef2d4784333caf4c548fd5a96abcc2aa56d8ba722",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter015",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "bb53941e0b39941f891b51bfe6f570d8b9f6cc31d6c0d7d3a1db2264238ac6dc",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter016",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "124444be57c9fef932054c52311bc6f04f2ab009c0c1d4a371dd19e035bda34a",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter017",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "659ec22dc9cbc4f40feb34b14ee9ca5fcddf0bc6613f24adf767c69a9b5bd67c",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter018",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "ab6d79cf68302243efb22febd5e5e6662107c49e1b9054a390e36e3cb8b13c07",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter019",
        "decision": "rejected",
        "parent": "checkpoints/wsp-005/iter000_initial.pt",
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
        "candidate_sha256": "5a4a5844b19b866e0d16c95dc7fdff06729779f4a6c792b1f8d152eabffc684f",
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
