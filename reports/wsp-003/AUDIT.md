# Audit report

```json
{
  "experiment": "wsp-003",
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
    "iterations_audited": 21,
    "replays": [
      {
        "id": "iter001",
        "decision": "accepted",
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
      },
      {
        "id": "iter004",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter005",
        "decision": "accepted",
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
        "decision": "accepted",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter010",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter011",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter012",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter013",
        "decision": "accepted",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter014",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter015",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter016",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter017",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter018",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter019",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter020",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      },
      {
        "id": "iter021",
        "decision": "rejected",
        "reproduced_from_parent_and_diff": true
      }
    ],
    "rejected_preserved": true,
    "no_test_metrics_in_iteration_history": true,
    "problems": []
  },
  "weight_updates": {
    "iterations_replayed": 21,
    "problems": [],
    "trail": [
      {
        "id": "iter001",
        "decision": "accepted",
        "parent": "checkpoints/wsp-003/iter000_initial.pt",
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
        "candidate_sha256": "abb9245aef6363319d3ec9c554063a519671f35b2bf96a0792cff1f40321b4cd",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter002",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter001.pt",
        "operations": [
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            412
          ],
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
          "conv2.weight": 412,
          "conv3.weight": 1296,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "db21b13a3207b2b7caefdca60d78be151e6d2f33b6ee02f17afed26649500d30",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter003",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter001.pt",
        "operations": [
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            412
          ],
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
          "conv2.weight": 412,
          "conv3.weight": 1296,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "cf8ccbf62174a306e477e7274bc06f92d32286bd1dc212f7e6a83c1cbaaf75d0",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter004",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter001.pt",
        "operations": [
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            420
          ],
          [
            "replace_parameter",
            "conv2.bias",
            "all",
            16
          ],
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
          "conv2.weight": 420,
          "conv2.bias": 16,
          "conv3.weight": 1296,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "a6624abfd2edcdeef8703f6d7a9eb65b53bfe48d75e6311b98c1d1ff7ea72d47",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter005",
        "decision": "accepted",
        "parent": "checkpoints/wsp-003/iter001.pt",
        "operations": [
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            390
          ],
          [
            "replace_parameter",
            "conv2.bias",
            "all",
            32
          ],
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
          "conv2.weight": 390,
          "conv2.bias": 32,
          "conv3.weight": 1296,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "57ede7f490025dab328b29aa22992232a2966908e109e71cb6fad94d46513b10",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter006",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter005.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv2.weight",
            "(3, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            4
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(4, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            2
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(5, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            4
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(11, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            4
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(12, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            2
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(13, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            2
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(19, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            4
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(20, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            4
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(21, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            4
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(27, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            2
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(28, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            2
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(29, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            2
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
          "conv2.weight": 36,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "e92291e631b34f5d8a8196546062f8b4b52603f79d6c12ffbaa61dffdcb61600",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter007",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter005.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv2.weight",
            "(0, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(1, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(2, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(3, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(4, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(5, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(6, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(7, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(8, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(9, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(10, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(11, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(12, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(13, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(14, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(15, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(16, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(17, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(18, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(19, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(20, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(21, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(22, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(23, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(24, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(25, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(26, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(27, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(28, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(29, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(30, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            9
          ],
          [
            "modify_parameter_group",
            "conv2.weight",
            "(31, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
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
          "conv2.weight": 288,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "c85375997caa81eed5d9bdd2e114f1eb7cf5a9c99d032b91522fdf15a6d64881",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter008",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter005.pt",
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
        "candidate_sha256": "57acfc4721f6e25ed858544c31ff5337141bd74154604b38d3c941737306a452",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter009",
        "decision": "accepted",
        "parent": "checkpoints/wsp-003/iter005.pt",
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
        "candidate_sha256": "84bcc3fab139c91548bbb8e2b3f219f676be70ef6f0e5ae4326e081a53211d50",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter010",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter009.pt",
        "operations": [
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            119
          ],
          [
            "replace_parameter",
            "conv2.bias",
            "all",
            16
          ],
          [
            "replace_parameter",
            "conv3.weight",
            "all",
            0
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
          "conv2.weight": 119,
          "conv2.bias": 16,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "fc4bee02073482da2704357e0219969d8b21531dad7e11a2ac825ef6cad35d4b",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter011",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter009.pt",
        "operations": [
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            206
          ],
          [
            "replace_parameter",
            "conv3.weight",
            "all",
            0
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
          "conv2.weight": 206,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "db7129f71237701254df419c758445e8d52cd0b21a006ffaff85c2714362c4e1",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter012",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter009.pt",
        "operations": [
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            206
          ],
          [
            "replace_parameter",
            "conv3.weight",
            "all",
            144
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
          "conv2.weight": 206,
          "conv3.weight": 144,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "1b9a3f93be2c457a699dd706e51621220fa5b2b25b9675eb29aec35f0c3e104a",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter013",
        "decision": "accepted",
        "parent": "checkpoints/wsp-003/iter009.pt",
        "operations": [
          [
            "replace_parameter",
            "conv2.bias",
            "all",
            32
          ],
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            206
          ],
          [
            "replace_parameter",
            "conv3.weight",
            "all",
            0
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
          "conv2.weight": 206,
          "conv2.bias": 32,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "8ec523ac2ba2f984eaea9e9385d5911ead30ba5c92724954140dd30e02e58a57",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter014",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter013.pt",
        "operations": [
          [
            "replace_parameter",
            "conv2.bias",
            "all",
            0
          ],
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            0
          ],
          [
            "replace_parameter",
            "conv3.weight",
            "all",
            144
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
          "conv3.weight": 144,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "1622e629c5c31df7a534594de8630899918574858cc2dfdadca8b620639361c6",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter015",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter013.pt",
        "operations": [
          [
            "modify_parameter_group",
            "conv1.weight",
            "(1, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            75
          ],
          [
            "modify_parameter_group",
            "conv1.weight",
            "(3, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            75
          ],
          [
            "modify_parameter_group",
            "conv1.weight",
            "(5, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
            75
          ],
          [
            "modify_parameter_group",
            "conv1.weight",
            "(7, slice(None, None, None), slice(None, None, None), slice(None, None, None))",
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
        "candidate_sha256": "786ac1730a701c255d5ace270f60447b4cf458ee493e2b32821120539834930f",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter016",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter013.pt",
        "operations": [
          [
            "replace_parameter",
            "conv2.bias",
            "all",
            0
          ],
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            144
          ],
          [
            "replace_parameter",
            "conv3.weight",
            "all",
            0
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
          "conv2.weight": 144,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "0f1802d0b085f321b9b1c0cf8125f4695415afcdf9808ed602128b2c41a56663",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter017",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter013.pt",
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
        "candidate_sha256": "2ef45e57730bc2f922538f956f57deba84c22655ee848c94735396ebba071ab3",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter018",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter013.pt",
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
        "candidate_sha256": "8e71ea5d1e29dda699f11e00aafca1153869a0423ed1eebce65b5657f68a026e",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter019",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter013.pt",
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
          ],
          [
            "replace_parameter",
            "conv2.bias",
            "all",
            0
          ],
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            144
          ],
          [
            "replace_parameter",
            "conv3.weight",
            "all",
            0
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
          "conv2.weight": 144,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "7df38fbab3bb6cd4aa0e684087fdbcf60d00c24d34cdd85f2514db0a422d1dd5",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter020",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter013.pt",
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
        "candidate_sha256": "eff7031981bc3efa202b69f27777298401772e352b227209f05b8ca10762faba",
        "replay_matches": true,
        "all_changes_logged": true
      },
      {
        "id": "iter021",
        "decision": "rejected",
        "parent": "checkpoints/wsp-003/iter013.pt",
        "operations": [
          [
            "replace_parameter",
            "conv2.bias",
            "all",
            0
          ],
          [
            "replace_parameter",
            "conv2.weight",
            "all",
            344
          ],
          [
            "replace_parameter",
            "conv3.weight",
            "all",
            18
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
          "conv2.weight": 344,
          "conv3.weight": 18,
          "head.weight": 512,
          "head.bias": 1
        },
        "candidate_sha256": "caafba8e1a6141318ce9e8d898cead8c0412c82673cdaef3f68b7f736bdc2468",
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
      "test": 6000
    },
    "previous_experiment_artifacts_changed": [],
    "wsp001_artifacts_changed": []
  },
  "passed": true
}
```
