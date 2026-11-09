# Weight-update audit trail

Every weight change goes through `WeightProgrammer`; each iteration was replayed from its parent checkpoint.

| id | decision | operations | changed per tensor | replay hash | all changes logged |
|---|---|---|---|---|---|
| iter001 | rejected | modify_parameter_group(conv2.weight, 20736 changed); replace_parameter(head.weight, 1536 changed); replace_parameter(head.bias, 1 changed) | {'conv2.weight': 20736, 'head.weight': 1536, 'head.bias': 1} | MATCH | yes |
| iter002 | rejected | modify_parameter_group(conv2.weight, 520 changed); modify_parameter_group(conv2.bias, 48 changed); replace_parameter(head.weight, 1536 changed); replace_parameter(head.bias, 1 changed) | {'conv2.weight': 520, 'conv2.bias': 48, 'head.weight': 1536, 'head.bias': 1} | MATCH | yes |
| iter003 | rejected | modify_parameter_group(conv1.weight, 75 changed); modify_parameter_group(conv1.weight, 75 changed); modify_parameter_group(conv1.weight, 75 changed); modify_parameter_group(conv1.weight, 75 changed); replace_parameter(head.weight, 1536 changed); replace_parameter(head.bias, 1 changed) | {'conv1.weight': 300, 'head.weight': 1536, 'head.bias': 1} | MATCH | yes |
