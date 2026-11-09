# Weight-update audit trail

Every weight change goes through `WeightProgrammer`; each iteration was replayed from its parent checkpoint.

| id | decision | operations | changed per tensor | replay hash | all changes logged |
|---|---|---|---|---|---|
| iter001 | rejected | replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter002 | rejected | modify_parameter_group(conv2.bias, 32 changed); replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'conv2.bias': 32, 'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter003 | accepted | modify_parameter_group(conv2.weight, 9 changed); modify_parameter_group(conv2.weight, 9 changed); modify_parameter_group(conv2.weight, 9 changed); modify_parameter_group(conv2.weight, 9 changed); modify_parameter_group(conv2.weight, 9 changed); modify_parameter_group(conv2.weight, 9 changed); modify_parameter_group(conv2.weight, 9 changed); modify_parameter_group(conv2.weight, 9 changed); replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'conv2.weight': 72, 'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter004 | rejected | replace_parameter(conv3.weight, 1296 changed); replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'conv3.weight': 1296, 'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter005 | rejected | replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter006 | rejected | replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter007 | rejected | replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter008 | rejected | modify_parameter_group(conv3.bias, 32 changed); replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'conv3.bias': 32, 'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter009 | rejected | replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter010 | rejected | replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter011 | rejected | modify_parameter_group(conv1.weight, 275 changed); replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'conv1.weight': 250, 'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter012 | rejected | modify_parameter_group(conv1.weight, 300 changed); replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'conv1.weight': 300, 'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter013 | rejected | modify_parameter_group(conv1.weight, 75 changed); modify_parameter_group(conv1.weight, 75 changed); modify_parameter_group(conv1.weight, 75 changed); modify_parameter_group(conv1.weight, 75 changed); replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'conv1.weight': 300, 'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter014 | rejected | modify_parameter_group(conv2.weight, 24 changed); replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'conv2.weight': 24, 'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter015 | rejected | modify_parameter_group(conv3.weight, 576 changed); replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'conv3.weight': 576, 'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter016 | rejected | replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter017 | rejected | replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter018 | rejected | modify_parameter_group(conv3.weight, 72 changed); modify_parameter_group(conv3.weight, 72 changed); modify_parameter_group(conv3.weight, 72 changed); modify_parameter_group(conv3.weight, 72 changed); modify_parameter_group(conv3.weight, 72 changed); modify_parameter_group(conv3.weight, 72 changed); modify_parameter_group(conv3.weight, 72 changed); modify_parameter_group(conv3.weight, 72 changed); replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'conv3.weight': 576, 'head.weight': 512, 'head.bias': 1} | MATCH | yes |
| iter019 | rejected | replace_parameter(head.weight, 512 changed); replace_parameter(head.bias, 1 changed) | {'head.weight': 512, 'head.bias': 1} | MATCH | yes |
