## 增强数据集

先运行`并行提问-精简.py`以增强训练数据集，得到json和batch文件。

可以用`sort_and_merge.py`会来合并这些batch文件，它同时会报告重复的ID数目以及是哪些。

用`检查与更正.py`来检验，只取做对了的题目，

最后`修改answer和instruction.py`来修改答案和指令，得到增强后的训练集。

## 微调模型

`qwen_ft_with_CoT.py`

与base代码基本一致，只是增大了max_length到512。保险起见也可以到1024，但会增加耗时。

## 推理

`infer_CoT.py`

修改instruction和answer的格式；  
设置max_new_tokens为512，截断可能的超长输出或者无限循环。

## 结果

存储在`submit.csv`中，包含了id和答案两列。
为防止推理异常中止导致csv不完整或者为空，保存log文件以便恢复。
可以使用`merge_logs_to_csv.py`来提取log文件中的数据，合并成csv文件。