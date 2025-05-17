先运行`enhance_train_with_cot.py`，或者换用`并行提问.py`以加速，
然后`sort_and_merge.py`会报告重复的ID数目以及是哪些，
用`check_if_gpt_is_correct.py`来检验，只取做对了的题目，
最后`修改answer和instruction.py`来修改答案和指令，得到增强后的训练集。