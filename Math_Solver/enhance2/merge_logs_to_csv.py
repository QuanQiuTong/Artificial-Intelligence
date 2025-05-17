import re
import csv
import argparse

def parse_log(file_path):
    """解析单个 log 文件，返回 {id: answer}"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 按 “=====” 分割出每个样本块
    blocks = re.split(r'^=====$', content, flags=re.MULTILINE)
    result = {}
    for block in blocks:
        # 清洗空行
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        # 第一行示例：样本ID 123 生成长度: 45
        m = re.match(r'^样本ID\s+(\d+)\s+生成长度:', lines[0])
        if not m:
            continue
        sample_id = m.group(1)
        # 最后一行即为提取的答案
        final_answer = lines[-1]
        result[sample_id] = final_answer
    return result

def main():
    parser = argparse.ArgumentParser(description="合并多份 log，提取样本 ID 和答案到 CSV")
    parser.add_argument('--logs', nargs='+', required=True, help='输入的 log 文件列表')
    parser.add_argument('--output', default='submit.csv', help='输出的 CSV 文件名')
    args = parser.parse_args()

    merged = {}
    # 逐个解析、合并
    for log_path in args.logs:
        parsed = parse_log(log_path)
        merged.update(parsed)

    # 按 ID 排序写入 CSV
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'answer'])
        for sid in sorted(merged.keys(), key=lambda x: int(x)):
            writer.writerow([sid, merged[sid]])

if __name__ == '__main__':
    main()