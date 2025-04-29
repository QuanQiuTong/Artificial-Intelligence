# %%
import torch as t
from 机器学习 import DigitsMobilenet, DigitsResnet18, DigitsDataset, parse2class, write2csv, config
from torch.utils.data import DataLoader

def main():
    # 选 mobilenet
    model = DigitsMobilenet(config.class_num).cuda()
    ckpt = t.load('./checkpoints/epoch-34-acc-67.67.pth')
    model.load_state_dict(ckpt['model'])
    model.eval()

    test_loader = DataLoader(DigitsDataset(mode='test', aug=False),
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers)
    results = []
    with t.no_grad():
        for img, names in test_loader:
            img = img.cuda()
            preds = model(img)
            codes = parse2class(preds)
            results += [[n, c] for n, c in zip(names, codes)]

    write2csv(results, 'results.csv')

if __name__ == '__main__':
    main()
