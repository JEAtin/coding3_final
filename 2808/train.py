from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.metrics import classification_report

from utils import MyDataset, set_seed
import args


def main():
    def collate_fn(data):
        texts = []
        labels = []
        for t, lab in data:
            texts.append(t)
            labels.append(lab)
        inputs = tokenizer.batch_encode_plus(texts, max_length=args.max_length, padding=True, return_tensors='pt', truncation=True)
        labels = torch.LongTensor(labels)
        return inputs, labels

    set_seed(args.seed)
    train_set = MyDataset(args.data_path)
    test_set = MyDataset(args.data_path, is_train=False)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_sz, collate_fn=collate_fn, shuffle=True)
    test_loader =  DataLoader(test_set,  batch_size=args.test_batch_sz, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    config = BertConfig.from_pretrained(args.pretrained_model_path)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model_path, config=config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # trian
    best_score = -1.0
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc='Train', ncols=100, nrows=20)
        for data in pbar:
            inputs, labels = data
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            loss = model(**inputs, labels=labels)[0]
            pbar.set_postfix({'Epoch': epoch + 1,
                              'Loss': '{:.4f}'.format(loss.item())})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad()

        # evaluate
        y_pred = []
        y_true = []
        with torch.no_grad():
            model.eval()
            pbar = tqdm(test_loader, desc='Eval ', ncols=100, nrows=20)
            for data in pbar:
                inputs, labels = data
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                output = model(**inputs, labels=labels)
                loss_eval = output.loss
                logits = output.logits
                pbar.set_postfix({'Epoch': epoch + 1,
                                  'Loss': '{:.4f}'.format(loss_eval.item())})
                y_pred.extend(torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist())
                y_true.extend(labels.detach().cpu().numpy().tolist())
        report = classification_report(y_pred, y_true, output_dict=True, labels=[0, 1], target_names=['neg', 'pos'])
        score = report['macro avg']
        print(score)

        # save model
        f1_score = score['f1-score']
        flag = f1_score > best_score
        if flag:
            best_score = f1_score
            model.save_pretrained(args.output_path + 'epoch {}/'.format(epoch + 1))
            tokenizer.save_pretrained(args.output_path + 'epoch {}/'.format(epoch + 1))


if __name__ == '__main__':
    main()

