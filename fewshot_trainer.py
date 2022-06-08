import os

import torch.cuda

from trainer import Trainer
from word_encoder import BERTWordEncoder
from data_loader import get_loader
from utils import load_tokenizer
from FewShotNERModel import Proto, FewShotNERFramework


class FewShotTrainer(Trainer):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        super(FewShotTrainer, self).__init__(args, train_dataset, dev_dataset, test_dataset)

        self.word_encoder = BERTWordEncoder(args.model_name_or_path)

    def train(self):
        trainN = self.args.trainN
        N = self.args.N
        K = self.args.K
        Q = self.args.Q
        batch_size = self.args.train_batch_size
        max_length = self.args.max_seq_len

        if torch.cuda.is_available() and not self.args.no_cuda:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.cuda_number)

        print("{}-way-{}-shot Few-Shot NER".format(N, K))
        print("max_length: {}".format(max_length))

        tokenizer = load_tokenizer(self.args)

        train_path = os.path.join(self.args.data_dir, 'train.txt')
        train_dataloader = get_loader(train_path, tokenizer, trainN, K, Q, batch_size, max_length, use_sampled_data=False)
        valid_path = os.path.join(self.args.data_dir, 'valid.txt')
        valid_dataloader = get_loader(valid_path, tokenizer, N, K, Q, batch_size, max_length, use_sampled_data=False)

        prefix = '-'.join([str(N), str(K), 'seed'+str(self.args.seed)])

        model = Proto(self.word_encoder)

        framework = FewShotNERFramework(train_dataloader, valid_dataloader, None, use_sampled_data=False)

        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')

        ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
        print('model-save-path: ', ckpt)

        if torch.cuda.is_available() and not self.args.no_cuda:
            model.cuda()

        framework.train(model, prefix, save_ckpt=ckpt, train_iter=self.args.train_iter,
                        warmup_step=int(self.args.train_iter * 0.1), val_iter=self.args.valid_iter,
                        learning_rate=self.args.learning_rate)

    def load_model(self):
        pass

    def evaluate(self, mode, step):
        N = self.args.N
        K = self.args.K
        Q = self.args.Q
        batch_size = self.args.train_batch_size
        max_length = self.args.max_seq_len

        if torch.cuda.is_available() and not self.args.no_cuda:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.cuda_number)

        prefix = '-'.join([str(N), str(K), 'seed'+str(self.args.seed)])
        ckpt = 'checkpoint/{}.pth.tar'.format(prefix)

        tokenizer = load_tokenizer(self.args)

        test_path = os.path.join(self.args.data_dir, 'test.txt')
        test_dataloader = get_loader(test_path, tokenizer, N, K, Q, batch_size, max_length, use_sampled_data=False)

        model = Proto(self.word_encoder)
        framework = FewShotNERFramework(None, None, test_dataloader, use_sampled_data=False)

        if torch.cuda.is_available() and not self.args.no_cuda:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.cuda_number)
            model.cuda()

        precision, recall, f1, fp, fn, within, outer = framework.eval(model, self.args.test_iter, ckpt=ckpt)
        print("RESULT: precision: %.4f, recall: %.4f, f1:%.4f" % (precision, recall, f1))
        print('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within:%.4f, outer: %.4f'%(fp, fn, within, outer))
