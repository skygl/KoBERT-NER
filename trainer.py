import os
import shutil
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import compute_metrics, get_labels, get_test_texts, show_report, MODEL_CLASSES, get_labels_from_path
from finetune_model import BertNER

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, unsup_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.unsup_dataset = unsup_dataset

        self.label_lst = get_labels(args)
        self.num_labels = len(self.label_lst)

        if args.task == 'naver-ner' or args.task == 'few-shot':
            # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
            self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

            self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

            self.config = self.config_class.from_pretrained(args.model_name_or_path,
                                                            num_labels=self.num_labels,
                                                            finetuning_task=args.task,
                                                            id2label={str(i): label for i, label in enumerate(self.label_lst)},
                                                            label2id={label: i for i, label in enumerate(self.label_lst)})
            if args.task == 'naver-ner':
                self.model = self.model_class.from_pretrained(args.model_name_or_path, config=self.config)
                if args.load_pretrained:
                    self.load_pretrained_model()
            else:
                self.model_class = BertNER
                self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                              dataset_label_nums=[len(self.label_lst)],
                                                              output_attentions=False,
                                                              output_hidden_states=False)
            # GPU or CPU
            self.device = f"cuda:{args.cuda_number}" if torch.cuda.is_available() and not args.no_cuda else "cpu"
            self.model.to(self.device)

            self.test_texts = None
            if args.write_pred:
                self.test_texts = get_test_texts(args)
                # Empty the original prediction files
                if os.path.exists(args.pred_dir):
                    shutil.rmtree(args.pred_dir)

    def load_pretrained_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.pretrained_model_dir):
            raise Exception("Model doesn't exists! Train first!")
        try:
            pretrained = self.model_class.from_pretrained(self.args.pretrained_model_dir)
            pretrained_dict = pretrained.state_dict()
            current_model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in current_model_dict and 'classifier' not in k}
            current_model_dict.update(pretrained_dict)
            self.model.load_state_dict(current_model_dict)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        unsup_batch_per_train = 0
        if self.args.do_self_train:
            unsup_sampler = RandomSampler(self.unsup_dataset)
            unsup_dataloader = DataLoader(self.unsup_dataset, sampler=unsup_sampler,
                                          batch_size=self.args.train_batch_size)
            unsup_batch_per_train = max(1, len(unsup_dataloader) // len(train_dataloader))
            unsup_iter = iter(unsup_dataloader)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()

                if self.args.do_self_train:
                    unsup_iter_loss = 0.0
                    for _ in range(unsup_batch_per_train):
                        self.model.train()
                        unsup_batch = next(unsup_iter)
                        unsup_batch = tuple(t.to(self.device) for t in unsup_batch)  # GPU or CPU
                        unsup_inputs = {'input_ids': unsup_batch[0],
                                        'attention_mask': unsup_batch[1],
                                        'labels': unsup_batch[3]}
                        if self.args.model_type != 'distilkobert':
                            unsup_inputs['token_type_ids'] = unsup_batch[2]
                        unsup_outputs = self.model(**unsup_inputs)
                        unsup_loss = unsup_outputs[0]
                        if self.args.gradient_accumulation_steps > 1:
                            unsup_loss = unsup_loss / self.args.gradient_accumulation_steps
                        unsup_loss.backward()

                        tr_loss += unsup_iter_loss

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("test", global_step)

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode, step):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.args.model_type != 'distilkobert' and self.args.task != 'few-shot':
                    inputs['token_type_ids'] = batch[2]
                if self.args.task == 'few-shot':
                    inputs['output_logits'] = True
                outputs = self.model(**inputs)
                if self.args.task == 'naver-ner':
                    tmp_eval_loss, logits = outputs[:2]
                else:
                    tmp_eval_loss, o, logits = outputs
                    if nb_eval_steps % 10 == 0:
                        p_tmp = logits.detach().cpu().numpy()
                        o_hat = np.argmax(p_tmp, axis=2)
                        print("predicted")
                        print(o_hat[:5])
                        print("inside predicted")
                        print(o[:5])
                        print("label")
                        print(inputs['labels'][:5])

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        preds = np.argmax(preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])

        if self.args.write_pred:
            if not os.path.exists(self.args.pred_dir):
                os.mkdir(self.args.pred_dir)

            with open(os.path.join(self.args.pred_dir, "pred_{}.txt".format(step)), "w", encoding="utf-8") as f:
                for text, true_label, pred_label in zip(self.test_texts, out_label_list, preds_list):
                    for t, tl, pl in zip(text, true_label, pred_label):
                        f.write("{} {} {}\n".format(t, tl, pl))
                    f.write("\n")

        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + show_report(out_label_list, preds_list))  # Get the report for each tag result

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            if self.args.task == 'naver-ner':
                self.model = self.model_class.from_pretrained(self.args.model_dir)
            else:
                pre_model = self.model
                self.model = torch.load(self.args.model_dir)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
