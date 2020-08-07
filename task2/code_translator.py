#!/usr/bin/env python
""" Translator Class and builder """
import argparse
import codecs
import os

import torch

from itertools import count
from onmt.utils.misc import tile
import numpy as np

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts
import onmt.decoders.ensemble
from onmt.translate.translator import Translator


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    if len(opt.models) > 1:
        # use ensemble decoding if more than one model is specified
        fields, model, model_opt = \
            onmt.decoders.ensemble.load_test_model(opt, dummy_opt.__dict__)
    else:
        fields, model, model_opt = \
            onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt)


    translator = CodeTranslator(model, fields, opt, model_opt,
                            global_scorer=scorer, out_file=out_file,
                            report_score=report_score, logger=logger)
    translator.src_path = opt.src
    return translator


class CodeTranslator(Translator):

    def translate(self,
                  src_path=None,
                  src_data_iter=None,
                  src_length=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  attn_debug=False,
                  ref_path=None):
        assert src_data_iter is not None or src_path is not None
        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters.build_dataset(self.fields,
                                       self.data_type,
                                       src_path=src_path,
                                       src_data_iter=src_data_iter,
                                       src_seq_length_trunc=src_length,
                                       tgt_path=tgt_path,
                                       tgt_data_iter=tgt_data_iter,
                                       src_dir=src_dir,
                                       sample_rate=self.sample_rate,
                                       window_size=self.window_size,
                                       window_stride=self.window_stride,
                                       window=self.window,
                                       use_filter_pred=self.use_filter_pred,
                                       ref_path=ref_path if self.refer else None,
                                       ref_seq_length_trunc=self.max_sent_length,
									   ignore_unk=False)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"
        # if self.refer:
        #     data.fields['ref'].vocab = data.fields['src'].vocab


        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        for batch in data_iter:
            batch_data = self.translate_batch(batch, data, fast=True, attn_debug=False)
            translations = builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()
        return all_scores, all_predictions







