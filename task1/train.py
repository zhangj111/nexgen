import sys
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model import *
from utils import *
from dataset import EHDataset
from sklearn import metrics

# Data parameters
data_folder = './output'
word_map = torch.load(os.path.join(data_folder, 'vocab.pt')).stoi

# Model parameters
n_classes = 1
emb_size = 128
word_rnn_size = 128  # word RNN size
sentence_rnn_size = 128  # character RNN size
word_rnn_layers = 1  # number of layers in character RNN
sentence_rnn_layers = 1  # number of layers in word RNN
word_att_size = 128  # size of the word-level attention layer (also the size of the word context vector)
sentence_att_size = 128  # size of the sentence-level attention layer (also the size of the sentence context vector)
dropout = 0.  # dropout
use_crf = False

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 64  # batch size
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
workers = 4  # number of workers for loading data in the DataLoader
epochs = 20  # number of epochs to run
grad_clip = None  # clip gradients at this value
print_freq = 200  # print training or validation status every __ batches
checkpoint = None  # path to model checkpoint, None if none

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


def main():
    """
    Training and validation.
    """
    global checkpoint, start_epoch, word_map

    # Initialize model or load checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        word_map = checkpoint['word_map']
        start_epoch = checkpoint['epoch'] + 1
        print(
            '\nLoaded checkpoint from epoch %d.\n' % (start_epoch - 1))
    else:
        if use_crf:
            model = HACRF(n_classes=n_classes,
                          vocab_size=len(word_map),
                          emb_size=emb_size,
                          word_rnn_size=word_rnn_size,
                          sentence_rnn_size=sentence_rnn_size,
                          word_rnn_layers=word_rnn_layers,
                          sentence_rnn_layers=sentence_rnn_layers,
                          word_att_size=word_att_size,
                          dropout=dropout,
                          device=device)
        else:
            model = HierarchicalAttentionNetwork(n_classes=n_classes,
                                                 vocab_size=len(word_map),
                                                 emb_size=emb_size,
                                                 word_rnn_size=word_rnn_size,
                                                 sentence_rnn_size=sentence_rnn_size,
                                                 word_rnn_layers=word_rnn_layers,
                                                 sentence_rnn_layers=sentence_rnn_layers,
                                                 word_att_size=word_att_size,
                                                 sentence_att_size=sentence_att_size,
                                                 dropout=dropout)

        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Loss functions
    criterion = nn.BCELoss() # CrossEntropyLoss()

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(EHDataset(data_folder, 'train'), batch_size=batch_size, shuffle=True,
                                               num_workers=workers)

    test_loader = torch.utils.data.DataLoader(EHDataset(data_folder, 'test'), batch_size=batch_size, shuffle=False,
                                              num_workers=workers)

    print("Start training...")
    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, word_map)
        if not use_crf:
            evaluate(model=model, test_loader=test_loader, criterion=criterion)
        else:
            evaluate_with_crf(model=model, test_loader=test_loader)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    # Track metrics
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    model.train()  # training mode enables dropout

    # Batches
    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(train_loader):

        documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.permute(0, 2, 1).squeeze(-1).to(device)
        if not use_crf:
            packed_labels = pack_padded_sequence(labels, sentences_per_document.tolist(),
                                                 batch_first=True, enforce_sorted=False)

            # Forward prop.
            scores = model(documents, sentences_per_document, words_per_sentence)  # (n_documents, n_classes)
            scores = scores.squeeze(-1)
            # Loss
            packed_scores = pack_padded_sequence(scores, sentences_per_document.tolist(),
                                                 batch_first=True, enforce_sorted=False)
            loss = criterion(packed_scores.data, packed_labels.data)  # scalar
        else:
            loss = model(documents, sentences_per_document, words_per_sentence, labels)
        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Update
        optimizer.step()

        # Find accuracy
        if not use_crf:
            predictions = scores.gt(0.5).float()
            res = []
            for j, length in enumerate(sentences_per_document.tolist()):
                truth = labels[j][:length]
                prediction = predictions[j][:length]
                # res.extend((prediction == truth).float().cpu())
                res.append(1) if prediction.equal(truth) else res.append(0)
            accuracy = sum(res) / len(res)
            # Keep track of metrics
            losses.update(loss.item(), len(res))
            accs.update(accuracy, len(res))
            # Print training status
            # Print training status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader), loss=losses, acc=accs))
        else:
            losses.update(loss.item(), len(labels))
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses))


def evaluate(model, test_loader, criterion):
    # Track metrics
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    model.eval()
    # Evaluate in batches
    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(
            tqdm(test_loader, desc='Evaluating')):
        documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.permute(0, 2, 1).squeeze(-1).to(device)
        packed_labels = pack_padded_sequence(labels, sentences_per_document.tolist(),
                                             batch_first=True, enforce_sorted=False)

        # Forward prop.
        scores = model(documents, sentences_per_document, words_per_sentence)  # (n_documents, n_classes)
        scores = scores.squeeze(-1)

        # Loss
        packed_scores = pack_padded_sequence(scores, sentences_per_document.tolist(),
                                             batch_first=True, enforce_sorted=False)
        loss = criterion(packed_scores.data, packed_labels.data)  # scalar

        # Find accuracy
        predictions = scores.gt(0.5).float()
        res = []
        for j, length in enumerate(sentences_per_document.tolist()):
            truth = labels[j][:length]
            prediction = predictions[j][:length]
            # res.extend((prediction == truth).float().cpu())
            res.append(1) if prediction.equal(truth) else res.append(0)
        accuracy = sum(res) / len(res)
        # Keep track of metrics
        losses.update(loss.item(), len(res))
        accs.update(accuracy, len(res))

    # Print final result
    print('\n *Evaluation Loss: %.4f\t Accuracy: %.3f\n' % (losses.avg, accs.avg))


def evaluate_with_crf(model, test_loader):
    # Track metrics
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    model.eval()
    # Evaluate in batches
    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(
            tqdm(test_loader, desc='Evaluating')):
        documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.permute(0, 2, 1).squeeze(-1).to(device)

        # Loss
        loss = model(documents, sentences_per_document, words_per_sentence, labels)  # scalar

        # Forward prop.
        batch_max_scores, batch_max_ids = model.decode(documents, sentences_per_document, words_per_sentence)  # (n_documents, n_classes)

        # Find accuracy

        res = [1 if p[:s].equal(t[:s]) else 0 for p, t, s in zip(batch_max_ids, labels, sentences_per_document)]

        accuracy = sum(res) / len(res)
        # Keep track of metrics
        losses.update(loss.item(), len(res))
        accs.update(accuracy, len(res))

    # Print final result
    print('\n *Evaluation Loss: %.4f\t Accuracy: %.3f\n' % (losses.avg, accs.avg))


def predict(epoch):
    test_loader = torch.utils.data.DataLoader(EHDataset(data_folder, 'test'), batch_size=batch_size, shuffle=False,
                                              num_workers=workers)
    model = torch.load('checkpoints/checkpoint_%d.pth.tar' % epoch)['model']
    model.eval()
    y_t, y_p = [], []
    res = []
    # Evaluate in batches
    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(
            tqdm(test_loader, desc='Evaluating')):
        documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.permute(0, 2, 1).squeeze(-1).to(device)

        # Forward prop.
        scores = model(documents, sentences_per_document, words_per_sentence)  # (n_documents, n_classes)
        scores = scores.squeeze(-1)

        # Find accuracy
        predictions = scores.gt(0.5).float()

        for j, (length, nums) in enumerate(zip(sentences_per_document.tolist(), words_per_sentence.tolist())):
            truth = labels[j][:length]
            prediction = predictions[j][:length]
            # res.extend((prediction == truth).float().cpu())
            res.append(1) if prediction.equal(truth) else res.append(0)

            for n, t, p in zip(nums, truth.tolist(), prediction.tolist()):
                y_t.extend([t])
                y_p.extend([p])

    acc = sum(res)/len(res)
    print('Accuracy:', acc)
    print('Precision:', metrics.precision_score(y_t, y_p))
    print('Recall:', metrics.recall_score(y_t, y_p))
    print('F1:', metrics.f1_score(y_t, y_p))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'train':
        main()
    elif mode == 'test':
        predict(eval(sys.argv[2]))
