from __future__ import unicode_literals, print_function, division

import random
import matplotlib.pyplot as plt

# plt.switch_backend('agg')
import matplotlib.ticker as ticker
from torch import optim
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

from transformer.transformer import *
from utils.helper_funcs import *
from train import train
from constants import *

input_lang, output_lang, pairs = prepare_data('eng_train', 'my_train', True)
# input_lang, output_lang, pairs = prepare_data('en.train', 'my.train', True)
print(random.choice(pairs))


def tensors_from_pair(pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, target_tensor


# Then we call train many times and occasionally print the progress (% of examples, time so far, estimated time) and
# average loss.
def train_iters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)  # SGD is better for dense data
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# Evaluation
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# We can evaluate random sentences from the training set and print out the input, target, and output to make some
# subjective quality judgements:
def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# Training and Evaluating
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.15).to(device)

train_iters(encoder1, attn_decoder1, 100000, print_every=5000)
# train_iters(encoder1, attn_decoder1, 100000, print_every=5000)
# evaluate_randomly(encoder1, attn_decoder1)


_, _, test_pairs = prepare_data('eng_test', 'my_test', True)
# candidate_corpus = random.choices(pairs, k=50)
reference = []
candidate = []
for i in test_pairs:
    try:
        output_words, attentions = evaluate(encoder1, attn_decoder1, i[0])
        output_words = output_words[:-1]
        # output_sentence = ' '.join(output_words)
        candidate.append(output_words)
        reference.append(i[1].split(' '))
    except:
        pass

meteor_score_actual = 0.
for i in range(len(reference)):
    # score += sentence_bleu([reference[i]], candidate[i])
    if len(reference[i]) == 1:
        meteor_score_actual += meteor_score([reference[i]], candidate[i])
    elif len(reference[i]) == 2:
        meteor_score_actual += meteor_score([reference[i]], candidate[i])
    elif len(reference[i]) == 3:
        meteor_score_actual += meteor_score([reference[i]], candidate[i])
    elif len(reference[i]) == 4:
        meteor_score_actual += meteor_score([reference[i]], candidate[i])
    else:
        meteor_score_actual += meteor_score([reference[i]], candidate[i])

score = 0.
for i in range(len(reference)):
    # score += sentence_bleu([reference[i]], candidate[i])
    if len(reference[i]) == 1:
        score += sentence_bleu([reference[i]], candidate[i], weights=(1.0,))
    elif len(reference[i]) == 2:
        score += sentence_bleu([reference[i]], candidate[i], weights=(1.0, 0))
    elif len(reference[i]) == 3:
        score += sentence_bleu([reference[i]], candidate[i], weights=(1.0, 0, 0))
    elif len(reference[i]) == 4:
        score += sentence_bleu([reference[i]], candidate[i], weights=(1.0, 0, 0, 0))
    else:
        score += sentence_bleu([reference[i]], candidate[i])

score /= len(reference)
meteor_score_actual /= len(reference)
print("The METEOR score is: " + str(meteor_score_actual * 100))
print("The BLEU score is: " + str(score * 100))

# print(f'BLEU Score: {bleu_score(candidate, reference)}')
# _, _, test_pairs = prepare_data('eng_test', 'my_test', True)
#
# test_reference = []
# test_candidate = []
# for i in test_pairs:
#
#     output_words, attentions = evaluate(encoder1, attn_decoder1, i[0])
#     output_words = output_words[:-1]
#     output_sentence = ' '.join(output_words)
#     test_candidate.append([output_sentence])
#     test_reference.append([[i[1]]])
#
# print(f'BLEU Score: {bleu_score(test_candidate, test_reference)}')

# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import corpus_bleu
# import itertools
#
# if len(reference[i]) > 2:
#     scores += corpus_bleu(list(itertools.permutations(reference[i])), candidate[i])
# scores += corpus_bleu(list(itertools.permutations(reference[i])), candidate[i])
# # bleu_score(candidate, reference)
