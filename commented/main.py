from __future__ import unicode_literals, print_function, division

import random
import matplotlib.pyplot as plt

# plt.switch_backend('agg')
import matplotlib.ticker as ticker
from torch import optim
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import sentence_bleu

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
    start = time.time()  # record the start time of training process
    plot_losses = []  # list to store the average loss for each plot_every iterations
    print_loss_total = 0  # variable to keep track of the total loss for each print_every iterations
    plot_loss_total = 0  # variable to keep track of the total loss for each plot_every iterations

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)  # SGD is better for dense data
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(random.choice(pairs)) for i in range(n_iters)]  # create training pairs
    criterion = nn.NLLLoss()  # use negative log likelihood loss as criterion

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]  # select a training example
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)  # update the model's parameters
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every  # calculate the average loss for the past print_every iterations
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))  # print the progress of the training

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every  # calculate the average loss for the past plot_
            plot_losses.append(plot_loss_avg)  # append the average loss to the plot_losses list
            plot_loss_total = 0  # reset the plot_loss_total variable

    show_plot(plot_losses)  # display a plot of the training loss over time


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
        input_tensor = tensor_from_sentence(input_lang, sentence)  # convert input sentence to tensor
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()  # initialize the hidden state of the encoder

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)  # create a tensor to store encoder outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)  # pass input and hidden state to encoder and get output
            encoder_outputs[ei] += encoder_output[0, 0]  # store encoder outputs

        decoder_input = torch.tensor([[SOS_token]], device=device)  # initialize decoder input with SOS_token
        decoder_hidden = encoder_hidden  # set decoder hidden state as encoder hidden state

        decoded_words = []  # list to store decoded words
        decoder_attentions = torch.zeros(max_length, max_length)  # create a tensor to store attention weights

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)  # pass input, hidden state and encoder outputs to decoder
            decoder_attentions[di] = decoder_attention.data  # store attention weights
            topv, topi = decoder_output.data.topk(1)  # get the top word from decoder's output
            if topi.item() == EOS_token:  # check if top word is EOS_token
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])  # append the top word to
            decoder_input = topi.squeeze().detach() # set the top word as the next input to the decoder

        # return the decoded words and attention weights
        return decoded_words, decoder_attentions[:di + 1]


# We can evaluate random sentences from the training set and print out the input, target, and output to make some
# subjective quality judgements:
def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0]) # print the input sentence
        print('=', pair[1]) # print the target sentence
        output_words, attentions = evaluate(encoder, decoder, pair[0]) # get the decoded words and attention weights
        output_sentence = ' '.join(output_words) # join the decoded words to form a sentence
        print('<', output_sentence) # print the decoded sentence
        print('')

# Training and Evaluating
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device) # initialize the encoder
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.15).to(device) # initialize the decoder

train_iters(encoder1, attn_decoder1, 100000, print_every=5000)
# evaluate_randomly(encoder1, attn_decoder1)


_, _, test_pairs = prepare_data('eng_test', 'my_test', True)
# Load the test dataset

reference = [] # list to hold the reference sentences
candidate = [] # list to hold the generated sentences

# Iterate through the test pairs
for i in test_pairs:
    try:
        # Get the generated sentence and attention scores
        output_words, attentions = evaluate(encoder1, attn_decoder1, i[0])

        # Remove the end of sentence token from the generated sentence
        output_words = output_words[:-1]

        # Add the generated sentence to the candidate list
        candidate.append(output_words)

        # Add the reference sentence to the reference list
        reference.append(i[1].split(' '))
    except NameError:
        pass

score = 0. # Initialize the BLEU score

for i in range(len(reference)):
    # score += sentence_bleu([reference[i]], candidate[i])
    # The above line is commented out, it would add the BLEU score for each sentence to the 'score' variable.

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
    # The above block of code checks the number of elements in the current sentence of the 'reference' list, and based on that, calls the 'sentence_bleu' function with different weights.
    # The 'weights' argument is used to weigh the importance of different n-grams (1-gram, 2-gram, 3-gram, etc.) in the BLEU score calculation.
    # In this case, if the length of the sentence is 1, it will only consider 1-grams and give them a weight of 1.0.
    # If the length is 2, it will consider 1-grams and 2-grams and give 1-grams a weight of 1.0 and 2-grams a weight of 0.
    # Similarly, for 3 and 4 length sentences, the weights for 3 and 4 grams are set to 0.
    # For all other cases, the function will be called without the 'weights' argument, which means that all n-grams will be considered, and their weights will be based on their relative frequency of occurence.

score /= len(reference)
# This line divides the total score by the number of sentences in the 'reference' list to get an average BLEU score.

print("The bleu score is: " + str(score * 100))
# This line prints the final BLEU score, multiplied by 100 to get a percentage value.

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
