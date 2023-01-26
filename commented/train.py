import random

from constants import *


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    # initialize encoder hidden state
    encoder_hidden = encoder.init_hidden()
    # zero the gradients of encoder and decoder optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # get the length of input and target tensors
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # initialize encoder outputs tensor
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    # loop through the input tensor and get encoder outputs
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # initialize decoder input and hidden state with SOS_token
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    # use teacher forcing with a probability of teacher_forcing_ratio
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    # backpropagation
    loss.backward()

    # update the parameters of encoder and decoder
    encoder_optimizer.step()
    decoder_optimizer.step()

    # return the average loss per target length
    return loss.item() / target_length
