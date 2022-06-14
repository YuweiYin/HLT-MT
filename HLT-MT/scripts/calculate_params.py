def calculate_standard_transformer_params(encoder_layers, decoder_layers, vocab_size, embed_dim, model_name="MNMT"):
    embed_params = embed_dim * vocab_size
    encoder_params = (embed_dim * embed_dim * 4 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2) * encoder_layers
    decoder_params = (embed_dim * embed_dim * 4 * 2 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2) * decoder_layers
    params = embed_params + encoder_params + decoder_params
    print("{}: {}".format(model_name, params))


def calculate_interleaved_transformer_params(encoder_layers, decoder_layers, vocab_size, embed_dim, model_name="MNMT"):
    embed_params = embed_dim * vocab_size
    encoder_params = (embed_dim * embed_dim * 4 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2) * encoder_layers
    decoder_params = (embed_dim * embed_dim * 4 * 2 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2 * 2) * decoder_layers
    params = embed_params + encoder_params + decoder_params
    print("{}: {}".format(model_name, params))


def calculate_our_params(encoder_layers, decoder_layers, vocab_size, embed_dim, ls_number=4, model_name="MNMT"):
    embed_params = embed_dim * vocab_size
    encoder_params = (embed_dim * embed_dim * 4 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2) * encoder_layers
    decoder_params = (embed_dim * embed_dim * 4 * 2 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2 * 2) * decoder_layers
    ls_params = (embed_dim + embed_dim * 4 * embed_dim * 2) * ls_number
    params = embed_params + encoder_params + decoder_params + ls_params
    print("{}: {}".format(model_name, params))


def calculate_monolingual_adapter_params(encoder_layers, decoder_layers, vocab_size, embed_dim, ls_number=11,
                                         ls_dim=256, model_name="monolingual_adapter"):
    embed_params = embed_dim * vocab_size
    encoder_params = (embed_dim * embed_dim * 4 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2) * encoder_layers
    decoder_params = (embed_dim * embed_dim * 4 * 2 + embed_dim * 2 + embed_dim * 4 * embed_dim * 2 * 2) * decoder_layers
    ls_params = (embed_dim + embed_dim * ls_dim * 2) * ls_number * 2
    params = embed_params + encoder_params + decoder_params + ls_params
    print("{}: {}".format(model_name, params))


calculate_standard_transformer_params(encoder_layers=6, decoder_layers=6, vocab_size=63997, embed_dim=1024,
                                      model_name="wmt10_MNMT")
calculate_interleaved_transformer_params(encoder_layers=12, decoder_layers=6, vocab_size=250000, embed_dim=768,
                                         model_name="wmt10_XLMT")
calculate_interleaved_transformer_params(encoder_layers=36, decoder_layers=6, vocab_size=250000, embed_dim=768,
                                         model_name="wmt10_MESED")
calculate_our_params(encoder_layers=12, decoder_layers=6, vocab_size=250000, embed_dim=768, ls_number=10,
                     model_name="wmt10_LSMNMT")
calculate_monolingual_adapter_params(encoder_layers=12, decoder_layers=6, vocab_size=250000, embed_dim=768,
                                     ls_number=11, model_name="wmt10_monolingual_adapter")
calculate_monolingual_adapter_params(encoder_layers=12, decoder_layers=6, vocab_size=250000, embed_dim=768,
                                     ls_number=95, model_name="opus_monolingual_adapter")
calculate_our_params(encoder_layers=12, decoder_layers=6, vocab_size=250000, embed_dim=768, ls_number=20,
                     model_name="LSMNMT")
