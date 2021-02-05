import argparse

UNK_IDX = 0
UNK_WORD = "UUUNKKK"
BOS_IDX = 1
EOS_IDX = 2
BOC_IDX = 3
BOV_IDX = 4
BOQK_IDX = 5
BOQV_IDX = 6
SUBSEC_IDX = 7

EOC_IDX = 8
MASK_IDX = 9

MAX_VALUE_LEN = 20
MAX_GEN_LEN = 200

METEOR_JAR = 'evaluation/meteor-1.5.jar'
METEOR_DATA = 'evaluation/data/paraphrase-en.gz'
MULTI_BLEU_PERL = 'evaluation/multi-bleu.perl'
RESOURCE_LINK = 'https://drive.google.com/drive/folders/1FJjvMldeZrJnQd-iVXJ3KGFBLEvsndNY?usp=sharing'


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_base_parser():
    parser = argparse.ArgumentParser(
        description='WikiTableT using PyTorch')
    parser.register('type', 'bool', str2bool)

    basic_group = parser.add_argument_group('basics')
    # Basics
    basic_group.add_argument('--debug', type="bool", default=False,
                             help='activation of debug mode (default: False)')
    basic_group.add_argument('--auto_disconnect', type="bool", default=True,
                             help='for slurm (default: True)')
    basic_group.add_argument('--save_prefix', type=str, default="experiments",
                             help='saving path prefix')
    basic_group.add_argument('--gen_prefix', type=str, default="gen",
                             help='generation saving file prefix')
    basic_group.add_argument('--gen_dir', type=str, default="gen",
                             help='generation saving path directory')

    data_group = parser.add_argument_group('data')
    # Data file
    data_group.add_argument('--train_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--vocab_file', type=str, default=None,
                            help='vocabulary file')
    data_group.add_argument('--bpe_codes', type=str, default=None,
                            help='bpe code file')
    data_group.add_argument('--bpe_vocab', type=str, default=None,
                            help='bpe vocabulary file')
    data_group.add_argument('--dev_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--test_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--wikidata_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--infobox_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--max_num_value', type=int, default=None,
                            help='max number of values per cell')

    config_group = parser.add_argument_group('model_configs')
    config_group.add_argument('-lr', '--learning_rate',
                              dest='lr',
                              type=float,
                              default=1e-3,
                              help='learning rate')
    config_group.add_argument('-dp', '--dropout',
                              dest='dp',
                              type=float,
                              default=0.0,
                              help='dropout rate')
    config_group.add_argument('-lratio', '--logloss_ratio',
                              dest='lratio',
                              type=float,
                              default=1.0,
                              help='ratio of log loss')
    config_group.add_argument('--eps',
                              type=float,
                              default=1e-5,
                              help='safty for avoiding numerical issues')
    config_group.add_argument('-edim', '--embed_dim',
                              dest='edim',
                              type=int, default=300,
                              help='size of embedding')
    config_group.add_argument('-gclip', '--grad_clip',
                              dest='gclip',
                              type=float, default=1.0,
                              help='gradient clipping threshold')

    # recurrent neural network detail
    config_group.add_argument('-ensize', '--encoder_size',
                              dest='ensize',
                              type=int, default=512,
                              help='encoder hidden size')
    config_group.add_argument('-desize', '--decoder_size',
                              dest='desize',
                              type=int, default=512,
                              help='decoder hidden size')
    config_group.add_argument('-elayer', '--encoder_num_layer',
                              dest='elayer',
                              type=int, default=3,
                              help='number of encoder layer')
    config_group.add_argument('-dlayer', '--decoder_num_layer',
                              dest='dlayer',
                              type=int, default=3,
                              help='number of decoder layer')
    config_group.add_argument('-asize', '--attn_size',
                              dest='asize',
                              type=int, default=100,
                              help='size of attention')
    config_group.add_argument('-bwdelayer', '--bwd_encoder_num_layer',
                              dest='bwdelayer',
                              type=int, default=2,
                              help='number of encoder layer for backward models')
    config_group.add_argument('-bwdnhead', '--bwd_num_head',
                              dest='bwdnhead',
                              type=int, default=4,
                              help='number of attention heads for backward models')
    config_group.add_argument('-bwddlayer', '--bwd_decoder_num_layer',
                              dest='bwddlayer',
                              type=int, default=2,
                              help='number of decoder layer for backward models')

    # transformer
    config_group.add_argument('-act_fn', '--activation_function',
                              dest='act_fn',
                              type=str, default="gelu",
                              help='types of activation function used in transformer model')
    config_group.add_argument('-nhead', '--num_head',
                              dest='nhead',
                              type=int, default=4,
                              help='number of attention heads')

    # optimization
    config_group.add_argument('--l2', type=float, default=0.,
                              help='l2 regularization')
    config_group.add_argument('-wstep', '--warmup_steps',
                              dest='wstep', type=int, default=0,
                              help='learning rate warmup steps')
    config_group.add_argument('-lm', '--label_smoothing',
                              dest='lm', type=float, default=0.0,
                              help='label smoothing')
    config_group.add_argument('-bwd_lm', '--backward_label_smoothing',
                              dest='bwd_lm', type=float, default=0.0,
                              help='label smoothing')
    config_group.add_argument('-gcs', '--gradient_accumulation_steps',
                              dest='gcs', type=int, default=1,
                              help='gradient accumulation steps')
    config_group.add_argument('-tloss', '--true_cyclic_loss',
                              dest='tloss', type=float, default=1.0,
                              help='cyclic loss based on reference input')
    config_group.add_argument('-floss', '--fake_cyclic_loss',
                              dest='floss', type=float, default=1.0,
                              help='cyclic loss based on model input')

    setup_group = parser.add_argument_group('train_setup')
    # train detail
    setup_group.add_argument('--model_file', type=str, default=None,
                             help='model save path')
    setup_group.add_argument('--save_dir', type=str, default=None,
                             help='model save path')
    basic_group.add_argument('--encoder_type',
                             type=str, default="transformer",
                             help='types of encoder')
    basic_group.add_argument('--decoder_type',
                             type=str, default="ctransformer",
                             help='types of decoder')
    setup_group.add_argument('--n_epoch', type=int, default=5,
                             help='number of epochs')
    setup_group.add_argument('--max_gen_len', type=int, default=200,
                             help='maximum length for generation')
    setup_group.add_argument('--min_gen_len', type=int, default=0,
                             help='minimum length for generation')
    setup_group.add_argument('--max_encoder_len', type=int, default=512,
                             help='maximum input length for encoder')
    setup_group.add_argument('--max_decoder_len', type=int, default=512,
                             help='maximum input length for encoder')
    setup_group.add_argument('--max_train_txt_len', type=int, default=500,
                             help='maximum length for target text during training')
    setup_group.add_argument('--top_p', type=float, default=None,
                             help='generation sampling')
    setup_group.add_argument('--top_k', type=int, default=None,
                             help='generation sampling')
    setup_group.add_argument('--batch_size', type=int, default=20,
                             help='batch size')
    setup_group.add_argument('--eval_batch_size', type=int, default=50,
                             help='batch size')
    setup_group.add_argument('--opt', type=str, default='adam',
                             choices=['sadam', 'adam', 'sgd', 'rmsprop'],
                             help='types of optimizer: adam (default), \
                             sgd, rmsprop')
    setup_group.add_argument('--filter_ngram', type=int, default=0,
                             help='filter ngram during beam search')
    setup_group.add_argument('--trigram_blocking', type="bool", default=False,
                             help='filter ngram during beam search')
    setup_group.add_argument('--beam_size', type=int, default=10,
                             help='size for beam search')
    setup_group.add_argument('--return_wikidata', type="bool", default=False,
                             help='whether to mask predict wikidata')
    setup_group.add_argument('--return_hyperlink', type="bool", default=True,
                             help='whether to mask predict hyperlink')
    setup_group.add_argument('--return_titles', type="bool", default=True,
                             help='whether to mask predict titles')
    setup_group.add_argument('--input_wikidata', type="bool", default=True,
                             help='whether to input wikidata')
    setup_group.add_argument('--input_hyperlink', type="bool", default=True,
                             help='whether to input hyperlink')
    setup_group.add_argument('--use_copy', type="bool", default=False,
                             help='whether to use copy mechanism')
    setup_group.add_argument('--use_fyl', type="bool", default=False,
                             help='whether to use FY loss')
    setup_group.add_argument('-force_copy', '--force_copy',
                             dest='force_copy', type="bool", default=False,
                             help='whether to force copy from source in the copy mechanism')
    setup_group.add_argument('-share_decoder_embedding', '--share_decoder_embedding',
                             dest='share_decoder_embedding', type="bool", default=False,
                             help='whether to share embeddings in decoders')

    misc_group = parser.add_argument_group('misc')
    # misc
    misc_group.add_argument('--print_every', type=int, default=500,
                            help='print training details after \
                            this number of iterations')
    misc_group.add_argument('--eval_every', type=int, default=5000,
                            help='evaluate model after \
                            this number of iterations')
    misc_group.add_argument('--save_every', type=int, default=2000,
                            help='save model after \
                            this number of iterations')
    return parser
