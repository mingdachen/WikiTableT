# WikiTableT

Code, data, and pretrained models for the paper "[Generating Wikipedia Article Sections from Diverse Data Sources](https://arxiv.org/abs/2012.14919)"

**Note: we refer to the section data as hyperlink data in both the processed json files and the codebase.**

## Resources

- [WikiTableT dataset](https://drive.google.com/file/d/1HRpnKLI6vZusB8NoR0cgUYeoD8aX5q2r/view?usp=sharing)
- [multi-bleu and METEOR score](https://drive.google.com/drive/folders/1FJjvMldeZrJnQd-iVXJ3KGFBLEvsndNY?usp=sharing)
- [Trained models (base+copy+cyc (trained on 500k instances) and large+copy+cyc (trained on the full dataset))](https://drive.google.com/drive/folders/1L8kzbWVwufnJXtMAmoB1slPez7mqsVzE?usp=sharing)
- [BPE code and vocab](https://drive.google.com/file/d/1PN_0lHLBCbBDHnJC3CTsdkC1poVBSG7M/view?usp=sharing) (We used https://github.com/rsennrich/subword-nmt)
- [Data for computing the PARENT scores](https://drive.google.com/file/d/1VjyqChwuzAhUcP1Me8Ay_UemZwL8qNNS/view?usp=sharing)

## Dependencies

- Python 3.7
- PyTorch 1.5.1
- NLTK
- [py-rouge](https://github.com/Diego999/py-rouge)
- [entmax](https://github.com/deep-spin/entmax)

## Usage

Tp train a new model, you may use a command similar to ``scripts/train_large_copy_cyc.sh``.

To perform beam search generation using a trained model, you may use a command similar to ``scripts/generate_beam_search.sh``. The process should generate 4 files including references. 2 of them are tokenized using NLTK for the convenience of latter evaluation steps.

If you want to generate your own version of reference data when computing the PARENT scores, use a command similar to ``scripts/convert2parent_dev.sh``.

Once you have the generated file, you may evaluate it against the reference using the command ``scripts/eval_dev.sh REF_FILE_PATH GEN_FILE_PATH``. Please make sure that you are using the tokenized files.

## Acknowledgement

Part of the code in this repository is adapted from the following repositories:

- https://github.com/huggingface/transformers
- https://github.com/facebookresearch/XLM
- https://github.com/google-research/language/tree/master/language/table_text_eval
- https://github.com/OpenNMT/OpenNMT-py