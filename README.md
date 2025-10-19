# ESM-PROTEIN-MUTATION: Protein Mutation Harmfulness Assessment Based on ESM
-----------------------------------------------------------------

## 1. Introduction
**ESM-PROTEIN-MUTATION** is a tool built on top of [ESM (Evolutionary Scale Modeling)](https://github.com/facebookresearch/esm) for evaluating the potential harmfulness of amino acid mutations in proteins.  
By computing the conditional probabilities of wild-type and mutant amino acids under the ESM language model, this tool provides a quantitative score (Δscore) that reflects the functional impact of mutations. This makes it useful for function prediction, pathogenicity assessment, and biomedical research.

## Features
- Input a protein sequence and specify a mutation (position, wild-type residue, mutant residue)  
- Utilize the pre-trained ESM-1b model to compute amino acid probabilities  
- Output the **Δscore** (log-likelihood ratio) as a quantitative indicator of mutation harmfulness  
- Support for both direct sequence input and FASTA file input  

## Use Cases
- Protein function analysis  
- Pathogenic mutation prediction  
- Protein engineering and drug design  
- Bioinformatics and computational biology research  


## Core Algorithm

Given a protein sequence $S = (s_1, s_2, \ldots, s_n)$ and a mutation at position $i$,  
where the wild-type amino acid is $a_{\text{wt}}$ and the mutant is $a_{\text{mut}}$.  

The ESM model computes the conditional probability of an amino acid at position \( i \) given the masked sequence context:

$P(a \mid S_{\setminus i})$

Wild-type and mutant probabilities are:

$p_{\text{wt}} = P(a_{\text{wt}} \mid S_{\setminus i}), \quad p_{\text{mut}} = P(a_{\text{mut}} \mid S_{\setminus i})$

The mutation score (Δscore) is defined as the log-likelihood ratio:

$\Delta = \log \frac{p_{\text{mut}}}{p_{\text{wt}}}$

- If $\Delta < 0$: the mutation is less probable than the wild type → potentially harmful.  
- If $\Delta > 0$: the mutation is more probable than the wild type → likely tolerated.  
  

## 2. Usage
Clone this repo and run following command:

```
docker run --rm --gpus all -v %cd%:/workspace -w /workspace duangao/esm-protein-mutation:latest python main.py --seq MLLRSSGKLSVGTKKEDGESTAPTPRPKILRCKCHHHCPEDSVNNICSTDGYCFTMIEEDDSGMPVVTSGCLGLEGSDFQCRDTPIPHQRRSIECCTERNECNKDLHPTLPPLKNRDFVDGPIHHKALLISVTVCSLLLVLIILFCYFRYKRQEARPRYSIGLEQDETYIPPGESLRDLIEQSQSSGSGSGLPLLVQRTIAKQIQMVKQIGKGRYGEVWMGKWRGEKVAVKVFFTTEEASWFRETEIYQTVLMRHENILGFIAADIKGTGSWTQLYLITDYHENGSLYDYLKSTTLDTKSMLKLAYSAVSGLCHLHTEIFSTQGKPAIAHRDLKSKNILVKKNGTCCIADLGLAVKFISDTNEVDIPPNTRVGTKRYMPPEVLDESLNRNHFQSYIMADMYSFGLILWEVARRCVSGGIVEEYQLPYHDLVPSDPSYEDMREIVCIKKLRPSFPNRWSSDECLRQMGKLMTECWAHNPASRLTALRVKKTLAKMSESQDIKL --pos 248 --wt Q --mut R
```
Change the -v hyperparameters from % to $ if you run in a linux command line.
### Example output

```
Elapsed time: 51.85 seconds
Wild type amino acid Q likelihood: 0.048806
Mutant type amino acid R likelihood: 0.000183
Δscore (log likelihood ratio): -5.5867
```
### PDB making
```
python make_pdb.py
```
use docker file:
duangao/esm-protein-mutation:v1

## 3. Code block interpretation  

### ✅ Description of the code block(1)

This script performs the following operations:

1. Generates sequences with applied masking;
2. Converts the masked sequences into tokenized representations;
3. Inputs the resulting tokens into the model for training.

---

### ✅ Code block

```python
    batch_converter = alphabet.get_batch_converter()

    seq = list(wildtype_sequence)
    seq[mutation_position] = alphabet.get_tok(alphabet.mask_idx)
    masked_sequence_str = "".join(seq)

    data = [("protein", masked_sequence_str)]
    _, _, batch_tokens = batch_converter(data)

```

---

### ✅ source
https://github.com/facebookresearch/esm/blob/main/esm/data.py
```python
      class Alphabet(object):
      def __init__(
          self,
          standard_toks: Sequence[str],
          prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
          append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
          prepend_bos: bool = True,
          append_eos: bool = False,
          use_msa: bool = False,
      ):
          self.standard_toks = list(standard_toks)
          self.prepend_toks = list(prepend_toks)
          self.append_toks = list(append_toks)
          self.prepend_bos = prepend_bos
          self.append_eos = append_eos
          self.use_msa = use_msa

          self.all_toks = list(self.prepend_toks)
          self.all_toks.extend(self.standard_toks)
          for i in range((8 - (len(self.all_toks) % 8)) % 8):
              self.all_toks.append(f"<null_{i  + 1}>")
          self.all_toks.extend(self.append_toks)

          self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

          self.unk_idx = self.tok_to_idx["<unk>"]
          self.padding_idx = self.get_idx("<pad>")
          self.cls_idx = self.get_idx("<cls>")
          self.mask_idx = self.get_idx("<mask>")
          self.eos_idx = self.get_idx("<eos>")
          self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
          self.unique_no_split_tokens = self.all_toks

      def __len__(self):
          return len(self.all_toks)

      def get_idx(self, tok):
          return self.tok_to_idx.get(tok, self.unk_idx)

      def get_tok(self, ind):
          return self.all_toks[ind]

      def to_dict(self):
          return self.tok_to_idx.copy()

      def get_batch_converter(self, truncation_seq_length: int = None):
          if self.use_msa:
              return MSABatchConverter(self, truncation_seq_length)
          else:
              return BatchConverter(self, truncation_seq_length)

      @classmethod
      def from_architecture(cls, name: str) -> "Alphabet":
          if name in ("ESM-1", "protein_bert_base"):
              standard_toks = proteinseq_toks["toks"]
              prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
              append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
              prepend_bos = True
              append_eos = False
              use_msa = False
          elif name in ("ESM-1b", "roberta_large"):
              standard_toks = proteinseq_toks["toks"]
              prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
              append_toks = ("<mask>",)
              prepend_bos = True
              append_eos = True
              use_msa = False
          elif name in ("MSA Transformer", "msa_transformer"):
              standard_toks = proteinseq_toks["toks"]
              prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
              append_toks = ("<mask>",)
              prepend_bos = True
              append_eos = False
              use_msa = True
          elif "invariant_gvp" in name.lower():
              standard_toks = proteinseq_toks["toks"]
              prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
              append_toks = ("<mask>", "<cath>", "<af2>")
              prepend_bos = True
              append_eos = False
              use_msa = False
          else:
              raise ValueError("Unknown architecture selected")
          return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)

      def _tokenize(self, text) -> str:
          return text.split()

      def tokenize(self, text, **kwargs) -> List[str]:
          """
          Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
          Converts a string in a sequence of tokens, using the tokenizer.

          Args:
              text (:obj:`str`):
                  The sequence to be encoded.

          Returns:
              :obj:`List[str]`: The list of tokens.
          """

          def split_on_token(tok, text):
              result = []
              split_text = text.split(tok)
              for i, sub_text in enumerate(split_text):
                  # AddedToken can control whitespace stripping around them.
                  # We use them for GPT2 and Roberta to have different behavior depending on the special token
                  # Cf. https://github.com/huggingface/transformers/pull/2778
                  # and https://github.com/huggingface/transformers/issues/3788
                  # We strip left and right by default
                  if i < len(split_text) - 1:
                      sub_text = sub_text.rstrip()
                  if i > 0:
                      sub_text = sub_text.lstrip()

                  if i == 0 and not sub_text:
                      result.append(tok)
                  elif i == len(split_text) - 1:
                      if sub_text:
                          result.append(sub_text)
                      else:
                          pass
                  else:
                      if sub_text:
                          result.append(sub_text)
                      result.append(tok)
              return result

          def split_on_tokens(tok_list, text):
              if not text.strip():
                  return []

              tokenized_text = []
              text_list = [text]
              for tok in tok_list:
                  tokenized_text = []
                  for sub_text in text_list:
                      if sub_text not in self.unique_no_split_tokens:
                          tokenized_text.extend(split_on_token(tok, sub_text))
                      else:
                          tokenized_text.append(sub_text)
                  text_list = tokenized_text

              return list(
                  itertools.chain.from_iterable(
                      (
                          self._tokenize(token)
                          if token not in self.unique_no_split_tokens
                          else [token]
                          for token in tokenized_text
                      )
                  )
              )

          no_split_token = self.unique_no_split_tokens
          tokenized_text = split_on_tokens(no_split_token, text)
          return tokenized_text

      def encode(self, text):
          return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


```

---

### 🧩 Examples

**Before Execution**

```
seq = ['M', 'L', 'L', 'R', 'S', 'S', 'G', 'K', 'L', 'S', 'V', 'G', 'T', 'K', 'K', 'E', 'D', 'G', 'E', 'S', 'T', 'A', 'P', 'T', 'P', 'R', 'P', 'K', 'I', 'L', 'R', 'C', 'K', 'C', 'H', 'H', 'H', 'C', 'P', 'E', 'D', 'S', 'V', 'N', 'N', 'I', 'C', 'S', 'T', 'D', 'G', 'Y', 'C', 'F', 'T', 'M', 'I', 'E', 'E', 'D', 'D', 'S', 'G', 'M', 'P', 'V', 'V', 'T', 'S', 'G', 'C', 'L', 'G', 'L', 'E', 'G', 'S', 'D', 'F', 'Q', 'C', 'R', 'D', 'T', 'P', 'I', 'P', 'H', 'Q', 'R', 'R', 'S', 'I', 'E', 'C', 'C', 'T', 'E', 'R', 'N', 'E', 'C', 'N', 'K', 'D', 'L', 'H', 'P', 'T', 'L', 'P', 'P', 'L', 'K', 'N', 'R', 'D', 'F', 'V', 'D', 'G', 'P', 'I', 'H', 'H', 'K', 'A', 'L', 'L', 'I', 'S', 'V', 'T', 'V', 'C', 'S', 'L', 'L', 'L', 'V', 'L', 'I', 'I', 'L', 'F', 'C', 'Y', 'F', 'R', 'Y', 'K', 'R', 'Q', 'E', 'A', 'R', 'P', 'R', 'Y', 'S', 'I', 'G', 'L', 'E', 'Q', 'D', 'E', 'T', 'Y', 'I', 'P', 'P', 'G', 'E', 'S', 'L', 'R', 'D', 'L', 'I', 'E', 'Q', 'S', 'Q', 'S', 'S', 'G', 'S', 'G', 'S', 'G', 'L', 'P', 'L', 'L', 'V', 'Q', 'R', 'T', 'I', 'A', 'K', 'Q', 'I', 'Q', 'M', 'V', 'K', 'Q', 'I', 'G', 'K', 'G', 'R', 'Y', 'G', 'E', 'V', 'W', 'M', 'G', 'K', 'W', 'R', 'G', 'E', 'K', 'V', 'A', 'V', 'K', 'V', 'F', 'F', 'T', 'T', 'E', 'E', 'A', 'S', 'W', 'F', 'R', 'E', 'T', 'E', 'I', 'Y', 'Q', 'T', 'V', 'L', 'M', 'R', 'H', 'E', 'N', 'I', 'L', 'G', 'F', 'I', 'A', 'A', 'D', 'I', 'K', 'G', 'T', 'G', 'S', 'W', 'T', 'Q', 'L', 'Y', 'L', 'I', 'T', 'D', 'Y', 'H', 'E', 'N', 'G', 'S', 'L', 'Y', 'D', 'Y', 'L', 'K', 'S', 'T', 'T', 'L', 'D', 'T', 'K', 'S', 'M', 'L', 'K', 'L', 'A', 'Y', 'S', 'A', 'V', 'S', 'G', 'L', 'C', 'H', 'L', 'H', 'T', 'E', 'I', 'F', 'S', 'T', 'Q', 'G', 'K', 'P', 'A', 'I', 'A', 'H', 'R', 'D', 'L', 'K', 'S', 'K', 'N', 'I', 'L', 'V', 'K', 'K', 'N', 'G', 'T', 'C', 'C', 'I', 'A', 'D', 'L', 'G', 'L', 'A', 'V', 'K', 'F', 'I', 'S', 'D', 'T', 'N', 'E', 'V', 'D', 'I', 'P', 'P', 'N', 'T', 'R', 'V', 'G', 'T', 'K', 'R', 'Y', 'M', 'P', 'P', 'E', 'V', 'L', 'D', 'E', 'S', 'L', 'N', 'R', 'N', 'H', 'F', 'Q', 'S', 'Y', 'I', 'M', 'A', 'D', 'M', 'Y', 'S', 'F', 'G', 'L', 'I', 'L', 'W', 'E', 'V', 'A', 'R', 'R', 'C', 'V', 'S', 'G', 'G', 'I', 'V', 'E', 'E', 'Y', 'Q', 'L', 'P', 'Y', 'H', 'D', 'L', 'V', 'P', 'S', 'D', 'P', 'S', 'Y', 'E', 'D', 'M', 'R', 'E', 'I', 'V', 'C', 'I', 'K', 'K', 'L', 'R', 'P', 'S', 'F', 'P', 'N', 'R', 'W', 'S', 'S', 'D', 'E', 'C', 'L', 'R', 'Q', 'M', 'G', 'K', 'L', 'M', 'T', 'E', 'C', 'W', 'A', 'H', 'N', 'P', 'A', 'S', 'R', 'L', 'T', 'A', 'L', 'R', 'V', 'K', 'K', 'T', 'L', 'A', 'K', 'M', 'S', 'E', 'S', 'Q', 'D', 'I', 'K', 'L']


```

**After Execution**

```
batch_tokens = tensor([[ 0, 20,  4,  4, 10,  8,  8,  6, 15,  4,  8,  7,  6, 11, 15, 15,  9, 13,
          6,  9,  8, 11,  5, 14, 11, 14, 10, 14, 15, 12,  4, 10, 23, 15, 23, 21,
         21, 21, 23, 14,  9, 13,  8,  7, 17, 17, 12, 23,  8, 11, 13,  6, 19, 23,
         18, 11, 20, 12,  9,  9, 13, 13,  8,  6, 20, 14,  7,  7, 11,  8,  6, 23,
          4,  6,  4,  9,  6,  8, 13, 18, 16, 23, 10, 13, 11, 14, 12, 14, 21, 16,
         10, 10,  8, 12,  9, 23, 23, 11,  9, 10, 17,  9, 23, 17, 15, 13,  4, 21,
         14, 11,  4, 14, 14,  4, 15, 17, 10, 13, 18,  7, 13,  6, 14, 12, 21, 21,
         15,  5,  4,  4, 12,  8,  7, 11,  7, 23,  8,  4,  4,  4,  7,  4, 12, 12,
          4, 18, 23, 19, 18, 10, 19, 15, 10, 16,  9,  5, 10, 14, 10, 19,  8, 12,
          6,  4,  9, 16, 13,  9, 11, 19, 12, 14, 14,  6,  9,  8,  4, 10, 13,  4,
         12,  9, 16,  8, 16,  8,  8,  6,  8,  6,  8,  6,  4, 14,  4,  4,  7, 16,
         10, 11, 12,  5, 15, 16, 12, 16, 20,  7, 15, 16, 12,  6, 15,  6, 10, 19,
          6,  9,  7, 22, 20,  6, 15, 22, 10,  6,  9, 15,  7,  5,  7, 15,  7, 18,
         18, 11, 11,  9,  9,  5,  8, 22, 18, 10,  9, 11,  9, 12, 19, 32, 11,  7,
          4, 20, 10, 21,  9, 17, 12,  4,  6, 18, 12,  5,  5, 13, 12, 15,  6, 11,
          6,  8, 22, 11, 16,  4, 19,  4, 12, 11, 13, 19, 21,  9, 17,  6,  8,  4,
         19, 13, 19,  4, 15,  8, 11, 11,  4, 13, 11, 15,  8, 20,  4, 15,  4,  5,
         19,  8,  5,  7,  8,  6,  4, 23, 21,  4, 21, 11,  9, 12, 18,  8, 11, 16,
          6, 15, 14,  5, 12,  5, 21, 10, 13,  4, 15,  8, 15, 17, 12,  4,  7, 15,
         15, 17,  6, 11, 23, 23, 12,  5, 13,  4,  6,  4,  5,  7, 15, 18, 12,  8,
         13, 11, 17,  9,  7, 13, 12, 14, 14, 17, 11, 10,  7,  6, 11, 15, 10, 19,
         20, 14, 14,  9,  7,  4, 13,  9,  8,  4, 17, 10, 17, 21, 18, 16,  8, 19,
         12, 20,  5, 13, 20, 19,  8, 18,  6,  4, 12,  4, 22,  9,  7,  5, 10, 10,
         23,  7,  8,  6,  6, 12,  7,  9,  9, 19, 16,  4, 14, 19, 21, 13,  4,  7,
         14,  8, 13, 14,  8, 19,  9, 13, 20, 10,  9, 12,  7, 23, 12, 15, 15,  4,
         10, 14,  8, 18, 14, 17, 10, 22,  8,  8, 13,  9, 23,  4, 10, 16, 20,  6,
         15,  4, 20, 11,  9, 23, 22,  5, 21, 17, 14,  5,  8, 10,  4, 11,  5,  4,
         10,  7, 15, 15, 11,  4,  5, 15, 20,  8,  9,  8, 16, 13, 12, 15,  4,  2]])
```

---

### ⚙️ Parameter Description

| Parameter | Purpose |
| --- | --- |
| alphabet.mask_idx | 12, int type |

### ✅ Description of the code block(2)

This script performs the following operations:

1. Input the Token into the transformer model and get the logits.

---

### ✅ Code block

```python
    with torch.no_grad():
        outputs = model(batch_tokens, repr_layers=[])
        logits = outputs["logits"]
```

---

### ✅ source
https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py
```python
      def load_model_and_alphabet_core(model_name, model_data, regression_data=None):
        if regression_data is not None:
            model_data["model"].update(regression_data["model"])

        if model_name.startswith("esm2"):
            model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)
        else:
            model, alphabet, model_state = _load_model_and_alphabet_core_v1(model_data)

        expected_keys = set(model.state_dict().keys())
        found_keys = set(model_state.keys())

        if regression_data is None:
            expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
            error_msgs = []
            missing = (expected_keys - found_keys) - expected_missing
            if missing:
                error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
            unexpected = found_keys - expected_keys
            if unexpected:
                error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

            if error_msgs:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
            if expected_missing - found_keys:
                warnings.warn(
                    "Regression weights not found, predicting contacts will not produce correct results."
                )

        model.load_state_dict(model_state, strict=regression_data is not None)

        return model, alphabet


```

---

### 🧩 Examples

**Before Execution**

```
batch_tokens = tensor([[ 0, 20,  4,  4, 10,  8,  8,  6, 15,  4,  8,  7,  6, 11, 15, 15,  9, 13,
          6,  9,  8, 11,  5, 14, 11, 14, 10, 14, 15, 12,  4, 10, 23, 15, 23, 21,
         21, 21, 23, 14,  9, 13,  8,  7, 17, 17, 12, 23,  8, 11, 13,  6, 19, 23,
         18, 11, 20, 12,  9,  9, 13, 13,  8,  6, 20, 14,  7,  7, 11,  8,  6, 23,
          4,  6,  4,  9,  6,  8, 13, 18, 16, 23, 10, 13, 11, 14, 12, 14, 21, 16,
         10, 10,  8, 12,  9, 23, 23, 11,  9, 10, 17,  9, 23, 17, 15, 13,  4, 21,
         14, 11,  4, 14, 14,  4, 15, 17, 10, 13, 18,  7, 13,  6, 14, 12, 21, 21,
         15,  5,  4,  4, 12,  8,  7, 11,  7, 23,  8,  4,  4,  4,  7,  4, 12, 12,
          4, 18, 23, 19, 18, 10, 19, 15, 10, 16,  9,  5, 10, 14, 10, 19,  8, 12,
          6,  4,  9, 16, 13,  9, 11, 19, 12, 14, 14,  6,  9,  8,  4, 10, 13,  4,
         12,  9, 16,  8, 16,  8,  8,  6,  8,  6,  8,  6,  4, 14,  4,  4,  7, 16,
         10, 11, 12,  5, 15, 16, 12, 16, 20,  7, 15, 16, 12,  6, 15,  6, 10, 19,
          6,  9,  7, 22, 20,  6, 15, 22, 10,  6,  9, 15,  7,  5,  7, 15,  7, 18,
         18, 11, 11,  9,  9,  5,  8, 22, 18, 10,  9, 11,  9, 12, 19, 32, 11,  7,
          4, 20, 10, 21,  9, 17, 12,  4,  6, 18, 12,  5,  5, 13, 12, 15,  6, 11,
          6,  8, 22, 11, 16,  4, 19,  4, 12, 11, 13, 19, 21,  9, 17,  6,  8,  4,
         19, 13, 19,  4, 15,  8, 11, 11,  4, 13, 11, 15,  8, 20,  4, 15,  4,  5,
         19,  8,  5,  7,  8,  6,  4, 23, 21,  4, 21, 11,  9, 12, 18,  8, 11, 16,
          6, 15, 14,  5, 12,  5, 21, 10, 13,  4, 15,  8, 15, 17, 12,  4,  7, 15,
         15, 17,  6, 11, 23, 23, 12,  5, 13,  4,  6,  4,  5,  7, 15, 18, 12,  8,
         13, 11, 17,  9,  7, 13, 12, 14, 14, 17, 11, 10,  7,  6, 11, 15, 10, 19,
         20, 14, 14,  9,  7,  4, 13,  9,  8,  4, 17, 10, 17, 21, 18, 16,  8, 19,
         12, 20,  5, 13, 20, 19,  8, 18,  6,  4, 12,  4, 22,  9,  7,  5, 10, 10,
         23,  7,  8,  6,  6, 12,  7,  9,  9, 19, 16,  4, 14, 19, 21, 13,  4,  7,
         14,  8, 13, 14,  8, 19,  9, 13, 20, 10,  9, 12,  7, 23, 12, 15, 15,  4,
         10, 14,  8, 18, 14, 17, 10, 22,  8,  8, 13,  9, 23,  4, 10, 16, 20,  6,
         15,  4, 20, 11,  9, 23, 22,  5, 21, 17, 14,  5,  8, 10,  4, 11,  5,  4,
         10,  7, 15, 15, 11,  4,  5, 15, 20,  8,  9,  8, 16, 13, 12, 15,  4,  2]])


```

**After Execution**

```
logits = [0.0821, 0.0357, 0.0612, 0.0489, 0.0935, 0.0276, 0.0741, 0.0598, 0.0193, 0.0872, 0.0684, 0.0425, 0.0391, 0.0716, 0.0538, 0.0219, 0.0657, 0.0463, 0.0582, 0.0337]

```

---


## 4. Contact  

**DUAN GAO** < gaoduan666@gmail.com >  

## Citations <a name="citations"></a>

```bibtex
@article{Meier2021LanguageME,
  title={Language models enable zero-shot prediction of the effects of mutations on protein function},
  author={Joshua Meier and Roshan Rao and Robert Verkuil and Jason Liu and Tom Sercu and Alexander Rives},
  journal={bioRxiv},
  year={2021},
  url={https://api.semanticscholar.org/CorpusID:235793688}
}
```