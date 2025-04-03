# Entropy-Reinforced Planning with Large Language Models for Drug Discovery

The implemtation of the paper _Entropy-Reinforced Planning with Large Language Models for Drug Discovery_ ([paper](https://proceedings.mlr.press/v235/liu24be.html), [preprint](https://arxiv.org/abs/2406.07025))

## Dependencies

See `install.md`.

## MC Tree Search

```sh
# e.g. 1
python search.py mol_metric.docking_dataset=cancer nn.checkpoint=jarod0411/zinc10M_gpt2_SMILES_bpe_combined_step1_finetune

# e.g. 2
python search.py mol_metric.docking_dataset=covid nn.checkpoint=jarod0411/zinc10M_gpt2_SMILES_bpe_combined_step1_finetune_covid
```

## Citation
```
@inproceedings{liu2024entropy,
  title={Entropy-Reinforced Planning with Large Language Models for Drug Discovery},
  author={Liu, Xuefeng and Tien, Chih-Chan and Ding, Peng and Jiang, Songhao and Stevens, Rick L},
  booktitle={International Conference on Machine Learning},
  pages={31917--31932},
  year={2024},
  organization={PMLR}
}
```
