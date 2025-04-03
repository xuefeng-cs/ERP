# install

## To train or use a transformer

```sh
# conda remove --name erp_mcts_llm --all
conda create --name erp_mcts_llm python=3.11
conda activate erp_mcts_llm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# conda install pytorch torchvision torchaudio cpuonly -c pytorch  # if cpu only
pip install -r requirements.txt
```
