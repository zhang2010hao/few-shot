This repo provides a Pytorch implementation of match network and prototypical network for NLP.

## Dataset
I make a few-shot dataset from NYT. This dataset contains 989 sentences. The task aims at predicting same entity in sentence.


## Acknowledgements
Special thanks to https://github.com/gitabcworld/MatchingNetworks and https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch for their implementations. 

## Training

You can train match network and prototypical network with src/mainNYT.py. If you train match network, you should use fit_match_net(); If you train prototypical network, you should use fit_prototypical_net();


$ python mainNYT.py

The script takes the following command line options:

- `dataset_root`: the root directory where tha dataset is stored, default to `'../NYT'`

You can ajust other parameters in src/config.py

## Performance
                            val acc     test acc
    match network           0.4769      0.4423        
    prototypical network    0.70625     0.6938
