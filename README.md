Transfer-Learning-With-DELTA

# Run the setup script
```
curl https://raw.githubusercontent.com/njacobs2019/Transfer-Learning-With-DELTA/main/setup.sh -o setup.sh && chmod +x setup.sh && ./setup.sh
```

# channel_eval.py
** Takes ~15.5 minutes **
This file determines the regularization weights for each filter.  It saves them to disk and needs to be run before train.


Running tensorboard:
```
tensorboard --logdir=runs
tensorboard --logdir ./logs --host 0.0.0.0 --port 6006
```