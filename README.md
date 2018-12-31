# NMDNET

NMDNET is a Python library which implements the tests and benchmarks carried for https://arxiv.org/abs/1812.09113.

## Usage

Go into the top level directory, that is "cd .../nmd_net/". From that directory, launch the "meta_rl_launcher" as:

```bash
./ReinforcementLearning/Training/meta_rl_launcher.py
```

1. -b option allows to chose de benchmark
2. -g options allows to chose wheter to run the program with or without GPU. GPU is active with the -g option.
3. -e sets the episodes budget
4. -t sets the type of architecture to use. "recurrent" for classic rnn and "nmdnet" for NMD nets.
4. -tb, -bs and -o options allow to chose a number of tests to launch and the starting test number. When running the program, it will start by running test number "bs * tb + o" and once done will run test number "bs * tb + o + 1" and so on until reaching test "bs * (tb + 1)"
5. -l allows to chose whether to start from scratch or load a previous test (which must exist). -l checkpoint will load the latest network trained for that particular test number.
