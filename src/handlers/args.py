import os
import argparse

def setup_argparse():
    
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--comp', type=str, default='DL', choices=['KD', 'DL', 'SDR', 'DR', 'SR'],
                    help='Components to choose: [KD, DL, SDR, DR, SR]')

    parser.add_argument('--data-train', nargs='*', default=[],
                        help='training files; supported syntax:'
                             ' (a) plain list, `--data-train /path/to/a/* /path/to/b/*`;'
                             ' (b) (named) groups [Recommended], `--data-train a:/path/to/a/* b:/path/to/b/*`,'
                             ' the file splitting (for each dataloader worker) will be performed per group,'
                             ' and then mixed together, to ensure a uniform mixing from all groups for each worker.'
                        )
    parser.add_argument('--data-val', nargs='*', default=[],
                        help='validation files; when not set, will use training files and split by `--train-val-split`')
    parser.add_argument('--data-test', nargs='*', default=[],
                        help='testing files; supported syntax:'
                             ' (a) plain list, `--data-test /path/to/a/* /path/to/b/*`;'
                             ' (b) keyword-based, `--data-test a:/path/to/a/* b:/path/to/b/*`, will produce output_a, output_b;'
                             ' (c) split output per N input files, `--data-test a%%10:/path/to/a/*`, will split per 10 input files')
    parser.add_argument('--data-config', type=str, default = None)
    
    parser.add_argument('--data-fraction', type=float, default=0.1,
                        help='fraction of events to load from each file; for training, the events are randomly selected for each epoch')
    parser.add_argument('--file-fraction', type=float, default=1,
                        help='fraction of files to load; for training, the files are randomly selected for each epoch')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=0)
    
    parser.add_argument('--optimizer', type=str, default = 'ranger')
    parser.add_argument('--optimizer-option', nargs=2, action='append', default=[],
                    help='options to pass to the optimizer class constructor, e.g., `--optimizer-option weight_decay 1e-4`')
    parser.add_argument('--start-lr', type=float, default=1e-03)
    parser.add_argument('--final-lr', type=float, default=1e-06)
    parser.add_argument('--lr-scheduler', type=str, default = 'flat+decay')
    
    parser.add_argument('--model-network', type=str, default = None)
    parser.add_argument('--model-path', type=str, default = None)
    parser.add_argument('--model-prefix', type=str, default = None)

    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--metrics-prefix', type=str, default=None)
    parser.add_argument('--figures-prefix', type=str, default=None)

# Knowledge Distillation
    
    parser.add_argument('--teacher-network', type=str, default = None)
    parser.add_argument('--teacher-path', type=str, default = None)
    parser.add_argument('--teacher-prefix', type=str, default = None)
    parser.add_argument('--kd-temp', type=float, default=1)
    parser.add_argument('--kd-anneal', action='store_true', default=False)
    parser.add_argument('--kd-weight', type=float, default=1)

# Dimensionality Reduction
    
    parser.add_argument('--dr-method', type=str, default='encoder', help = 'Available Options: encoder')
    parser.add_argument('--dr-network', type=str, default=None)
    parser.add_argument('--dr-path', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=4.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--kl-anneal', action='store_true', default=False)
    parser.add_argument('--svae-weight', type=float, default=1.0)

# Symbolic Regression

    parser.add_argument('--sr-prefix', type=str, default=None)
    
    parser.add_argument('--max-size', type=int, default=40)
    parser.add_argument('--n-iterations', type=int, default=5600)
    parser.add_argument('--n-populations', type=int, default=48)
    parser.add_argument('--population-size', type=int, default=27)
    parser.add_argument('--iteration-cycles', type=int, default=1520)
    
    parser.add_argument('--weight-optimize', type=float, default = 0.001)
    parser.add_argument('--parsimony', type=float, default=0.01)
    parser.add_argument('--sr-loss', type=str, default='HuberLoss(1.35)')
    
    parser.add_argument('--binary-operators', default=[
        "+", 
        "-", 
        "*", 
        "/", 
        "^",
    ])
    parser.add_argument('--unary-operators', default=[
        "sqrt", 
        "tanh",
        "sin",
    ])
    parser.add_argument('--constraints', default={
        '^': (-1, 1)
    })
    parser.add_argument('--nested-constraints', default={
        "*": {"tanh": 2},
        "tanh": {"tanh": 0, "^": 1, "sin": 1},
        "sin": {"sin": 0}       
    })

    parser.add_argument('--sr-annealing', action='store_true', default = True)
    parser.add_argument('--sr-batching', action='store_true', default=True)
    
    return parser