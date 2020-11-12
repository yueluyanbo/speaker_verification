#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import yaml
import numpy
import pdb
import torch
import glob
import random
from tuneThreshold import tuneThresholdfromScore
from SpeakerNet_evaluate import SpeakerNet_evaluate
from DatasetLoader import get_data_loader
from Record import Record

parser = argparse.ArgumentParser(description = "SpeakerNet");

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file');

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=400,    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch');
parser.add_argument('--max_seg_per_spk', type=int,  default=100,    help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads');
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="angleproto",     help='Loss function');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate');
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin',         type=float, default=1,      help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=15,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="pretrained_model/baseline_lite_ap.model",     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="exps/test", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list',     type=str,   default="",     help='Train list');
parser.add_argument('--test_list',      type=str,   default="",     help='Evaluation list');
parser.add_argument('--wav1',      type=str,   default="",     help='wav1 file');
parser.add_argument('--wav2',      type=str,   default="",     help='wav2 file');
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set');
parser.add_argument('--test_path',      type=str,   default="tmp/registered", help='Absolute path to the test set');
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set');
parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set');

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks');
parser.add_argument('--log_input',      type=bool,  default=True,  help='Log input features')
parser.add_argument('--model',          type=str,   default="ResNetSE34L",     help='Name of model definition');
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer');

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', default=True, help='Eval only')

args = parser.parse_args();

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

## Initialise directories
model_save_path     = args.save_path + "/model"
result_save_path    = args.save_path + "/result"

if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)
        
if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)

## Load models
s = SpeakerNet_evaluate(**vars(args));

it          = 1;
prevloss    = float("inf");
sumloss     = 0;
min_eer     = [100];

## Load model weights
modelfiles = glob.glob('%s/model0*.model'%model_save_path)
modelfiles.sort()

if len(modelfiles) >= 1:
    s.loadParameters(modelfiles[-1]);
    print("Model %s loaded from previous state!"%modelfiles[-1]);
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
elif(args.initial_model != ""):
    s.loadParameters(args.initial_model);
    print("Model %s loaded!"%args.initial_model);

for ii in range(0,it-1):
    s.__scheduler__.step()
    
## make decision
threshold = -0.9860221743583679
def predict(sc, th):
    if sc > th:
        return 1
    else:
        return 0

def process(testdir):
    # Use interactive user input
    # Input the user name in termial
    test_name = input("Enter your name: ")   
    testfile = testdir + "/1.wav"
    
    # Record an wav file to the test dir
    if os.path.exists(testfile):
        os.remove(testfile)
 
    res = [random.randrange(0, 9, 1) for i in range(10)] 
    print("Please read the following numbers when you see" + ' "Recording..."' + ": ")
    print(str(res))
    
    Record.record(testfile) 
    return test_name.lower()
    
## Evaluation code
if args.eval == True:
    path = "~/registered"
    testdir = path + '/test'

    # Create target Directory if don't exist
    if not os.path.exists(testdir):
        os.mkdir(testdir)

    test_name = process(testdir) # perform some operation(s) on given string    
     
    wav1 = test_name.lower() + "/1.wav"
    wav2 = "test/1.wav"
    
    score, trials = s.evaluateFromList(wav1, wav2, print_interval=100, test_path=args.test_path, eval_frames=args.eval_frames)
    print("*******************************************************************")
    print("Predict score is: ", round(score, 4))
    print("The threshold is: ", round(threshold, 4))
    print("")
    
    if predict(score, threshold) == 1:
        print("Pass! The test user is " + test_name + "!")
    else:
        print("Reject! The test user is NOT " + test_name + "!")
    print("*******************************************************************")


