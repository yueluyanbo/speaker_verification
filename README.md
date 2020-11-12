## Speaker verification

This demo base on "clovaai/voxceleb_trainer".
Utilize voxceleb pretrained model "baseline_lite_ap.model", with EER = 2.1792, threshold = -0.9860 in voxceleb1 test data.

### How to run this demo

#### Step-1 Register
Use interactive user input, and input the user name in termial
When the user see "Recording...", please read the randomly displayed numbers

Before running the following code, you need to change "path = "~/voxceleb_trainer" " to your local path
```
python3 register.py
```

#### Step-2 Show the registration list
Before running the following code, you need to change "path = "~/voxceleb_trainer" " to your local path
```
python3 show_registration_list.py
```

#### Step-3 Verify
Before running the following code, you need to change "path = "~/voxceleb_trainer" " to your local path
```
python3 verify.py
```
