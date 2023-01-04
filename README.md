## This is the documentation of my source code for MIDOG2022
Challenge details can be found <a href="https://imig.science/midog/the-challgenge/" target="_blank">here</a>.

For this task, I have done classification between:
1.  Mitotic Figure
2.  Non-mitotic figure

and their corresponding bounding box prediction . This is on the initial state. It will be updated time to time.

### Directory 
***
``` 
midog
├── configs
│   ├── preprocess.yaml                     
│   ├── train.yaml                       
├── data (ignored in .gitignore)                    
│   ├── directory for images and json files
├── dataset
│   ├── __init__.py
│   ├── dataset.py                       
│   ├── utils.py
├── outputs (ignored in .gitignore)
│   ├── directory for hydra logs                       
├── retinanet
│   ├── __init__.py
│   ├── anchors.py
│   ├── coco_eval.py
│   ├── losses.py
│   ├── model.py
│   ├── utils.py                            
├── runs (ignored in .gitignore)
│   ├── contains all the saved files of the model                           
├── preprocess.py                           
├── train.py                           
├── README.md                           
├── requirements.txt                    
                            
```
<br>

### Installation
***
1. First create a virtual environment using PIP (For **Windows** system)
- if pip is not up-to-date
```
python -m pip install --upgrade pip                 # upgrade pip
python -m pip install --user virtualenv             # installing virtualenv
```
- creating a virtual environment. It will create an environment named env in the project folder
```
python -m venv env                                  #(env is the name of the env)
```
- activate the virtual environment
```
.\env\Scripts\activate
```
- now install the dependencies
```
pip install -r requirement.txt
```

If the installation is successful now you can run the code.
<br>
### Editing the config files

Inside the configs folder there are several configuration files. These config files
are parsed using hydra. For the time being there are two different config files.

`preprocess.py`
```
read_dir:
  image: 'path/to/coco/images'
  json: 'path/to/MIDOG2022_training.json'

write_dir:
  image: 'path/to/save/new/image/slice'
  train_json: 'path/to/save/training.json'        ## this should contain the filename also
  val_json: 'path/to/save/validation.json'        ## this should also contain the filename with extension
size:
  step: 512
  patch: 512

```