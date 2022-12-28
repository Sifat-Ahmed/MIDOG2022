## This is the documentation of my source code for MIDOG2022

### Directory 
***
``` 
midog
├── configs
│   ├── preprocess.yaml                     
│   ├── train.yaml                       
├── data (ignored in .gitignore)                    
│   ├── directory for images and json files                     
├── retinanet                            
├── readme.md                           
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

if the installation is successful now you can run the code.

<br>