# EduEmo
Source code for the paper "Elementary Discourse Units with Sparse Attention for Multi-label Emotion"
## Requirement
python 3.7  
torch 1.9.0  
Other packages can be installed via:

    pip install -r requirements.txt
    
## Usage
#### Preprocessing
- SemEval18: Download EDU segmentation toolkits [SegBot](http://138.197.118.157:8000/segbot/) and run *data/preprocess.py*. 
- Ren_CECps: The use of this data set requires permission from the author. Download EDU segmentation toolkits ([here](https://github.com/abccaba2000/discourse-parser)) and run *parser.py*.

#### Training

    python scripts/train.py 

#### Evaluation1

    python scripts/test.py



