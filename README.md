# ML Operations Group 73

This is the cookiecutter template we'll use for our dtu mlops project

##  Names and Student numbers
Adrià Bosch Matas s232775
<br/>Edgar Fabregat Calbet s242781
<br/>Paula Gambús Moreno s233219
<br/>Jofre Bonillo Mesegué s240661

## Project Description
In this project we will focus on the [RecSys Challenge from Ekstra Bladet](https://recsys.eb.dk/). The challenge consists on building a news recommender system capable of ordering several articles based on how likely is a user of clicking on each of them. More precisely, we will build upon the Ebnerd implementation, which a baseline solution to the challenge mantained by the organisers. The code corresponding to this implementation can be found in [github](https://github.com/ebanalyse/ebnerd-benchmark). In short terms, the model leverages multi-head attention layers to create embeddings for the articles and for the user. Then, the embeddings created for the user and the article analyzed are combined through a dot product to obtain a score for that article. The same operation is repeated for each of the considered articles, which are then ordered based on this score.
<br/>The dataset used is also provided by the organisers in the [challenge website](https://recsys.eb.dk/). More precisely, we will be using an already processed dataset, which already provides the computed embeddings of the articles considered, sparing then the need of computing the embeddings through multi-head attention, which is quite computationally heavy, making the training and inference much faster, as we will probably run it locally. Besides this dataset with the article embeddings, we are also using the dataset with the information of the users, enclosing, among other features, which articles has each user visited.
<br/>The main goal of the project would be to apply most of the knowledge we have acquired during this course to the ebnerd RecSys project, in order to make it more reproducible and reusable, and also facilitate the process of training and running the project. More precisely we would like to implement some tools, Such as docker, to make the code more easy to reproduce, and also some more sophisticated tools for doing a more intuitive hyper-parameter tuning, such as Weight and Biases.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
