# MathMod Project

This repo is the project in the course MathMod (MA1487) HT2021 where we learn about mathematics for statistic analysis.
The course is a part of the program Im attending at Blekinge Institute of Technology (BTH).

Ive used pyenv to generate a virtual environment for this project. To install and use pyenv, follow the instructions
on the [pyenv github page](https://github.com/pyenv/pyenv). If you do not feel like installing pyenv, you can create any
python v3.9.* virtual environment the normal way and install the requirements in the requirements.txt file. This way
should be equally as good to run det app.

If you may need virtualenv:  
`pip install virtualenv`

Create your environment:  
`python -m venv <virtual-environment-name>`

Activate your environment:  
`source <virtual-environment-name>/bin/activate`

Install the project requirements:  
`pip install -r requirements.txt`

To run the app to generate the plots and tables:  
`python mathmod_project.py`

The project is calculating several key statistics for the report written from a few given datasets. The datasets are csv
files and can be found in the csv directory. The plots are generated in the plots directory. The data tables are
generated in the tables.

Enjoy!
