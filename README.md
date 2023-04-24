# Introduction 
This project is an attempt to create an effective recommendation system. A python based package suitable for cloud deployment is realised.
The python package will be cloud agnostic. A DevOps solution to make the code operate as a databricks workflow job is outlined. 
Recommendations will be extracted based on three methods and a consolidated recommendation will be provided to the user

1. popularity-based model, 
2. the content-based model, and 
3. the ML model-based model. 
## Getting Started

1.	Installation Process : 
    * cd into the root directory where setup.py is located 
    * use python3 -m pip install .
2.	Software dependencies
    * Install any additional dependencies associated with requirements_test.txt for local unit test
3.	Latest releases
    * describe the latest releases. For example this could include " Devops improvements made "
4.	API references (add API references if any)

## Build and Test
 
1.	Build : 
    * cd into the root directory where setup.py is located 
    * use python setup.py bdist_wheel to create the dist/*.egg file
2.	Test:
    *  pip install -r requirements_test.txt
    * Use pytest . -vv to run the unit tests. 

# 📦hybrid_recommender\
 ┣ 📂AnalyticsCore\
 ┣ 📂ContentBasedRecommender\
 ┣ 📂ModelBasedRecommender\
 ┣ 📂PopularityBasedRecommender\
 ┣ 📂WeightedHybridRecommender\
 ┗ 📜__init__.py\