# NLP: spam classification



## DATASET

The dataset comes from a Kaggle competition: https://www.kaggle.com/uciml/sms-spam-collection-dataset

It contains 5,574 SMS labeled with the ham (legitimate) category or with the spam category. 

## Results 

The notebook Spam_Detection_2 is the result. It uses ressources from the data folder, to get the Spams and also to get the list of 30k Spam Words.<br/>
It also uses scripts from the cnn_text_classification_tf_master folder. Those scripts are used to implement the CNN based classification.<br/>

With those files at hand, you can rerun the notebook if you'd like. The preprocessing part runs quite quickly, the models training can be a little longer depending on the model.<br/>
The way the notebook is displyed here (on GitLab), you can see the results of the models (since I showed those results as dataframe), but the data analysis part at the beginning can be quickly re-run: to get the proper graphs.<br/>
The very first cell of the notebook is a summary, very useful to go from one place of the notebook to another, since it is a little bit long.
