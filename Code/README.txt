Steps to Run the code


1. Download the Data from the link - https://www.kaggle.com/yelp-dataset/yelp-dataset


2. copy the code files in the same location where the data is placed


3. Create a new empty directory /data in the location where code is present

4. Execute Yelp_DataClean.py to create new data files that are specific to our prefered cities Toronto and Pheonix

5. Execute the code files 
  -- Yelp_User_Based_tfIDF.py
  -- Yelp_User_Based_NMF.py
  -- Yelp_User_Based_doc2vec.py
  -- Yelp_item_Based_NMF.py
  -- Yelp_item_Based_tfIDF.py
  -- Yelp_item_Based_doc2vec.py

inorder to determine the User Based Similarity and Item Based Similarity. Each similarity technique was carried out using 3 feature extraction algorithms - NMF, TF-IDF and Doc2Vec.

6. Execute the ALS_Matrix_Factorization.py to make recommendations or predictions (filtering) about a user’s interests by compiling preferences from a group of users (collaborating).

7. Execute the word2vec_with_LDA.py to find the topic modelling on the user reviews to tag reviews to user defined topics and extract relevant reviews.