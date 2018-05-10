import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data and format it
# only collect movies with atleast a 4.0 rating
data = fetch_movielens(min_rating = 4.0)
# create an interaction matrix from csv file and store
# it in data variable as a dictionary
# splits dataset into training and testing data


# Retrieve training and testing data by using the keys
# Print them both
print(repr(data['train']))
print(repr(data['test']))

# Store model in variable called model
# loss means loss function and measure difference between
# models prediction and the desired output, we want to minimize it during
# training so our model gets more accurate over time
# warp = Weighted approximated-rank pairwise
# it helps create recomendations by
# using gradient descent algorithm to iteratively find the weights
# and improve our prediction over time
model = LightFM(loss = 'warp')

# Train model
# The fit model takes 3 parameters, the data set we want to train it on,
# the number of epochs we want to run the training for,
# and the number of threads we want to run this on
model.fit(data['train'], epochs = 30, num_threads = 2)

def sample_recommendation(model, data, user_ids):
    """ takes 3 parameters, model, data, and list of user ids
    that we want to generate recommendations for"""
    # number of users and movies in training data

    n_users, n_items = data['train'].shape

    #generate recommendations for each user we input
    for user_id in user_ids:

        #movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # rank them in order of most liked to least
        # argsort will sort them in order of score and will return
        # in descending order because of negative sign
        top_itmes = data['item_labels'][np.argsort(-scores)]

        #print out the results
        #print user ID
        print("User %s" % user_id)

        print("     Known positives:")
        # print top 3 known positive movies that the user has picked
        for x in known_positives[:3]:
            print("           %s" % x)

        print("      Recommended:")

        # print out top 3 recommended movies that the model predicts
        for x in top_itmes[:3]:
            print("           %s" % x)


sample_recommendation(model, data, [3, 24, 450])









