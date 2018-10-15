import json
import os
import pickle

os.chdir(os.path.dirname(__file__))

def make_dictionary(dataset, output_name=None):
    work_list = []

    for contains in dataset:
        if type(contains) is list:
            for parts in contains:
                work_list.append(parts)
        else:
            work_list.append(contains)

    work_list = set(work_list)

    work_dict = {}
    for i, name in enumerate(work_list):
        work_dict[name] = i + 1

    with open('%s.dump'%(output_name), 'wb') as f:
        pickle.dump(work_dict, f)

def load_dict(cuisine=None, ingredients=None):

    global cuisine_dict
    global ingredient_dict

    cuisine_dict_path = './cuisines.dump'
    ingredient_dict_path = './ingredients.dump'

    cuisines_exist = os.path.exists(cuisine_dict_path)
    ingredients_exist = os.path.exists(ingredient_dict_path)

    if cuisines_exist==True and ingredients_exist==True:
        cuisine_dict = pickle.load(open('cuisines.dump', 'rb'))
        ingredient_dict = pickle.load(open('ingredients.dump', 'rb'))
    else:
        if cuisines_exist == False:
            make_dictionary(cuisine, output_name='cuisines')
        elif ingredients_exist == False:
            make_dictionary(ingredients, output_name='ingredients')
        else:
            make_dictionary(cuisine, output_name='cuisines')
            make_dictionary(ingredients, output_name='ingredients')

    return cuisine_dict, ingredient_dict

def load_train_data(file_name=None):
    open_file = open(file_name, 'r')
    infs = json.load(open_file)

    ingredients = []
    cuisines = []
    for recipe in infs:
        cuisines.append(recipe['cuisine'])
        ingredients.append(recipe['ingredients'])

    cuisine_dict, ingredient_dict = load_dict()

    cuisine = [cuisine_dict[cuis] for cuis in cuisines]
    ingredients = [[ingredient_dict[ingred] for ingred in ingred_list] for ingred_list in ingredients]

    return ingredients, cuisine

def load_test_data(file_name=None):
    open_file = open(file_name, 'r')
    infs = json.load(open_file)

    ingredients = []
    for recipe in infs:
        ingredients.append(recipe['ingredients'])

    cuisine_dict, ingredient_dict = load_dict()

    for ingred_list in ingredients:
        for ingred in ingred_list:
            if ingred not in ingredient_dict:
                ingredient_dict[ingred] = 0

    ingredients = [[ingredient_dict[ingred] for ingred in ingred_list] for ingred_list in ingredients]

    return ingredients

#The parameters for train
train_data, train_label = load_train_data(file_name='train.json')
train_data_leg = [len(data) for data in train_data]
max_data_leg = max(train_data_leg)
train_data = [data + [0]*(max_data_leg - len(data)) for data in train_data]
test_data = load_test_data(file_name='test.json')
test_data = [data + [0]*(max_data_leg - len(data)) for data in test_data]
cuisine_dict, ingredient_dict = load_dict()
