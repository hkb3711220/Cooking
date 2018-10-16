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
        work_dict[name] = i
    if output_name == "ingredients":
        work_dict['<EOS>'] = len(work_dict)

    with open('%s.dump'%(output_name), 'wb') as f:
        pickle.dump(work_dict, f)

def load_dict(cuisine = None, ingredient = None):

    cuisine_dict_path = './cuisines.dump'
    ingredient_dict_path = './ingredients.dump'

    cuisine_exist = os.path.exists(cuisine_dict_path)
    ingredient_exist = os.path.exists(ingredient_dict_path)

    if cuisine_exist == False:
        make_dictionary(cuisine, output_name='cuisines')
    if ingredient_exist == False:
        make_dictionary(ingredient, output_name='ingredients')

    cuisine_dict = pickle.load(open('cuisines.dump', 'rb'))
    ingredient_dict = pickle.load(open('ingredients.dump', 'rb'))

    return cuisine_dict, ingredient_dict


def load_train_data(file_name=None):
    open_file = open(file_name, 'r')
    infs = json.load(open_file)

    ingredients = []
    cuisines = []
    for recipe in infs:
        cuisines.append(recipe['cuisine'])
        ingredients.append(recipe['ingredients'])

    cuisine_dict, ingredient_dict = load_dict(cuisine=cuisines, ingredient=ingredients)

    cuisines = [cuisine_dict[cuis] for cuis in cuisines]
    ingredients = [[ingredient_dict[ingred] for ingred in ingred_list] for ingred_list in ingredients]

    return ingredients, cuisines

def load_test_data(file_name=None):
    open_file = open(file_name, 'r')
    infs = json.load(open_file)

    ingredients = []
    for recipe in infs:
        ingredients.append(recipe['ingredients'])

    cuisine_dict, ingredient_dict = load_dict(cuisine = None, ingredient = None)

    for ingred_list in ingredients:
        for ingred in ingred_list:
            if ingred not in ingredient_dict:
                ingredient_dict[ingred] = 0

    ingredients = [[ingredient_dict[ingred] for ingred in ingred_list] for ingred_list in ingredients]

    return ingredients

#The parameters for train and test

train_data, train_label = load_train_data(file_name='train.json')
train_data_leg = [len(data) for data in train_data]
max_data_leg = max(train_data_leg)
cuisine_dict, ingredient_dict = load_dict()
train_data = [data + [ingredient_dict['<EOS>']]*(max_data_leg - len(data)) for data in train_data]
test_data = load_test_data(file_name='test.json')
test_data = [data + [ingredient_dict['<EOS>']]*(max_data_leg - len(data)) for data in test_data]
