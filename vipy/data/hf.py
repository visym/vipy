import vipy
import numpy as np
import json

vipy.util.try_import('datasets'); from datasets import load_dataset


def cifar10():
    """Huggingface wrapper for cifar10, returns (train,test) tuple"""
    
    D = load_dataset('cifar10', trust_remote_code=True)
    d_idx_to_category = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
    loader = lambda r, category=d_idx_to_category: vipy.image.ImageCategory(array=np.array(r['img']), category=category[r['label']])
    return (vipy.dataset.Dataset(D['train'], id='cifar10_train', loader=loader, strict=False), 
            vipy.dataset.Dataset(D['test'], id='cifar10_test', loader=loader, strict=False))


def cifar100():
    """Huggingface wrapper for cifar100 returns (train,test) tuple"""
    
    d_idx_to_fine = {int(k):v for (k,v) in json.loads('{"0": "apple", "1": "aquarium_fish", "2": "baby", "3": "bear", "4": "beaver", "5": "bed", "6": "bee", "7": "beetle", "8": "bicycle", "9": "bottle", "10": "bowl", "11": "boy", "12": "bridge", "13": "bus", "14": "butterfly", "15": "camel", "16": "can", "17": "castle", "18": "caterpillar", "19": "cattle", "20": "chair", "21": "chimpanzee", "22": "clock", "23": "cloud", "24": "cockroach", "25": "couch", "26": "crab", "27": "crocodile", "28": "cup", "29": "dinosaur", "30": "dolphin", "31": "elephant", "32": "flatfish", "33": "forest", "34": "fox", "35": "girl", "36": "hamster", "37": "house", "38": "kangaroo", "39": "keyboard", "40": "lamp", "41": "lawn_mower", "42": "leopard", "43": "lion", "44": "lizard", "45": "lobster", "46": "man", "47": "maple_tree", "48": "motorcycle", "49": "mountain", "50": "mouse", "51": "mushroom", "52": "oak_tree", "53": "orange", "54": "orchid", "55": "otter", "56": "palm_tree", "57": "pear", "58": "pickup_truck", "59": "pine_tree", "60": "plain", "61": "plate", "62": "poppy", "63": "porcupine", "64": "possum", "65": "rabbit", "66": "raccoon", "67": "ray", "68": "road", "69": "rocket", "70": "rose", "71": "sea", "72": "seal", "73": "shark", "74": "shrew", "75": "skunk", "76": "skyscraper", "77": "snail", "78": "snake", "79": "spider", "80": "squirrel", "81": "streetcar", "82": "sunflower", "83": "sweet_pepper", "84": "table", "85": "tank", "86": "telephone", "87": "television", "88": "tiger", "89": "tractor", "90": "train", "91": "trout", "92": "tulip", "93": "turtle", "94": "wardrobe", "95": "whale", "96": "willow_tree", "97": "wolf", "98": "woman", "99": "worm"}').items()}

    d_idx_to_coarse = {int(k):v for (k,v) in json.loads('{"0": "aquatic_mammals", "1": "fish", "2": "flowers", "3": "food_containers", "4": "fruit_and_vegetables", "5": "household_electrical_devices", "6": "household_furniture", "7": "insects", "8": "large_carnivores", "9": "large_man-made_outdoor_things", "10": "large_natural_outdoor_scenes", "11": "large_omnivores_and_herbivores", "12": "medium_mammals", "13": "non-insect_invertebrates", "14": "people", "15": "reptiles", "16": "small_mammals", "17": "trees", "18": "vehicles_1", "19": "vehicles_2"}').items()}
    
    D = load_dataset('cifar100', trust_remote_code=True)
    loader = lambda r, fine=d_idx_to_fine, coarse=d_idx_to_coarse: vipy.image.ImageCategory(array=np.array(r['img']), category=fine[r['fine_label']], attributes={'tags':coarse[r['coarse_label']]})
    return (vipy.dataset.Dataset(D['train'], id='cifar100_train', loader=loader, strict=False), 
            vipy.dataset.Dataset(D['test'], id='cifar100_test', loader=loader, strict=False))
    

    
def sun397():
    """Sub-397 dataset: https://vision.princeton.edu/projects/2010/SUN/"""

    configs = ['standard-part1-120k', 'standard-part2-120k', 'standard-part3-120k', 'standard-part4-120k', 'standard-part5-120k', 'standard-part6-120k', 'standard-part7-120k', 'standard-part8-120k', 'standard-part9-120k', 'standard-part10-120k']
    
    D = load_dataset("HuggingFaceM4/sun397", 'standard-part1-120k', trust_remote_code=True)
    loader = lambda r: vipy.image.ImageCategory(array=np.array(r['img']), category=r['label'])
    return (vipy.dataset.Dataset(D['train'], id='sun397_train', loader=loader, strict=False), 
            vipy.dataset.Dataset(D['test'], id='sun397_test', loader=loader, strict=False))
    
