import os
import vipy
import numpy as np
import json
import PIL
import io


# Huggingface datasets
vipy.util.try_import('datasets'); from datasets import load_dataset


def mnist():
    dataset = load_dataset("ylecun/mnist", trust_remote_code=True)

    loader = lambda r: vipy.image.ImageCategory(category=str(r['label'])).loader(lambda f: r['image']())
    trainset = vipy.dataset.Dataset([{'label':y, 'image':lambda k=k, ds=dataset['train']: np.uint8(ds[k]['image'])} for (k,y) in enumerate(dataset['train']['label'])], id='mnist', loader=loader, strict=False)
    testset = vipy.dataset.Dataset([{'label':y, 'image':lambda k=k, ds=dataset['test']: np.uint8(ds[k]['image'])} for (k,y) in enumerate(dataset['test']['label'])], id='mnist:test', loader=loader, strict=False)
    return (trainset, testset)
    
def cifar10():
    """Huggingface wrapper for cifar10, returns (train,test) tuple"""
    
    dataset = load_dataset('cifar10', trust_remote_code=True)
    d_idx_to_category = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
    
    loader = lambda r, category=d_idx_to_category: vipy.image.ImageCategory(category=category[r['label']]).loader(lambda f: r['image']())
    trainset = vipy.dataset.Dataset([{'label':y, 'image':lambda k=k, ds=dataset['train']: np.uint8(ds[k]['img'])} for (k,y) in enumerate(dataset['train']['label'])], id='cifar10', loader=loader, strict=False)
    testset = vipy.dataset.Dataset([{'label':y, 'image':lambda k=k, ds=dataset['test']: np.uint8(ds[k]['img'])} for (k,y) in enumerate(dataset['test']['label'])], id='cifar10:test', loader=loader, strict=False)
    return (trainset, testset)


def cifar100():
    """Huggingface wrapper for cifar100 returns (train,test) tuple"""
    
    d_idx_to_fine = {int(k):v for (k,v) in json.loads('{"0": "apple", "1": "aquarium_fish", "2": "baby", "3": "bear", "4": "beaver", "5": "bed", "6": "bee", "7": "beetle", "8": "bicycle", "9": "bottle", "10": "bowl", "11": "boy", "12": "bridge", "13": "bus", "14": "butterfly", "15": "camel", "16": "can", "17": "castle", "18": "caterpillar", "19": "cattle", "20": "chair", "21": "chimpanzee", "22": "clock", "23": "cloud", "24": "cockroach", "25": "couch", "26": "crab", "27": "crocodile", "28": "cup", "29": "dinosaur", "30": "dolphin", "31": "elephant", "32": "flatfish", "33": "forest", "34": "fox", "35": "girl", "36": "hamster", "37": "house", "38": "kangaroo", "39": "keyboard", "40": "lamp", "41": "lawn_mower", "42": "leopard", "43": "lion", "44": "lizard", "45": "lobster", "46": "man", "47": "maple_tree", "48": "motorcycle", "49": "mountain", "50": "mouse", "51": "mushroom", "52": "oak_tree", "53": "orange", "54": "orchid", "55": "otter", "56": "palm_tree", "57": "pear", "58": "pickup_truck", "59": "pine_tree", "60": "plain", "61": "plate", "62": "poppy", "63": "porcupine", "64": "possum", "65": "rabbit", "66": "raccoon", "67": "ray", "68": "road", "69": "rocket", "70": "rose", "71": "sea", "72": "seal", "73": "shark", "74": "shrew", "75": "skunk", "76": "skyscraper", "77": "snail", "78": "snake", "79": "spider", "80": "squirrel", "81": "streetcar", "82": "sunflower", "83": "sweet_pepper", "84": "table", "85": "tank", "86": "telephone", "87": "television", "88": "tiger", "89": "tractor", "90": "train", "91": "trout", "92": "tulip", "93": "turtle", "94": "wardrobe", "95": "whale", "96": "willow_tree", "97": "wolf", "98": "woman", "99": "worm"}').items()}

    d_idx_to_coarse = {int(k):v for (k,v) in json.loads('{"0": "aquatic_mammals", "1": "fish", "2": "flowers", "3": "food_containers", "4": "fruit_and_vegetables", "5": "household_electrical_devices", "6": "household_furniture", "7": "insects", "8": "large_carnivores", "9": "large_man-made_outdoor_things", "10": "large_natural_outdoor_scenes", "11": "large_omnivores_and_herbivores", "12": "medium_mammals", "13": "non-insect_invertebrates", "14": "people", "15": "reptiles", "16": "small_mammals", "17": "trees", "18": "vehicles_1", "19": "vehicles_2"}').items()}
    
    D = load_dataset('cifar100', trust_remote_code=True)
    loader = lambda r, fine=d_idx_to_fine, coarse=d_idx_to_coarse: vipy.image.ImageCategory(array=np.array(r['img']), category=fine[r['fine_label']], attributes={'tags':coarse[r['coarse_label']]})
    return (vipy.dataset.Dataset(D['train'], id='cifar100:train', loader=loader, strict=False), 
            vipy.dataset.Dataset(D['test'], id='cifar100:test', loader=loader, strict=False))
    

def oxford_pets():
    """https://www.robots.ox.ac.uk/~vgg/data/pets/"""
    D = load_dataset("visual-layer/vl-oxford-iiit-pets")
    loader = lambda r: vipy.image.ImageCategory(array=np.array(r['image']), category=str(r['label']), attributes={'dog':bool(r['dog']), 'cat':not bool(r['dog'])})
    return vipy.dataset.Dataset(D['train'], id='oxford_pets', loader=loader, strict=False)

    
def sun397():
    """Sun-397 dataset: https://vision.princeton.edu/projects/2010/SUN/"""

    jsonfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sun397.json')
    d_idx_to_category = {k:v.split('/') for (k,v) in vipy.util.readjson(jsonfile).items()}
    
    configs = ['standard-part1-120k', 'standard-part2-120k', 'standard-part3-120k', 'standard-part4-120k', 'standard-part5-120k', 'standard-part6-120k', 'standard-part7-120k', 'standard-part8-120k', 'standard-part9-120k', 'standard-part10-120k']    
    D = load_dataset("HuggingFaceM4/sun397", 'standard-part1-120k', trust_remote_code=True)
    loader = lambda r, d_idx_to_category=d_idx_to_category: vipy.image.ImageCategory(array=np.array(r['image']), category=d_idx_to_category[str(r['label'])])
    return (vipy.dataset.Dataset(D['train'], id='sun397:train', loader=loader, strict=False), 
            vipy.dataset.Dataset(D['test'], id='sun397:test', loader=loader, strict=False),
            vipy.dataset.Dataset(D['other'], id='sun397_other', loader=loader, strict=False))            

def flickr30k():
    """http://shannon.cs.illinois.edu/DenotationGraph/data/index.html"""
    D = load_dataset("lmms-lab/flickr30k")
    loader = lambda r: vipy.image.ImageCategory(array=np.array(r['image']), category=r['caption'], attributes={'sentid':r['sentids']})
    return vipy.dataset.Dataset(D['test'], id='flickr30k', loader=loader, strict=False)

    
def oxford_fgvc_aircraft():
    """https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"""
    jsonfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'oxford_fgvc_aircraft.json')
    d_idx_to_category = vipy.util.readjson(jsonfile)
    
    D = load_dataset("HuggingFaceM4/FGVC-Aircraft", trust_remote_code=True)
    loader = lambda r, d_idx_to_category=d_idx_to_category: vipy.image.ImageDetection(array=np.array(r['image']),
                                                                                      category=d_idx_to_category['family'][str(r['family'])],
                                                                                      xmin=r['bbox']['xmin'], ymin=r['bbox']['ymin'], xmax=r['bbox']['xmax'], ymax=r['bbox']['ymax'],
                                                                                      attributes={'manufacturer':d_idx_to_category['manufacturer'][str(r['manufacturer'])], 'variant':d_idx_to_category['variant'][str(r['variant'])]})
    return vipy.dataset.Dataset(D['train'], id='oxford_fgvc_aircraft', loader=loader, strict=False)


def pascal_voc_2007():
    """http://host.robots.ox.ac.uk/pascal/VOC/"""
    D = load_dataset("HuggingFaceM4/pascal_voc", 'voc2007_main', trust_remote_code=True)

    # https://huggingface.co/datasets/HuggingFaceM4/pascal_voc/blob/main/pascal_voc.py
    CLASS_INFOS = [('aeroplane', 0, 0, (128, 0, 0)),
                   ('bicycle', 1, 1, (0, 128, 0)),
                   ('bird', 2, 2, (128, 128, 0)),
                   ('boat', 3, 3, (0, 0, 128)),
                   ('bottle', 4, 4, (128, 0, 128)),
                   ('bus', 5, 5, (0, 128, 128)),
                   ('car', 6, 6, (128, 128, 128)),
                   ('cat', 7, 7, (64, 0, 0)),
                   ('chair', 8, 8, (192, 0, 0)),
                   ('cow', 9, 9, (64, 128, 0)),
                   ('diningtable', 10, 10, (192, 128, 0)),
                   ('dog', 11, 11, (64, 0, 128)),
                   ('horse', 12, 12, (192, 0, 128)),
                   ('motorbike', 13, 13, (64, 128, 128)),
                   ('person', 14, 14, (192, 128, 128)),
                   ('pottedplant', 15, 15, (0, 64, 0)),
                   ('sheep', 16, 16, (128, 64, 0)),
                   ('sofa', 17, 17, (0, 192, 0)),
                   ('train', 18, 18, (128, 192, 0)),
                   ('tvmonitor', 19, 19, (0, 64, 128)),
                   ('background', 20, 20, (0, 0, 0)),
                   ('borderingregion', 255, 21, (224, 224, 192))]
    
    d_index_to_class = {v[2]:v[0] for v in CLASS_INFOS}
    loader = lambda r, d_index_to_class=d_index_to_class: vipy.image.Scene(array=np.array(r['image']),
                                        category=sorted(set([d_index_to_class[c] for c in r['classes']])),  # scene category may not include all object categories
                                        objects=[vipy.object.Detection(category=d_index_to_class[c], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax) for (c, (xmin,ymin,xmax,ymax)) in zip(r['objects']['classes'], r['objects']['bboxes'])])
    
    return (vipy.dataset.Dataset(D['train'], id='pascal_voc_2007:train', loader=loader, strict=False),
            vipy.dataset.Dataset(D['validation'], id='pascal_voc_2007:val', loader=loader, strict=False),
            vipy.dataset.Dataset(D['test'], id='pascal_voc_2007:test', loader=loader, strict=False))
    
    
def yfcc100m():
    """https://multimediacommons.wordpress.com/yfcc100m-core-dataset/"""
    dataset = load_dataset("dalle-mini/YFCC100M_OpenAI_subset", trust_remote_code=True)

    loader = lambda r: vipy.image.ImageCategory(array=np.array(PIL.Image.open(io.BytesIO(r['img']))), category=r['description_clean'], attributes={k:v for (k,v) in r.items() if k != 'img'})    
    return (vipy.dataset.Dataset(dataset['train'], id='yfcc100m:train', loader=loader, strict=False),
            vipy.dataset.Dataset(dataset['validation'], id='yfcc100m:val', loader=loader, strict=False))
    

def open_images_v7():
    dataset = load_dataset("dalle-mini/open-images")
    loader = lambda r: vipy.image.Image(url=r['url'])  # no labels (use vipy.data.openimages.open_images_v7 instead)
    return (vipy.dataset.Dataset(dataset['train'], id='open_images_v7:train', loader=loader, strict=False),
            vipy.dataset.Dataset(dataset['validation'], id='open_images_v7:val', loader=loader, strict=False),
            vipy.dataset.Dataset(dataset['test'], id='open_images_v7:test', loader=loader, strict=False))        



def tiny_imagenet():
    D = load_dataset("zh-plus/tiny-imagenet")
    labels = vipy.util.readjson(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tiny_imagenet.json'))    
    d_idx_to_category = {k: sorted([l.strip() for l in labels['wnid_to_category'][wnid].split(',')])  for (k,wnid) in enumerate(labels['idx_to_wnid'])}    
    loader = lambda r, d_idx_to_category=d_idx_to_category: vipy.image.ImageCategory(array=np.array(r['image']), category=d_idx_to_category[int(r['label'])])    
    return (vipy.dataset.Dataset(D['train'], id='tiny_imagenet:train', loader=loader, strict=False),
            vipy.dataset.Dataset(D['valid'], id='tiny_imagenet:val', loader=loader, strict=False))


def the_cauldron():
    """https://huggingface.co/datasets/HuggingFaceM4/the_cauldron"""
    ds = load_dataset("HuggingFaceM4/the_cauldron", "ai2d")

    #>>> ds['train'][0]
    #{'images': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=299x227>],
    #  'texts': [{'user': 'Question: What do respiration and combustion give out\nChoices:\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Heat\nAnswer with the letter.',
    #                'assistant': 'Answer: B',
    #                'source': 'AI2D'}]}
    

def imageinwords():
    """https://huggingface.co/datasets/google/imageinwords"""
    dataset = load_dataset('google/imageinwords', token=None, name="IIW-400", trust_remote_code=True)

def docci():
    """https://huggingface.co/datasets/google/docci"""
    dataset = load_dataset("google/docci")


def coyo300m(threshold=0.2):
    dataset = load_dataset("kakaobrain/coyo-labeled-300m")

    labels = vipy.util.readjson(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'coyo300m.json'))    
    d_idx_to_category = {k:[c.strip() for c in v.split(',')] for (k,v) in labels['class_description'].items()}
    d_idx_to_wnid = {v:k for (k,v) in labels['class_list'].items()}
    loader = lambda r, d_idx_to_wnid=d_idx_to_wnid, d_idx_to_category=d_idx_to_category, threshold=threshold: vipy.image.ImageCategory(url=r['url'], 
                                                                                                                                       category=[d_idx_to_category[str(c)] for (c,p) in zip(r['labels'], r['label_probs']) if float(p)>threshold],
                                                                                                                                       attributes={'wordnet':[d_idx_to_wnid[c] for (c,p) in zip(r['labels'], r['label_probs']) if float(p)>threshold]})
    return vipy.dataset.Dataset(dataset['train'], id='coyo300m', loader=loader, strict=False)
    

def coyo700m():
    dataset = load_dataset("kakaobrain/coyo-700m")

def as100m():
    dataset = load_dataset("OpenGVLab/AS-100M")
