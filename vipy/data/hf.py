import os
import numpy as np
import json
import io

from vipy.downloader import download_and_unpack
from vipy.dataset import Dataset
from vipy.image import ImageCategory, TaggedImage, Scene, Image
from vipy.util import try_import, readjson, tocache, cache, flatlist, filebase
from vipy.object import Detection


# Huggingface datasets
try_import('datasets'); import datasets
from datasets import load_dataset
from datasets import Image as HFImage


def mnist():
    dataset = load_dataset("ylecun/mnist", trust_remote_code=True).cast_column("image", HFImage(decode=False))
    trainset = dataset['train'].add_column("instanceid", [f'mnist:train:{i}' for i in range(len(dataset['train']))])
    testset = dataset['test'].add_column("instanceid", [f'mnist:test:{i}' for i in range(len(dataset['test']))])    
    
    loader = lambda r: ImageCategory(category=str(r['label'])).instanceid(r['instanceid']).loader(Image.bytes_array_loader, r['image']['bytes'])
    return (Dataset(trainset, id='mnist', loader=loader).load(),  # metadata in arrow, slow if we need to access this multiple times, load it instead
            Dataset(testset, id='mnist:test', loader=loader))


def cifar10():
    """Huggingface wrapper for cifar10, returns (train,test) tuple"""    
    dataset = load_dataset('cifar10', trust_remote_code=True).cast_column("img", HFImage(decode=False))
    trainset = dataset['train'].add_column("instanceid", [f'cifar10:train:{i}' for i in range(len(dataset['train']))])
    testset = dataset['test'].add_column("instanceid", [f'cifar10:test:{i}' for i in range(len(dataset['test']))])    
    
    d_idx_to_category = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
    
    loader = lambda r, category=d_idx_to_category: ImageCategory(category=category[r['label']]).loader(Image.bytes_array_loader, r['img']['bytes']).instanceid(r['instanceid'])
    return (Dataset(trainset, id='cifar10', loader=loader).load(),
            Dataset(testset, id='cifar10:test', loader=loader))


def cifar100():
    """Huggingface wrapper for cifar100 returns (train,test) tuple"""
    
    d_idx_to_fine = {int(k):v for (k,v) in json.loads('{"0": "apple", "1": "aquarium_fish", "2": "baby", "3": "bear", "4": "beaver", "5": "bed", "6": "bee", "7": "beetle", "8": "bicycle", "9": "bottle", "10": "bowl", "11": "boy", "12": "bridge", "13": "bus", "14": "butterfly", "15": "camel", "16": "can", "17": "castle", "18": "caterpillar", "19": "cattle", "20": "chair", "21": "chimpanzee", "22": "clock", "23": "cloud", "24": "cockroach", "25": "couch", "26": "crab", "27": "crocodile", "28": "cup", "29": "dinosaur", "30": "dolphin", "31": "elephant", "32": "flatfish", "33": "forest", "34": "fox", "35": "girl", "36": "hamster", "37": "house", "38": "kangaroo", "39": "keyboard", "40": "lamp", "41": "lawn_mower", "42": "leopard", "43": "lion", "44": "lizard", "45": "lobster", "46": "man", "47": "maple_tree", "48": "motorcycle", "49": "mountain", "50": "mouse", "51": "mushroom", "52": "oak_tree", "53": "orange", "54": "orchid", "55": "otter", "56": "palm_tree", "57": "pear", "58": "pickup_truck", "59": "pine_tree", "60": "plain", "61": "plate", "62": "poppy", "63": "porcupine", "64": "possum", "65": "rabbit", "66": "raccoon", "67": "ray", "68": "road", "69": "rocket", "70": "rose", "71": "sea", "72": "seal", "73": "shark", "74": "shrew", "75": "skunk", "76": "skyscraper", "77": "snail", "78": "snake", "79": "spider", "80": "squirrel", "81": "streetcar", "82": "sunflower", "83": "sweet_pepper", "84": "table", "85": "tank", "86": "telephone", "87": "television", "88": "tiger", "89": "tractor", "90": "train", "91": "trout", "92": "tulip", "93": "turtle", "94": "wardrobe", "95": "whale", "96": "willow_tree", "97": "wolf", "98": "woman", "99": "worm"}').items()}

    d_idx_to_coarse = {int(k):v for (k,v) in json.loads('{"0": "aquatic_mammals", "1": "fish", "2": "flowers", "3": "food_containers", "4": "fruit_and_vegetables", "5": "household_electrical_devices", "6": "household_furniture", "7": "insects", "8": "large_carnivores", "9": "large_man-made_outdoor_things", "10": "large_natural_outdoor_scenes", "11": "large_omnivores_and_herbivores", "12": "medium_mammals", "13": "non-insect_invertebrates", "14": "people", "15": "reptiles", "16": "small_mammals", "17": "trees", "18": "vehicles_1", "19": "vehicles_2"}').items()}
    
    dataset = load_dataset('cifar100', trust_remote_code=True).cast_column("img", HFImage(decode=False))
    trainset = dataset['train'].add_column("instanceid", [f'cifar100:train:{i}' for i in range(len(dataset['train']))])
    testset = dataset['test'].add_column("instanceid", [f'cifar100:test:{i}' for i in range(len(dataset['test']))])    
    
    loader = lambda r, fine=d_idx_to_fine, coarse=d_idx_to_coarse: ImageCategory(category=fine[r['fine_label']], attributes={'tags':coarse[r['coarse_label']]}).loader(Image.bytes_array_loader, r['img']['bytes']).instanceid(r['instanceid'])
    return (Dataset(trainset, id='cifar100', loader=loader), 
            Dataset(testset, id='cifar100:test', loader=loader))
    

def oxford_pets():
    """https://www.robots.ox.ac.uk/~vgg/data/pets/"""
    dataset = load_dataset("visual-layer/oxford-iiit-pet-vl-enriched").cast_column("image", HFImage(decode=False))
    trainset = dataset['train'].add_column("instanceid", [f'oxford_pets:train:{i}' for i in range(len(dataset['train']))])
    testset = dataset['test'].add_column("instanceid", [f'oxford_pets:test:{i}' for i in range(len(dataset['test']))])    
    
    loader = lambda r: Scene(category=str(r['label_breed']),
                                        objects=[Detection(xywh=d['bbox'], category=d['label']) for d in r['label_bbox_enriched']] if 'label_bbox_enriched' in r and r['label_bbox_enriched'] is not None else [],
                                        attributes={k:r[k] for k in ('label_cat_dog', 'caption_enriched', 'image_id', 'issues')}).loader(Image.bytes_array_loader, r['image']['bytes']).instanceid(r['instanceid'])
    
    return (Dataset(trainset, id='oxford_pets', loader=loader),
            Dataset(testset, id='oxford_pets:test', loader=loader))

    
def sun397():
    """Sun-397 dataset: https://vision.princeton.edu/projects/2010/SUN/"""

    jsonfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sun397.json')
    d_idx_to_category = {k:v.split('/') for (k,v) in readjson(jsonfile).items()}
    
    configs = ['standard-part1-120k', 'standard-part2-120k', 'standard-part3-120k', 'standard-part4-120k', 'standard-part5-120k', 'standard-part6-120k', 'standard-part7-120k', 'standard-part8-120k', 'standard-part9-120k', 'standard-part10-120k']    
    D = load_dataset("HuggingFaceM4/sun397", 'standard-part1-120k', trust_remote_code=True).cast_column("image", HFImage(decode=False))
    trainset = D['train'].add_column("instanceid", [f'sun397:train:{i}' for i in range(len(D['train']))])
    testset = D['test'].add_column("instanceid", [f'sun397:test:{i}' for i in range(len(D['test']))])    
    otherset = D['other'].add_column("instanceid", [f'sun397:other:{i}' for i in range(len(D['other']))])    
    
    loader = lambda r, d_idx_to_category=d_idx_to_category: TaggedImage(tags=d_idx_to_category[str(r['label'])]).loader(Image.bytes_array_loader, r['image']['bytes']).instanceid(r['instanceid'])
    return (Dataset(trainset, id='sun397:train', loader=loader), 
            Dataset(testset, id='sun397:test', loader=loader),
            Dataset(otherset, id='sun397:other', loader=loader))            

def flickr30k():
    """http://shannon.cs.illinois.edu/DenotationGraph/data/index.html"""
    D = load_dataset("lmms-lab/flickr30k").cast_column("image", HFImage(decode=False))
    testset = D['test'].add_column("instanceid", [f'flickr30k:{i}' for i in range(len(D['test']))])
    loader = lambda r: TaggedImage(caption=r['caption'], attributes={'sentid':r['sentids']}).loader(Image.bytes_array_loader, r["image"]["bytes"]).instanceid(r['instanceid'])
    return Dataset(testset, id='flickr30k', loader=loader)

    
def oxford_fgvc_aircraft():
    """https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"""
    jsonfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'oxford_fgvc_aircraft.json')
    d_idx_to_category = readjson(jsonfile)
    
    D = load_dataset("HuggingFaceM4/FGVC-Aircraft", trust_remote_code=True).cast_column("image", HFImage(decode=False))
    trainset = D['train'].add_column("instanceid", [f'oxford_fgvc_aircraft:{i}' for i in range(len(D['train']))])
    
    loader = lambda r, d_idx_to_category=d_idx_to_category: ImageDetection(category=d_idx_to_category['family'][str(r['family'])],
                                                                                      xmin=r['bbox']['xmin'], ymin=r['bbox']['ymin'], xmax=r['bbox']['xmax'], ymax=r['bbox']['ymax'],
                                                                                      attributes={'manufacturer':d_idx_to_category['manufacturer'][str(r['manufacturer'])],
                                                                                                  'variant':d_idx_to_category['variant'][str(r['variant'])]}).loader(Image.bytes_array_loader, r['image']['bytes']).instanceid(r['instanceid'])
    return Dataset(trainset, id='oxford_fgvc_aircraft', loader=loader)


def pascal_voc_2007():
    """http://host.robots.ox.ac.uk/pascal/VOC/"""
    D = load_dataset("HuggingFaceM4/pascal_voc", 'voc2007_main', trust_remote_code=True).cast_column("image", HFImage(decode=False))
    trainset = D['train'].add_column("instanceid", [f'pascal_voc_2007:train:{i}' for i in range(len(D['train']))])
    testset = D['test'].add_column("instanceid", [f'pascal_voc_2007:test:{i}' for i in range(len(D['test']))])    
    valset = D['validation'].add_column("instanceid", [f'pascal_voc_2007:other:{i}' for i in range(len(D['validation']))])    
    
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
    loader = lambda r, d_index_to_class=d_index_to_class: Scene(tags=[d_index_to_class[c] for c in r['classes']],  # scene category may not include all object categories
                                                                objects=[Detection(category=d_index_to_class[c], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                                                                         for (c, (xmin,ymin,xmax,ymax)) in zip(r['objects']['classes'], r['objects']['bboxes'])]).loader(Image.bytes_array_loader, r['image']['bytes']).instanceid(r['instanceid'])
    
    return (Dataset(trainset, id='pascal_voc_2007:train', loader=loader),
            Dataset(valset, id='pascal_voc_2007:val', loader=loader),
            Dataset(testset, id='pascal_voc_2007:test', loader=loader))
    
    
def yfcc100m():
    """https://multimediacommons.wordpress.com/yfcc100m-core-dataset/
    
    This dataset loads a bytes array containing an image on each row.  Create a separate dataset that ignores the img column and replaces with the backing URL.  
    This is useful for fast analysis of the dataset metadata.
    """

    dataset = load_dataset("dalle-mini/YFCC100M_OpenAI_subset", trust_remote_code=True).select_columns(['title_clean','description_clean','downloadurl','img'])
    trainset = dataset['train'].add_column("instanceid", [f'yfcc100m:train:{i}' for i in range(len(dataset['train']))])
    valset = dataset['validation'].add_column("instanceid", [f'yfcc100m:test:{i}' for i in range(len(dataset['validation']))])    
    
    loader = lambda r: TaggedImage(caption=[r['title_clean'],r['description_clean']], url=r['downloadurl']).loader(Image.bytes_array_loader, r['img']).instanceid(r['instanceid'])
    loader_url = lambda r: TaggedImage(caption=[r['title_clean'],r['description_clean']], url=r['downloadurl']).instanceid(r['instanceid'])
    
    return (Dataset(trainset, id='yfcc100m:train', loader=loader),
            Dataset(trainset.remove_columns(['img']), id='yfcc100m_url:train', loader=loader_url),  # faster loading for metadata analysis
            Dataset(valset, id='yfcc100m:val', loader=loader),
            Dataset(valset.remove_columns(['img']), id='yfcc100m_url:val', loader=loader_url))                            

def tiny_imagenet():
    D = load_dataset("zh-plus/tiny-imagenet").cast_column("image", HFImage(decode=False))
    trainset = D['train'].add_column("instanceid", [f'tiny_imagenet:train:{i}' for i in range(len(D['train']))])
    valset = D['valid'].add_column("instanceid", [f'tiny_imagenet:val:{i}' for i in range(len(D['valid']))])    
    
    labels = readjson(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tiny_imagenet.json'))    
    d_idx_to_category = {k: [l.strip() for l in labels['wnid_to_category'][wnid].split(',')]  for (k,wnid) in enumerate(labels['idx_to_wnid'])}    
    loader = lambda r, d_idx_to_category=d_idx_to_category: TaggedImage(tags=d_idx_to_category[int(r['label'])]).loader(Image.bytes_array_loader, r['image']['bytes']).instanceid(r['instanceid'])
    return (Dataset(trainset, id='tiny_imagenet:train', loader=loader),
            Dataset(valset, id='tiny_imagenet:val', loader=loader))


def coyo300m(threshold=0.2):
    """https://huggingface.co/datasets/kakaobrain/coyo-labeled-300m  (Machine labeled)"""
    dataset = load_dataset("kakaobrain/coyo-labeled-300m").select_columns(['url','labels','label_probs'])
    trainset = dataset['train'].add_column("instanceid", [f'coyo300m:train:{i}' for i in range(len(dataset['train']))])
    
    labels = readjson(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'coyo300m.json'))    
    d_idx_to_category = {k:[c.strip() for c in v.split(',')] for (k,v) in labels['class_description'].items()}
    d_idx_to_wnid = {v:k for (k,v) in labels['class_list'].items()}
    loader = lambda r, d_idx_to_wnid=d_idx_to_wnid, d_idx_to_category=d_idx_to_category, threshold=threshold: TaggedImage(url=r['url'], 
                                                                                                                          tags=flatlist([d_idx_to_category[str(c)] for (c,p) in zip(r['labels'], r['label_probs']) if float(p)>threshold]),
                                                                                                                          attributes={'wordnet':[d_idx_to_wnid[c] for (c,p) in zip(r['labels'], r['label_probs']) if float(p)>threshold]}).instanceid(r['instanceid'])
    return Dataset(trainset, id='coyo300m', loader=loader)

def coyo700m():
    "https://huggingface.co/datasets/kakaobrain/coyo-700m"
    dataset = load_dataset("kakaobrain/coyo-700m").select_columns(['url','text'])
    trainset = dataset['train'].add_column("instanceid", [f'coyo700m:train:{i}' for i in range(len(dataset['train']))])    
    loader = lambda r: TaggedImage(url=r['url'], caption=r['text']).instanceid(r['instanceid'])
    return Dataset(trainset, id='coyo700m', loader=loader)


def laion2b():
    """https://huggingface.co/datasets/laion/relaion2B-en-research-safe"""
    dataset = load_dataset("laion/relaion2B-en-research-safe").select_columns(['url','caption'])
    trainset = dataset['train'].add_column("instanceid", [f'laion2b:train:{i}' for i in range(len(dataset['train']))])
    loader = lambda r: TaggedImage(url=r['url'], caption=r['caption']).instanceid(r['instanceid'])
    return Dataset(trainset, id='laion2b', loader=loader)

def datacomp_1b():
    """https://huggingface.co/datasets/mlfoundations/datacomp_1b"""
    dataset = load_dataset("mlfoundations/datacomp_1b").select_columns(['url','text'])
    trainset = dataset['train'].add_column("instanceid", [f'datacomp_1b:train:{i}' for i in range(len(dataset['train']))])
    loader = lambda r: TaggedImage(url=r['url'], caption=r['text']).instanceid(r['instanceid'])
    return Dataset(trainset, id='datacomp1b', loader=loader)


def imageinwords():
    """https://huggingface.co/datasets/google/imageinwords"""
    dataset = load_dataset('google/imageinwords', token=None, name="IIW-400", trust_remote_code=True)

def docci():
    """https://huggingface.co/datasets/google/docci"""
    dataset = load_dataset("google/docci")

def as100m():
    dataset = load_dataset("OpenGVLab/AS-100M")
        
def objects365():
    raise ValueError('image data no longer available')

    dataset = load_dataset("jxu124/objects365")
    assert os.path.exists(tocache('objects365')), "download image data from https://www.objects365.org to '%s'" % cache() 
    
    loader = lambda r: Scene(filename=os.path.join(tocache('objects365'), r['image_path']),
                                        objects=[Detection(category=d['category'], ulbr=d['bbox']) for d in r['anns_info']])
    
    return (Dataset(dataset['train'], id='objects365:train', loader=loader),
            Dataset(dataset['validation'], id='objects365:val', loader=loader))
            
    
def wakevision():
    dataset = load_dataset("Harvard-Edge/Wake-Vision").cast_column("image", HFImage(decode=False))
    trainset = dataset['train_quality'].add_column("instanceid", [f'wakevision:train:{i}' for i in range(len(dataset['train_quality']))])
    valset = dataset['validation'].add_column("instanceid", [f'wakevision:val:{i}' for i in range(len(dataset['validation']))])
    testset = dataset['test'].add_column("instanceid", [f'wakevision:test:{i}' for i in range(len(dataset['test']))])        
    
    loader = lambda r: ImageCategory(category='person' if r['person']==1 else 'non-person', attributes={k:v for (k,v) in r.items() if k not in ['image']}).loader(Image.bytes_array_loader, r['image']['bytes']).instanceid(r['instanceid'])
    return (Dataset(trainset, id='wakevision:train', loader=loader),
            Dataset(valset, id='wakevision:val', loader=loader),
            Dataset(testset, id='wakevision:test', loader=loader))            


def rasmd():
    """https://huggingface.co/datasets/STL-Yonsei/RASMD/"""
    outdir = os.path.join(datasets.config.HF_DATASETS_CACHE, 'rasmd')
    if not os.path.isdir(os.path.join(outdir, 'RASMD_DATASET')):
        download_and_unpack('https://huggingface.co/datasets/STL-Yonsei/RASMD/resolve/main/RASMD_aligned.zip', outdir)
    D = Dataset.from_directory(outdir, filetype='images')
    swir = {int(filebase(im.filename()).split('_')[-1]):im for im in D.clone().filter(lambda im: 'swir/' in im.filename())}
    rgb = {int(filebase(im.filename()).split('_')[-1]):im for im in D.clone().filter(lambda im: 'rgb/' in im.filename())}    
    return Dataset([(ImageCategory.cast(swir[k]).new_category('swir_original').instanceid(f'rasmd_aligned:{k}'),
                     ImageCategory.cast(rgb[k]).new_category('rgb_aligned').instanceid(f'rasmd_aligned:{k}')) for k in rgb.keys()], id='rasmd_aligned')
