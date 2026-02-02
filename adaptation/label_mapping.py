"""
Label mapping module for cross-dataset adversarial example evaluation.

This module provides mappings between source dataset labels and victim model dataset labels,
with semantic similarity checking to properly evaluate attack success.
"""

import torch
from typing import Dict, List, Set, Optional, Tuple
import torchvision.datasets as datasets
from functools import lru_cache

# ============================================================================
# Source Dataset Class Names
# ============================================================================

# CINIC-10: Same classes as CIFAR-10
CINIC10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# STL-10: 10 classes
STL10_CLASSES = [
    'airplane', 'bird', 'car', 'cat', 'deer',
    'dog', 'horse', 'monkey', 'ship', 'truck'
]

# Combined dataset structure:
# - Stanford Cars: 0-195 (196 classes)
# - ImageWoof: 196-205 (10 classes) - dog breeds
# - FGVC-Aircraft: 206-305 (100 classes)

# ImageWoof classes (ImageNet synsets)
IMAGEWOOF_CLASSES = [
    "Shih-Tzu",  # Shih-Tzu
    "Rhodesian_ridgeback",  # Rhodesian ridgeback
    "beagle",  # beagle
    'English_foxhound',  # English foxhound
    'Border_terrier',  # Border terrier
    'Australian_terrier',  # Australian terrier
    'golden_retriever',  # Golden retriever
    'Old_English_sheepdog',  # Old English sheepdog
    'Samoyed',  # Samoyed
    'dingo',  # Dingo
]

# ============================================================================
# Victim Model Dataset Class Names
# ============================================================================

# CIFAR-10 classes
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CIFAR-100 fine classes (100 classes)
CIFAR100_FINE_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
    'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
    'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
    'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# ============================================================================
# Semantic Similarity Groups
# ============================================================================

# Groups of semantically similar classes that should NOT count as successful attacks
# if the victim model predicts one from the same group

DOG_SEMANTIC_GROUP = {
        # ImageNet dog breeds (from labels 151-273)
        # Small dogs
        'chihuahua', 'japanese_spaniel', 'maltese_dog', 'pekinese', 'shih-tzu',
        'blenheim_spaniel', 'papillon', 'toy_terrier', 'dog',
        # Hounds
        'rhodesian_ridgeback', 'afghan_hound', 'basset', 'beagle', 'bloodhound',
        'bluetick', 'black-and-tan_coonhound', 'walker_hound', 'english_foxhound',
        'redbone', 'borzoi', 'irish_wolfhound', 'italian_greyhound', 'whippet',
        'ibizan_hound', 'norwegian_elkhound', 'otterhound', 'saluki',
        'scottish_deerhound', 'weimaraner',
        # Terriers
        'staffordshire_bullterrier', 'american_staffordshire_terrier',
        'bedlington_terrier', 'border_terrier', 'kerry_blue_terrier',
        'irish_terrier', 'norfolk_terrier', 'norwich_terrier', 'yorkshire_terrier',
        'wire-haired_fox_terrier', 'lakeland_terrier', 'sealyham_terrier',
        'airedale', 'cairn', 'australian_terrier', 'dandie_dinmont',
        'boston_bull', 'miniature_schnauzer', 'giant_schnauzer',
        'standard_schnauzer', 'scotch_terrier', 'tibetan_terrier',
        'silky_terrier', 'soft-coated_wheaten_terrier',
        'west_highland_white_terrier', 'lhasa',
        # Retrievers
        'flat-coated_retriever', 'curly-coated_retriever', 'golden_retriever',
        'labrador_retriever', 'chesapeake_bay_retriever',
        # Pointers and setters
        'german_short-haired_pointer', 'vizsla', 'english_setter',
        'irish_setter', 'gordon_setter',
        # Spaniels
        'brittany_spaniel', 'clumber', 'english_springer',
        'welsh_springer_spaniel', 'cocker_spaniel', 'sussex_spaniel',
        'irish_water_spaniel',
        # Working dogs
        'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard',
        'kelpie', 'komondor', 'old_english_sheepdog', 'shetland_sheepdog',
        'collie', 'border_collie', 'bouvier_des_flandres', 'rottweiler',
        'german_shepherd', 'doberman', 'miniature_pinscher',
        'greater_swiss_mountain_dog', 'bernese_mountain_dog', 'appenzeller',
        'entlebucher', 'boxer', 'bull_mastiff', 'tibetan_mastiff',
        'french_bulldog', 'great_dane', 'saint_bernard', 'eskimo_dog',
        'malamute', 'siberian_husky', 'dalmatian', 'affenpinscher',
        'basenji', 'pug', 'leonberg', 'newfoundland', 'great_pyrenees',
        'samoyed', 'pomeranian', 'chow', 'keeshond', 'brabancon_griffon',
        'pembroke', 'cardigan', 'toy_poodle', 'miniature_poodle',
        'standard_poodle', 'mexican_hairless',
        # Oxford Pet dog breeds (normalized)
        'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle',
        'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter',
        'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
        'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland',
        'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier',
        'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier'
    }


# Updated CAT_SEMANTIC_GROUP for ImageNet
CAT_SEMANTIC_GROUP = {

    # ImageNet cat breeds (from labels 281-285)
    'tabby', 'tiger_cat', 'persian_cat', 'siamese_cat', 'egyptian_cat',
    # Big cats (also semantically similar)
    'cougar', 'lynx', 'leopard', 'snow_leopard', 'jaguar', 'lion', 'tiger', 'cheetah',
    # Oxford Pet cat breeds (normalized)
    'abyssinian', 'bengal', 'birman', 'bombay', 'british_shorthair',
    'egyptian_mau', 'maine_coon', 'persian', 'ragdoll', 'russian_blue',
    'siamese', 'sphynx', 'cat'
    
}

# Updated AIRCRAFT_SEMANTIC_GROUP for ImageNet
AIRCRAFT_SEMANTIC_GROUP = {


    # Aircraft types from ImageNet
    'aircraft_carrier',  # label 403
    'airliner',          # label 404
    'airship',           # label 405
    'warplane',          # label 895
    'amphibian',         # label 408 (amphibious aircraft)
    'aircraft'
}


# Updated CAR_SEMANTIC_GROUP for ImageNet
CAR_SEMANTIC_GROUP = {
        # Vehicle types from ImageNet (all should be semantically similar)
        'convertible',           # label 511
        'limousine',             # label 627
        'minivan',               # label 656
        'model_t',               # label 661
        'racer',                 # label 751
        'sports_car',            # label 817
        'ambulance',             # label 407
        'beach_wagon',           # label 436
        'cab',                   # label 468
        'fire_engine',           # label 555
        'garbage_truck',         # label 569
        'go-kart',               # label 573
        'golfcart',              # label 575
        'motor_scooter',         # label 670
        'mountain_bike',         # label 671
        'pickup',                # label 717
        'police_van',            # label 734
        'recreational_vehicle',  # label 757
        'school_bus',            # label 779
        'snowplow',              # label 803
        'streetcar',             # label 829
        'tank',                  # label 847
        'tractor',               # label 866
        'trailer_truck',         # label 867
        'car',
        'automobile',
        'bus',
        'pickup_truck'
    
}


# ============================================================================
# Helper Functions
# ============================================================================
def get_oxford_pet_classes(root=None):
    """Get Oxford-IIIT-Pet class names dynamically."""
    imagenet_style_names = [
        "Abyssinian",
        "American_bulldog",
        "American_Staffordshire_terrier",   # ImageNet 没有 "Pit Bull"，用最近等价类
        "basset",
        "beagle",
        "Bengal_cat",
        "Birman",
        "Bombay_cat",
        "boxer",
        "British_shorthair",
        "Chihuahua",
        "Egyptian_cat",                     # ImageNet uses "Egyptian_cat"
        "cocker_spaniel",
        "English_setter",
        "German_short-haired_pointer",
        "Great_Pyrenees",
        "Havanese",
        "Japanese_spaniel",                 # ImageNet 用的是 Japanese_spaniel
        "keeshond",
        "Leonberg",
        "Maine_coon",
        "miniature_pinscher",
        "Newfoundland",
        "Persian_cat",
        "Pomeranian",
        "pug",
        "Ragdoll",
        "Russian_Blue",
        "Saint_Bernard",
        "Samoyed",
        "Scotch_terrier",                   # ImageNet: Scotch_terrier
        "Shiba_Inu",
        "Siamese_cat",
        "Sphynx",
        "Staffordshire_bullterrier",
        "soft-coated_wheaten_terrier",
        "Yorkshire_terrier",
    ]
    return imagenet_style_names



def get_source_dataset_classes(dataset_name: str, root: str = '/home/newdrive/huan1932/data') -> List[str]:
    """Get class names for source dataset."""
    if dataset_name == 'cinic10' or dataset_name == 'cinic-10':
        return CINIC10_CLASSES
    elif dataset_name == 'stl10':
        return STL10_CLASSES
    elif dataset_name == 'oxford-iiit-pet':
        return get_oxford_pet_classes(root)
    elif dataset_name == 'combined':
        return None  # Handled by label index ranges
    else:
        raise ValueError(f"Unknown source dataset: {dataset_name}")

def get_victim_dataset_classes(victim_dataset: str) -> List[str]:
    """Get class names for victim model dataset."""
    if victim_dataset == 'cifar10':
        return CIFAR10_CLASSES
    elif victim_dataset == 'cifar100':
        return CIFAR100_FINE_CLASSES
    elif victim_dataset in ['imagenet', 'imagenet1k']:
        # Will be loaded dynamically
        return None
    else:
        raise ValueError(f"Unknown victim dataset: {victim_dataset}")

def normalize_class_name(class_name: str) -> str:
    """Normalize class name for comparison (lowercase, replace spaces/underscores/hyphens)."""
    # Convert to lowercase
    normalized = class_name.lower()
    # Replace spaces, hyphens, and underscores with single underscore
    normalized = normalized.replace(' ', '_').replace('-', '_')
    # Remove multiple consecutive underscores
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    return normalized

def are_semantically_similar(
    class1: str,
    class2: str,
    victim_dataset: str,
    source_dataset: str = None,
    source_label: int = None
) -> bool:
    """
    Check if two classes are semantically similar (should not count as attack success).
    
    Handles combined dataset by using label ranges:
    - Combined labels 0-195: Cars (Stanford Cars)
    - Combined labels 196-205: Dogs (ImageWoof)
    - Combined labels 206-305: Aircraft (FGVC-Aircraft)
    
    Args:
        class1: First class name (from source dataset) or generic category for combined dataset
        class2: Second class name (from victim dataset prediction)
        victim_dataset: Victim model dataset
        source_dataset: Source dataset (optional, needed for combined dataset)
        source_label: Source label index (optional, needed for combined dataset to determine category)
    
    Returns:
        bool: True if semantically similar
    """
    class1_norm = normalize_class_name(class1)
    class2_norm = normalize_class_name(class2)
    
    # Handle combined dataset - determine semantic category from label range
    if source_dataset == 'combined' and source_label is not None:
        if source_label < 196:
            # Stanford Cars (0-195) -> car category
            class1_category = 'car'
        elif source_label < 206:
            # ImageWoof (196-205) -> dog category
            class1_category = 'dog'
        else:
            # FGVC-Aircraft (206-305) -> aircraft category
            class1_category = 'aircraft'
        
        # Check if class2 (victim prediction) is in the same semantic group
        if class1_category == 'car':
            return class2_norm in CAR_SEMANTIC_GROUP
        elif class1_category == 'dog':
            return class2_norm in DOG_SEMANTIC_GROUP
        elif class1_category == 'aircraft':
            return class2_norm in AIRCRAFT_SEMANTIC_GROUP
        else:
            return False
    
    # For other datasets, check if both are in the same universal semantic group
    for group in [DOG_SEMANTIC_GROUP, CAT_SEMANTIC_GROUP, AIRCRAFT_SEMANTIC_GROUP, CAR_SEMANTIC_GROUP]:
        if class1_norm in group and class2_norm in group:
            return True
    
    return False


# ============================================================================
# Label Mapping Functions
# ============================================================================

def map_source_to_victim_label(
    source_label: int,
    source_dataset: str,
    victim_dataset: str,
    source_classes: Optional[List[str]] = None,
    victim_classes: Optional[List[str]] = None
) -> Tuple[Optional[int], bool, Optional[str]]:
    """
    Map a source dataset label to a victim dataset label.
    
    Returns:
        (victim_label, label_exists, source_class_name): 
        - victim_label: The corresponding label in victim dataset, or None if not found
        - label_exists: True if exact match exists, False if only semantic similarity
        - source_class_name: The source class name for semantic comparison
    """
    # Get class names if not provided
    if source_classes is None:
        source_classes = get_source_dataset_classes(source_dataset)
    if victim_classes is None:
        victim_classes = get_victim_dataset_classes(victim_dataset)
    
    # Handle combined dataset specially
    if source_dataset == 'combined':
        return map_combined_to_victim_label(source_label, victim_dataset, victim_classes)
    
    # Get source class name
    if source_classes is None or source_label >= len(source_classes):
        return None, False, None
    source_class_name = source_classes[source_label]
    
    # Try to find exact match in victim dataset
    if victim_classes is not None:
        source_norm = normalize_class_name(source_class_name)
        for i, victim_class in enumerate(victim_classes):
            victim_norm = normalize_class_name(victim_class)
            # print(f"victim_norm: {victim_norm}, source_norm: {source_norm}")
            if victim_norm == source_norm:
                return i, True, source_class_name
    
    # No exact match found
    return None, False, source_class_name
@lru_cache(maxsize=1)
def get_stanford_cars_classes(root='/home/newdrive/huan1932/data'):
    """Get Stanford Cars class names dynamically."""
    try:
        dataset = datasets.StanfordCars(
            root=root, split='train', download=False,
            transform=None
        )
        return dataset.classes
    except Exception as e:
        print(f"Warning: Could not load Stanford Cars classes: {e}")
        return None
@lru_cache(maxsize=1)
def get_fgvc_aircraft_classes(root='/home/newdrive/huan1932/data'):
    """Get FGVC-Aircraft class names dynamically."""
    try:
        dataset = datasets.FGVCAircraft(
            root=root, split='trainval', download=False,
            transform=None
        )
        return dataset.classes
    except Exception as e:
        print(f"Warning: Could not load FGVC-Aircraft classes: {e}")
        return None

def get_combined_dataset_class_name(source_label: int, root: str = '/home/newdrive/huan1932/data') -> Optional[str]:
    """
    Get the actual class name for a combined dataset label.
    
    Args:
        source_label: Label index in combined dataset (0-305)
        root: Root directory for datasets
    
    Returns:
        Class name string, or None if not found
    """
    if source_label < 196:
        # Stanford Cars (0-195)
        classes = get_stanford_cars_classes(root)
        if classes is not None and source_label < len(classes):
            return classes[source_label]
        return None
    elif source_label < 206:
        # ImageWoof (196-205)
        imagewoof_idx = source_label - 196
        if imagewoof_idx < len(IMAGEWOOF_CLASSES):
            return IMAGEWOOF_CLASSES[imagewoof_idx]
        return None
    else:
        # FGVC-Aircraft (206-305)
        fgvc_idx = source_label - 206
        classes = get_fgvc_aircraft_classes(root)
        if classes is not None and fgvc_idx < len(classes):
            return classes[fgvc_idx]
        return None

def map_combined_to_victim_label(
    source_label: int,
    victim_dataset: str,
    victim_classes: Optional[List[str]] = None,
    root: str = '/home/newdrive/huan1932/data'
) -> Tuple[Optional[int], bool, str]:
  
    # Get the ACTUAL class name for this source label
    actual_class_name = get_combined_dataset_class_name(source_label, root)
    
    # Get victim classes if not provided
    if victim_classes is None:
        victim_classes = get_victim_dataset_classes(victim_dataset)
    
    
    if source_label < 196:
        # Stanford Cars -> generic 'car' category
        generic_category = 'car'
        if victim_dataset == 'cifar10':
            if victim_classes is None:
                victim_classes = CIFAR10_CLASSES
            try:
                idx = victim_classes.index('automobile')
                return idx, False, generic_category  # Semantic similarity, not exact
            except ValueError:
                return None, False, generic_category
        elif victim_dataset == 'cifar100':
            # Check for any car-related class in CIFAR-100
            if victim_classes is None:
                victim_classes = CIFAR100_FINE_CLASSES
            # CIFAR-100 has 'bus' which is semantically similar to car
            try:
                idx = victim_classes.index('bus')
                return idx, False, generic_category  # Semantic similarity
            except ValueError:
                return None, False, generic_category
        elif victim_dataset in ['imagenet', 'imagenet1k']:
            return None, False, generic_category
    
    elif source_label < 206:
        # ImageWoof (dog breeds) -> generic 'dog' category
        # FIRST: Try to find EXACT match with actual class name
        if actual_class_name is not None and victim_classes is not None:
            source_norm = normalize_class_name(actual_class_name)
            # Check for exact match in victim dataset
            for i, victim_class in enumerate(victim_classes):
                victim_norm = normalize_class_name(victim_class)
                if victim_norm == source_norm:
                    # EXACT MATCH FOUND! Return with label_exists=True
                    return i, True, actual_class_name
        generic_category = 'dog'
        if victim_dataset == 'cifar10':
            if victim_classes is None:
                victim_classes = CIFAR10_CLASSES
            try:
                idx = victim_classes.index('dog')
                return idx, False, generic_category  # Semantic similarity
            except ValueError:
                return None, False, generic_category
        elif victim_dataset == 'cifar100':
            # CIFAR-100 has 'beagle' - but we already checked for exact match above
            # If we're here, it means the specific ImageWoof breed doesn't match 'beagle' exactly
            # So we use generic 'dog' category
            if victim_classes is None:
                victim_classes = CIFAR100_FINE_CLASSES
            try:
                idx = victim_classes.index('beagle')
                return idx, False, generic_category  # Semantic similarity (generic dog -> beagle)
            except ValueError:
                return None, False, generic_category
        elif victim_dataset in ['imagenet', 'imagenet1k']:
            # ImageWoof classes ARE ImageNet synsets, so exact matches should have been found above
            # If we're here, it means no exact match (unlikely but possible)
            return None, False, generic_category
    
    else:
        # FGVC-Aircraft -> generic 'aircraft' category
        generic_category = 'aircraft'
        if victim_dataset == 'cifar10':
            if victim_classes is None:
                victim_classes = CIFAR10_CLASSES
            try:
                idx = victim_classes.index('airplane')
                return idx, False, generic_category  # Semantic similarity
            except ValueError:
                return None, False, generic_category
        elif victim_dataset == 'cifar100':
            return None, False, generic_category
        elif victim_dataset in ['imagenet', 'imagenet1k']:
            # Some aircraft models might match ImageNet classes exactly (checked above)
            # If we're here, use generic category
            return None, False, generic_category
    
    return None, False, 'unknown'