import re

# Read the log file
with open('/home/huan1932/TARFA/log/evaluate_oxford_pets_tinyvit.log', 'r') as f:
    content = f.read()

# Define patterns to extract data
# Pattern to find evaluation blocks
eval_pattern = r'Fine Model: ([\w\.\_]+)\nAttack Method: (Trixfer_\w+).*?Victim Dataset: (\w+).*?Adaptation Method: (\w+).*?Successful attacks \(ASR\): ([\d\.]+)%'
conditional_pattern = r'Evaluating: eval_.*?conditional_results\.pt.*?Fine Model: ([\w\.\_]+)\nAttack Method: (Trixfer_\w+).*?Victim Dataset: (\w+).*?Adaptation Method: (\w+).*?Successful attacks \(ASR\): ([\d\.]+)%'

# Find all matches
all_matches = re.findall(eval_pattern, content, re.DOTALL)
conditional_matches = re.findall(conditional_pattern, content, re.DOTALL)

# Organize data by surrogate model and attack method
data = {}

for match in all_matches:
    surrogate, attack, dataset, adaptation, asr = match
    key = (surrogate, attack, adaptation)
    
    if key not in data:
        data[key] = {'imagenet1k': None, 'cifar10': None, 'cifar10_conditional': None}
    
    if dataset == 'imagenet1k':
        data[key]['imagenet1k'] = float(asr)
    elif dataset == 'cifar10':
        data[key]['cifar10'] = float(asr)

# Add conditional results
for match in conditional_matches:
    surrogate, attack, dataset, adaptation, asr = match
    key = (surrogate, attack, adaptation)
    
    if key not in data:
        data[key] = {'imagenet1k': None, 'cifar10': None, 'cifar10_conditional': None}
    
    if dataset == 'cifar10':
        data[key]['cifar10_conditional'] = float(asr)

# Print organized results
surrogate_models = ['mobilenetv2_100.ra_in1k', 'convnextv2_tiny.fcmae_ft_in1k', 
                   'efficientvit_b2.r224_in1k', 'inception_next_small.sail_in1k']
attack_methods = ['Trixfer_FGSM', 'Trixfer_IFGSM', 'Trixfer_MIFGSM', 'Trixfer_DIFGSM', 'Trixfer_TIFGSM']
adaptation = 'linear_probing'

print("TinyViT - Linear Probing (LastLayerFT) Results:")
print("="*100)
print(f"{'Surrogate':<30} {'Attack':<20} {'ASR^P (ImageNet)':<20} {'ASR^T (CIFAR-10)':<20} {'Conditional ASR^T':<20}")
print("="*100)

for surrogate in surrogate_models:
    for attack in attack_methods:
        key = (surrogate, attack, adaptation)
        if key in data:
            asr_p = data[key]['imagenet1k'] if data[key]['imagenet1k'] else 'N/A'
            asr_t = data[key]['cifar10'] if data[key]['cifar10'] else 'N/A'
            asr_t_cond = data[key]['cifar10_conditional'] if data[key]['cifar10_conditional'] else 'N/A'
            print(f"{surrogate:<30} {attack:<20} {str(asr_p):<20} {str(asr_t):<20} {str(asr_t_cond):<20}")
        else:
            print(f"{surrogate:<30} {attack:<20} {'N/A':<20} {'N/A':<20} {'N/A':<20}")

print("\n" + "="*100)
print("\nData ready for table format:")
print("="*100)

# Format for the table
for attack in attack_methods:
    print(f"\n{attack}:")
    for surrogate in surrogate_models:
        key = (surrogate, attack, adaptation)
        if key in data:
            asr_p = f"{data[key]['imagenet1k']:.2f}" if data[key]['imagenet1k'] else 'N/A'
            asr_t = f"{data[key]['cifar10']:.2f}" if data[key]['cifar10'] else 'N/A'
            asr_t_cond = f"{data[key]['cifar10_conditional']:.2f}" if data[key]['cifar10_conditional'] else 'N/A'
            print(f"  {surrogate}: ASR^P={asr_p}%, ASR^T={asr_t}%, Conditional ASR^T={asr_t_cond}%")