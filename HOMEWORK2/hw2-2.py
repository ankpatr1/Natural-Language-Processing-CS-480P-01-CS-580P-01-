
#Name : Ankita Patra
#B-number : B01101280

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch

print("="*60)
print("MAC-OPTIMIZED VERSION")
print("="*60)

# Detect device
if torch.cuda.is_available():
    device = "cuda:0"
    print("✓ Using: CUDA GPU")
elif torch.backends.mps.is_available():
    device = "mps"
    print("✓ Using: Apple Silicon (MPS)")
else:
    device = "cpu"
    print("✓ Using: CPU (this will be slow)")

print(f"Device: {device}")

# here I Choosed model size based on your Mac 
# step 1: FLAN-T5-XL (original, but SLOW on Mac)  
# Step: FLAN-T5-BASE (recommended for Mac, much faster)
# Set to False to use BASE (faster for testing)
USE_LARGE_MODEL = True 
if USE_LARGE_MODEL:
    model_name = 'google/flan-t5-xl'
    print("✓ Using FLAN-T5-XL as required by assignment")
    print("⚠️  WARNING: This will be SLOW on Mac (expect 2-4 hours total)")
else:
    model_name = 'google/flan-t5-base'
    print("✓ Using FLAN-T5-BASE (faster for testing)")
    print("  Assignment requires XL - set USE_LARGE_MODEL = True")

# Load dataset - FIXED to use SST5 instead of RTE
sst5_dataset = load_dataset('SetFit/sst5', split='validation')
sst5_train = load_dataset('SetFit/sst5', split='train')

# Load model with proper device mapping for Mac
print(f"\nLoading {model_name}...")
if device == "cpu":
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
elif device == "mps":
    # MPS (Apple Silicon) - direct loading
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
else:
    # CUDA (unlikely on Mac)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='cuda:0')

model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"✓ Model loaded successfully")
print(f"✓ Validation set size: {len(sst5_dataset)}")
print(f"✓ Training set size: {len(sst5_train)}")

# SST5 has 5 sentiment classes
label_names = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive"
}

# USE flan-t5-xl to COMPLETE the Evaluation on SST5

def predict_sentiment(text, model, tokenizer, device):
    # Helper function to predict sentiment
    prompt = f"What is the sentiment of this sentence? Answer with: very negative, negative, neutral, positive, or very positive.\n\nSentence: {text}\nSentiment:"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    return prediction

def map_prediction_to_label(prediction):
    # Map text prediction to label (0-4)
    prediction = prediction.lower()
    if "very negative" in prediction:
        return 0
    elif "very positive" in prediction:
        return 4
    elif "negative" in prediction:
        return 1
    elif "positive" in prediction:
        return 3
    elif "neutral" in prediction:
        return 2
    else:
        return 2  # default to neutral

def evaluate_zero_shot(dataset, model, tokenizer, device, max_samples=None):
    # Evaluate without any examples (zero-shot)
    correct = 0
    
    # Allow limiting samples for faster testing on Mac
    eval_dataset = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    total = len(eval_dataset)
    
    print("\n" + "="*60)
    print("TODO 1: ZERO-SHOT EVALUATION")
    print("="*60)
    print(f"Evaluating on {total} samples")
    
    for example in tqdm(eval_dataset, desc="Evaluating"):
        text = example['text']
        true_label = example['label']
        
        prediction_text = predict_sentiment(text, model, tokenizer, device)
        pred_label = map_prediction_to_label(prediction_text)
        
        if pred_label == true_label:
            correct += 1
    
    accuracy = correct / total
    return accuracy

# Run zero-shot evaluation
# For quick testing on Mac, you can limit samples with max_samples=100
zero_shot_accuracy = evaluate_zero_shot(sst5_dataset, model, tokenizer, device)
print(f"\n✓ Zero-shot Accuracy: {zero_shot_accuracy:.4f} ({zero_shot_accuracy*100:.2f}%)")

# Are you able to use SST5 training datset to get a better performance via In-Context Learning 

def create_few_shot_prompt(demo_examples, test_text, k):
    # Create prompt with k demonstration examples#
    prompt_parts = ["Here are some examples:\n"]
    
    for i, ex in enumerate(demo_examples[:k], 1):
        sentiment = label_names[ex['label']]
        prompt_parts.append(f"Sentence: {ex['text']}")
        prompt_parts.append(f"Sentiment: {sentiment}\n")
    
    prompt_parts.append(f"Sentence: {test_text}")
    prompt_parts.append(f"Sentiment:")
    
    return "\n".join(prompt_parts)

def get_balanced_examples(train_dataset, k=10):
    #Get balanced examples from training set#
    examples_by_label = {i: [] for i in range(5)}
    
    for ex in train_dataset:
        examples_by_label[ex['label']].append(ex)
    
    balanced = []
    per_class = max(1, k // 5)
    for label in range(5):
        balanced.extend(examples_by_label[label][:per_class])
    
    return balanced[:k]

def evaluate_few_shot(dataset, train_dataset, model, tokenizer, device, k=10, max_samples=None):
    #Evaluate with k-shot in-context learning#
    correct = 0
    
    eval_dataset = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    total = len(eval_dataset)
    
    demo_examples = get_balanced_examples(train_dataset, k)
    
    print("\n" + "="*60)
    print(f"TODO 2: {k}-SHOT IN-CONTEXT LEARNING")
    print("="*60)
    print(f"Using {k} training examples, evaluating on {total} samples")
    
    for example in tqdm(eval_dataset, desc=f"{k}-shot"):
        text = example['text']
        true_label = example['label']
        
        prompt = create_few_shot_prompt(demo_examples, text, k)
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        pred_label = map_prediction_to_label(prediction_text)
        
        if pred_label == true_label:
            correct += 1
    
    accuracy = correct / total
    return accuracy

# Test with different K values
print("\n" + "="*60)
print("TESTING IN-CONTEXT LEARNING")
print("="*60)

icl_results = {}
for k in [5, 10]:  # Using fewer examples for Mac speed
    icl_acc = evaluate_few_shot(sst5_dataset, sst5_train, model, tokenizer, device, k=k)
    icl_results[k] = icl_acc
    print(f"\n✓ {k}-shot Accuracy: {icl_acc:.4f} ({icl_acc*100:.2f}%)")
    improvement = (icl_acc - zero_shot_accuracy) * 100
    print(f"  Improvement: {improvement:+.2f} percentage points")



"""

(venv) ankitapatra@Ankitas-MacBook-Pro HOMEWORK2 % python hw2-2.py
============================================================
MAC-OPTIMIZED VERSION
============================================================
Using: Apple Silicon (MPS)
Device: mps
Using FLAN-T5-XL as required by assignment
⚠️  WARNING: This will be SLOW on Mac (expect 2-4 hours total)
Repo card metadata block was not found. Setting CardData to empty.
Repo card metadata block was not found. Setting CardData to empty.

Loading google/flan-t5-xl...
Loading checkpoint shards: 100%|████████████████████████████████████████████████| 2/2 [00:00<00:00, 21.73it/s]
Model loaded successfully
Validation set size: 1101
Training set size: 8544




# USE flan-t5-xl to COMPLETE the Evaluation on SST5, what's the accuracy you get?
Zero-shot accuracy on SST5 validation set: 29.43%
    Model: google/flan-t5-xl
    Device: Apple Silicon (MPS)
    Dataset: SetFit/sst5 (1,101 validation samples)

    
# Are you able to use SST5 training datset to get a better performance via In-Context Learning? Write a code to test so and report what you find.

YES - In-Context Learning significantly improves performance

Model used: google/flan-t5-xl
Device: mps (MAC)
5-shot: 48.96% 
10-shot: 49.14% 
#
"""