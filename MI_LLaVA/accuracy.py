import json
import re

def compute_accuracy(filepath):
    total = 0
    matches = 0
    print(filepath)
    with open(filepath, "r") as file:
        for line in file:
            item = json.loads(line)
            total += 1

            question_id = item.get("question_id", "")
            text = item.get("text", "")

            # Clean the question_id: remove numbers, replace underscores with spaces, then remove spaces
            cleaned_id = re.sub(r'\d+', '', question_id)
            cleaned_id = cleaned_id.replace("_", " ").strip()

            # Clean the text: remove spaces and lowercase
            cleaned_text = text.strip()
            
            if cleaned_id == cleaned_text:
                matches += 1

    accuracy = matches / total if total > 0 else 0.0
    return accuracy, matches, total


accuracy, matches, total = compute_accuracy("./results/llava-v1.6-vicuna-7b_facescrub_test.json")
print(f"Accuracy: {accuracy:.2%} ({matches} out of {total} matched)") 
