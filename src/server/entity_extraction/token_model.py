import json
import spacy
from dotenv import load_dotenv
import pdb
import os

load_dotenv()
spacy_model = os.getenv("SPACY_MODEL")

def analyze_tokens(text: str) -> list:
    """Tokenizes a string and returns metadata about each token."""
    # Load the English NLP model
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        raise OSError("Please install the spaCy model using: python -m spacy download en_core_web_sm")
    
    # Process the text
    doc = nlp(text)
    token_list = []
    
    for token in doc:
        # Determine if it's a noun and classify its category
        # NOUN = Common Noun, PROPN = Proper Noun
        is_noun = token.pos_ in ["NOUN", "PROPN"]
        
        if is_noun:
            noun_category = "PROPER_NOUN" if token.pos_ == "PROPN" else "COMMON_NOUN"
        else:
            noun_category = "NOT_A_NOUN"
            
        # Structure the token dictionary exactly as requested
        token_data = {
            "token_string": token.text,
            "first_character_index": token.idx,
            "token_length": len(token.text),
            "token_type": {
                "part_of_speech": token.pos_,  # e.g., 'PROPN', 'VERB', 'DET', 'PUNCT'
                "is_noun": is_noun,
                "noun_category": noun_category
            }
        }
        token_list.append(token_data)
        
    return token_list

# --- Example Usage ---
if __name__ == "__main__":
    sample_text = "The Golden Gate Bridge is a massive bridge."
    
    output = analyze_tokens(sample_text)
    
    print(f"Original Text:\n{sample_text}")
    # Pretty print the first 6 tokens for readability
    print(json.dumps(output[:6], indent=4))
