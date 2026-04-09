import pandas as pd
import re

# Load dataset
df = pd.read_csv('final_scam_dataset.csv')

# Indian Script Unicode Ranges
scripts = {
    'hi': r'[\u0900-\u097F]', # Devanagari (Hindi, Marathi, etc.)
    'te': r'[\u0C00-\u0C7F]', # Telugu
    'ta': r'[\u0B80-\u0BFF]', # Tamil
    'ml': r'[\u0D00-\u0D7F]', # Malayalam
    'gu': r'[\u0A80-\u0AFF]', # Gujarati
    'bn': r'[\u0980-\u09FF]', # Bengali
    'kn': r'[\u0C80-\u0CFF]', # Kannada
    'pa': r'[\u0A00-\u0A7F]', # Gurmukhi (Punjabi)
    'or': r'[\u0B00-\u0B7F]', # Odia
}

hinglish_words = {"hai", "hain", "ki", "ka", "ke", "ko", "se", "mein", "pe", "par", "bhi", "toh", "kya", "yeh", "woh", "karo", "karein", "liye", "aur", "ab", "apna", "apni", "apne", "mera", "meri", "mere", "tum", "tumhara", "tm", "aap", "aapka", "sirf", "kuch", "jab", "tab", "agar", "ho", "raha", "rahi", "rahe", "nahi", "nhi", "na", "mat", "aaj", "kal", "din", "raat", "saath", "baar", "karna", "hua", "hoga", "sakte", "chahiye", "diya", "dena", "le", "lo", "wala", "wali", "wale", "kese", "kaise", "koi", "ya", "bechne", "kharidne"}

# Convert sets to regex pattern for fast whole-word matching
hinglish_pattern = re.compile(r'\b(' + '|'.join(hinglish_words) + r')\b', flags=re.IGNORECASE)

def detect_language(row):
    text = str(row['text'])
    current_lang = row['language']
    
    # Check Indian Scripts first
    for lang_code, pattern in scripts.items():
        if re.search(pattern, text):
            return lang_code
            
    # Check for Hinglish
    # Consider it Hinglish if it has at least 2 distinct Hinglish stopwords, or if it has 1 and was not 'en' previously
    # Actually, many false 'en' could be Hinglish too, but let's check matches
    matches = hinglish_pattern.findall(text.lower())
    if len(matches) >= 2 or (len(matches) >= 1 and current_lang not in ['en']):
        return 'hinglish'
        
    # Otherwise keep the current lang
    return current_lang

df['language'] = df.apply(detect_language, axis=1)

print("New language counts:")
print(df['language'].value_counts().head(20))

df.to_csv('final_scam_dataset.csv', index=False)
print("\nUpdated final_scam_dataset.csv with new Indian language classifications!")
