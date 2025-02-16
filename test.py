def capitalize_first_letter(text):
    print("Original:", text)
    print("ASCII Codes:", [ord(c) for c in text])  # Debug each character

    if len(text) > 0:
        capitalized = text[0].upper() + text[1:]
        print("Modified:", capitalized)
        return capitalized
    return text

text = "oke, so it was u, oke?"
capitalize_first_letter(text)