def count_charater(word:str):
    result = {}
    for char in word.lower():
        result[char] = result.get(char, 0) + 1
    return result

def main():
    word = "smiles"
    assert count_charater("Baby") == {'b': 2, 'a': 1, 'y': 1}
    print(count_charater(word))

if __name__ == "__main__":
    main()
